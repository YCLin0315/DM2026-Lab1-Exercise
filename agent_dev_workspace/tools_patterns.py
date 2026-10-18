import os
import tempfile
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Literal
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

import PAMI.extras.convert.DF2DB as DF2DB
import PAMI.frequentPattern.basic.FPGrowth as FPGrowth
import PAMI.frequentPattern.topk.FAE as FAE
import PAMI.frequentPattern.maximal.MaxFPGrowth as MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = 100,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mine frequent patterns/itemsets for a specific document category.

        Filters category-local vocabulary by TF-IDF, variance, or term frequency,
        converts the filtered category text to a transactional database, and mines
        frequent patterns using FPGrowth, FAE (Top-K), or MaxFPGrowth.

        Args:
            category_name: The category/class name to mine (must exist in session.dataframe).
            filtering_method: Metric to filter terms ('tfidf', 'variance', or 'term_frequency'). Defaults to 'tfidf'.
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth'). Defaults to 'fpgrowth'.
            min_sup: Minimum support count (defaults to 100).
            k: Top-k number of patterns to keep (required for 'topk').

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm,
            n_patterns, and the top 10 mined patterns (with full results in full_report).
        """
        # 1. Check for loaded dataframe
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded in session"}

        if "category_name" not in session.dataframe.columns:
            return {"error": "session.dataframe does not contain 'category_name' column"}

        # 2. Validate category_name
        valid_categories = sorted(session.dataframe["category_name"].dropna().unique().tolist())
        if category_name not in valid_categories:
            return {
                "error": f"Category '{category_name}' not found. Valid categories: {valid_categories}"
            }

        # 3. Filter dataframe to category rows
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        texts = cat_df["text"].dropna().tolist()
        n_docs = len(texts)

        result_id = session.next_result_id("mine_patterns")

        if not texts:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no documents found for category",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 5. Fit category-local CountVectorizer with English stop words
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts)
        except ValueError:
            # Vectorizer raises ValueError if all words are stop words or empty
            dtm = None

        if dtm is None or dtm.shape[1] == 0:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived stop_words removal",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Calculate average distinct terms per document after stop word removal
        avg_distinct_terms = dtm.getnnz(axis=1).mean()
        if avg_distinct_terms > 50:
            min_allowed_sup = max(1, min(100, round(0.2 * n_docs)))
        else:
            min_allowed_sup = 1

        # 4. Validate algorithm parameters
        if algorithm in ("fpgrowth", "maxfpgrowth"):
            if min_sup is None or min_sup < min_allowed_sup:
                return {
                    "error": f"min_sup must be at least {min_allowed_sup} for category '{category_name}' (minimum support threshold)"
                }
        elif algorithm == "topk":
            if k is None or k < 1:
                return {"error": "k (int >= 1) is required for algorithm 'topk'"}

        feature_names = np.array(vectorizer.get_feature_names_out())
        dense_dtm = dtm.toarray()

        # 6. Vocabulary filtering (inclusive bounds >= and <=)
        if filtering_method == "variance":
            variances = np.var(dense_dtm, axis=0)
            p5 = np.percentile(variances, 5)
            p95 = np.percentile(variances, 95)
            keep_mask = (variances >= p5) & (variances <= p95)
        elif filtering_method == "term_frequency":
            term_sums = np.sum(dense_dtm, axis=0)
            p5 = np.percentile(term_sums, 5)
            p95 = np.percentile(term_sums, 95)
            keep_mask = (term_sums >= p5) & (term_sums <= p95)
        elif filtering_method == "tfidf":
            tfidf_mat = TfidfTransformer().fit_transform(dtm).toarray()
            mean_tfidf = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            keep_mask = mean_tfidf >= p20
        else:
            return {"error": f"Invalid filtering_method: {filtering_method}"}

        surviving_features = feature_names[keep_mask]
        surviving_dtm = dense_dtm[:, keep_mask]

        # 7. Encoding guard: drop non-ASCII features
        ascii_mask = np.array([f.isascii() for f in surviving_features], dtype=bool)
        if not np.all(ascii_mask):
            surviving_features = surviving_features[ascii_mask]
            surviving_dtm = surviving_dtm[:, ascii_mask]

        if surviving_features.size == 0 or surviving_dtm.shape[1] == 0:
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 8. Convert to binary presence DataFrame (1 if count >= 1 else 0)
        binary_dtm = (surviving_dtm >= 1).astype(int)
        df_for_pami = pd.DataFrame(binary_dtm, columns=surviving_features)

        # 9. Create transactional DB file using PAMI DF2DB
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        temp_path = temp_file.name
        temp_file.close()

        raw_patterns_df = None
        try:
            # Convert DF to transactional database text file
            converter = DF2DB.DF2DB(df_for_pami)
            converter.convert2TransactionalDatabase(temp_path, ">=", 1)

            # 10. Run chosen PAMI algorithm
            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(temp_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                miner = FAE.FAE(temp_path, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(temp_path, minSup=min_sup, sep="\t")

            miner.mine()
            raw_patterns_df = miner.getPatternsAsDataFrame()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # 11. Parse patterns
        parsed_patterns = []
        if raw_patterns_df is not None and not raw_patterns_df.empty:
            for _, row in raw_patterns_df.iterrows():
                pat_str = str(row["Patterns"])
                sup_val = int(row["Support"])
                # Split pattern string by tab into term lists
                terms = [t.strip() for t in pat_str.split("\t") if t.strip()]
                if terms:
                    parsed_patterns.append({"pattern": terms, "support": sup_val})

        # Sort patterns by support descending
        parsed_patterns.sort(key=lambda x: x["support"], reverse=True)

        # Build full_report DataFrame
        report_rows = [
            {"pattern": " ".join(item["pattern"]), "support": item["support"]}
            for item in parsed_patterns
        ]
        full_report = pd.DataFrame(report_rows)

        # Store in session.artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = parsed_patterns

        # Summary capping (top 10 patterns)
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(parsed_patterns),
            "patterns": parsed_patterns[:10],
        }

        session.store_result(
            tool_name="mine_patterns_tool",
            args={
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "min_sup": min_sup,
                "k": k,
            },
            summary=summary,
            full_report=full_report,
        )

        return summary

    return [mine_patterns_tool]
