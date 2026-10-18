from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1],
        min_df: Union[int, float] = 1,
        max_df: Union[int, float] = 1.0
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from document text in session.dataframe.

        Args:
            max_features: Maximum number of features (terms) to keep in vocabulary.
            ngram_range: Lower and upper boundary of n-gram range, e.g. [1, 1] for unigrams.
            min_df: Minimum document frequency threshold (count or proportion).
            max_df: Maximum document frequency threshold (count or proportion).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}

        if "text" not in session.dataframe.columns:
            return {"error": "Missing 'text' column in session.dataframe."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "Missing 'category_name' column in session.dataframe."}

        # Replace NaN in text with empty string to prevent vectorizer failure
        texts = session.dataframe["text"].fillna("").astype(str)

        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=tuple(ngram_range),
            min_df=min_df,
            max_df=max_df
        )

        try:
            X = vectorizer.fit_transform(texts)
        except Exception as e:
            return {"error": f"Vectorization failed: {str(e)}"}

        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = X
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        labels = session.dataframe["category_name"].to_numpy()
        session.set_labels(labels)

        n_rows, n_cols = X.shape
        non_zero = int(X.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = float(100.0 * (1.0 - non_zero / total_elements)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result("build_dtm_tool", {"max_features": max_features, "ngram_range": ngram_range, "min_df": min_df, "max_df": max_df}, summary)

        return summary

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = 30
    ) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in the global document-term matrix.

        Args:
            top_n: Number of top frequent terms to return in the summary (default 30).
                   If None, all terms are returned in the summary.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not built yet. Run build_dtm_tool first."}

        # Sum term counts across all documents
        col_sums = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        
        # Sort terms descending by frequency
        sorted_indices = np.argsort(-col_sums)
        
        full_terms = [session.feature_names[i] for i in sorted_indices]
        full_counts = [int(col_sums[i]) for i in sorted_indices]

        # Full report DataFrame with 'term' and 'frequency' columns for visualize_result_tool
        full_report = pd.DataFrame({
            "term": full_terms,
            "frequency": full_counts
        })

        total_terms = len(full_terms)

        # Slice for summary dict
        if top_n is not None and top_n > 0:
            summary_terms = full_terms[:top_n]
            summary_counts = full_counts[:top_n]
        else:
            summary_terms = full_terms
            summary_counts = full_counts

        term_frequencies = dict(zip(summary_terms, summary_counts))

        result_id = session.next_result_id("term_freq")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_terms": total_terms,
            "returned_terms": len(term_frequencies),
            "term_frequencies": term_frequencies
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary, full_report=full_report)

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Generates a raw positional slice heatmap of the global document-term matrix.

        Args:
            n_terms: Number of terms (columns) to include from the DTM (default 20).
            n_documents: Number of documents (rows) to include from the DTM (default 20).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "Document-term matrix not built yet. Run build_dtm_tool first."}

        n_docs_slice = min(n_documents, session.feature_matrix.shape[0])
        n_terms_slice = min(n_terms, session.feature_matrix.shape[1])

        # Slice sub-block of matrix
        matrix_slice = session.feature_matrix[0:n_docs_slice, 0:n_terms_slice].toarray()
        terms_included = session.feature_names[0:n_terms_slice]
        doc_labels = [f"Doc {i}" for i in range(n_docs_slice)]

        # Create plot explicitly
        fig, ax = plt.subplots(figsize=(max(6, n_terms_slice * 0.8), max(4, n_docs_slice * 0.5)))
        sns.heatmap(
            matrix_slice,
            xticklabels=terms_included,
            yticklabels=doc_labels,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            ax=ax
        )
        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Assign figure to session.pending_figure
        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_docs_slice,
            "n_terms": n_terms_slice,
            "documents": doc_labels,
            "terms": terms_included,
            "matrix_slice": matrix_slice.tolist()
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
