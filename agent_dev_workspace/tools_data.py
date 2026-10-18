from typing import Dict, Any, Optional
import os
import pathlib
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def check_missing_tool(
        text_column: str = "text"
    ) -> Dict[str, Any]:
        """Checks for missing, NaN, or whitespace-only empty text values in session.dataframe.

        Args:
            text_column: Name of the column containing document text (default 'text').
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in session.dataframe."}

        df = session.dataframe
        # Identify null/NaN or whitespace-only empty strings
        is_missing = df[text_column].isna() | (df[text_column].astype(str).str.strip() == "")

        total_rows = len(df)
        missing_count = int(is_missing.sum())
        missing_indices = df.index[is_missing].tolist()

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_indices": missing_indices
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)

        return summary

    @tool
    def check_duplicates_tool(
        text_column: str = "text",
        drop: bool = False
    ) -> Dict[str, Any]:
        """Checks for (and optionally drops) duplicate documents in session.dataframe.

        Args:
            text_column: Column name containing document text (default 'text').
            drop: If True, drops all copies of duplicated documents (keep=False) and resets index.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in session.dataframe."}

        total_rows_before = len(session.dataframe)

        # duplicate_count: count of extra copies (keep='first')
        extra_duplicates_mask = session.dataframe.duplicated(subset=[text_column], keep="first")
        duplicate_count = int(extra_duplicates_mask.sum())

        # duplicate_group_indices: all indices belonging to duplicate groups (keep=False)
        all_duplicates_mask = session.dataframe.duplicated(subset=[text_column], keep=False)
        duplicate_group_indices = session.dataframe.index[all_duplicates_mask].tolist()

        if drop and duplicate_count > 0:
            # Drop all copies of duplicated rows matching keep=False contract and reset index
            session.dataframe = session.dataframe.drop_duplicates(subset=[text_column], keep=False).reset_index(drop=True)
            
            # Clear DTM and labels state so they stay aligned with session.dataframe
            session.feature_matrix = None
            session.feature_names = None
            session.labels = None
            session.categories = []
            session.artifacts.pop("count_vectorizer", None)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows_before": total_rows_before,
            "duplicate_count": duplicate_count,
            "duplicate_group_indices": duplicate_group_indices,
            "dropped": drop,
            "total_rows_after": len(session.dataframe)
        }

        session.store_result("check_duplicates_tool", {"text_column": text_column, "drop": drop}, summary)

        return summary

    @tool
    def tokenize_tool(
        text_column: str = "text",
        output_column: str = "unigrams"
    ) -> Dict[str, Any]:
        """Tokenizes text documents into lowercased unigrams using NLTK.

        Args:
            text_column: Column name containing text documents (default 'text').
            output_column: Column name to store tokenized lists (default 'unigrams').
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in session.dataframe."}

        # Ensure nltk data is downloaded inside tokenize_tool
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
                nltk.download("punkt", quiet=True)
            except Exception:
                pass

        def _tokenize(val):
            if pd.isna(val) or str(val).strip() == "":
                return []
            return word_tokenize(str(val).lower())

        # Copy dataframe, perform assignment, then reassign to session.dataframe
        df_copy = session.dataframe.copy()
        df_copy[output_column] = df_copy[text_column].apply(_tokenize)
        session.dataframe = df_copy

        total_documents = len(session.dataframe)
        total_tokens = sum(len(tokens) for tokens in session.dataframe[output_column])
        avg_tokens_per_doc = float(total_tokens / total_documents) if total_documents > 0 else 0.0

        # Truncate sample_tokens to first 30 tokens of first 3 documents
        sample_tokens = [
            tokens[:30] for tokens in session.dataframe[output_column].head(3).tolist()
        ]

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_documents": total_documents,
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens_per_doc, 4),
            "sample_tokens": sample_tokens
        }

        session.store_result("tokenize_tool", {"text_column": text_column, "output_column": output_column}, summary)

        return summary

    @tool
    def sample_data_tool(
        n: int = 5,
        random_state: Optional[int] = 42
    ) -> Dict[str, Any]:
        """Subsamples n rows from session.dataframe and updates session state.

        Args:
            n: Number of rows to sample (default 5).
            random_state: Random seed for reproducibility (default 42).
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}

        total_rows_before = len(session.dataframe)

        if n <= 0:
            return {"error": f"Sample size n must be positive, got {n}."}

        if n > total_rows_before:
            return {"error": f"Sample size n={n} exceeds dataset size of {total_rows_before} rows."}

        # Subsample rows using pandas sample
        sampled_df = session.dataframe.sample(n=n, random_state=random_state)
        original_indices = sampled_df.index.tolist()

        # Build report dataframe with original_index column before resetting index
        full_report_df = sampled_df.copy()
        full_report_df["original_index"] = original_indices

        # Update session.dataframe with fresh index
        session.dataframe = sampled_df.reset_index(drop=True)

        # Clear DTM and labels state so downstream tools stay aligned
        session.feature_matrix = None
        session.feature_names = None
        session.labels = None
        session.categories = []
        session.artifacts.pop("count_vectorizer", None)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows_before": total_rows_before,
            "sampled_rows": n,
            "total_rows_after": len(session.dataframe),
            "sampled_indices": original_indices[:10]
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "random_state": random_state},
            summary,
            full_report=full_report_df
        )

        return summary

    @tool
    def list_files_tool(
        subdirectory: str = ""
    ) -> Dict[str, Any]:
        """Lists files and directories in the repository root or a given subdirectory.

        Args:
            subdirectory: Relative path to a subdirectory within the repository root (default "").
        """
        repo_root = pathlib.Path.cwd().resolve()
        target_path = (repo_root / subdirectory).resolve()

        # Security check: ensure target_path is within repo_root
        if repo_root not in target_path.parents and target_path != repo_root:
            return {"error": f"Path '{subdirectory}' resolves outside repository root."}

        if not target_path.exists() or not target_path.is_dir():
            return {"error": f"Directory '{subdirectory}' does not exist or is not a directory."}

        dirs = []
        files = []
        full_report_rows = []

        for item in target_path.iterdir():
            if item.name.startswith("."):
                continue
            if item.is_dir():
                dirs.append(item.name)
                full_report_rows.append({"name": item.name, "type": "directory"})
            elif item.is_file():
                files.append(item.name)
                full_report_rows.append({"name": item.name, "type": "file"})

        dirs.sort()
        files.sort()

        full_report_df = pd.DataFrame(full_report_rows)
        if not full_report_df.empty:
            full_report_df.sort_values(by="name", inplace=True)
            full_report_df.reset_index(drop=True, inplace=True)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "status": "success",
            "subdirectory": subdirectory,
            "directories": dirs[:20],
            "files": files[:20]
        }

        session.store_result(
            "list_files_tool",
            {"subdirectory": subdirectory},
            summary,
            full_report=full_report_df
        )

        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text"
    ) -> Dict[str, Any]:
        """Computes summary statistics on text character length and generates a boxplot by category.

        Args:
            text_column: Column name containing text documents (default 'text').
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session.dataframe."}

        if text_column not in session.dataframe.columns:
            return {"error": f"Column '{text_column}' not found in session.dataframe."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "Column 'category_name' not found in session.dataframe."}

        # Calculate character length without modifying session.dataframe
        df_temp = session.dataframe.copy()
        df_temp["text_length"] = df_temp[text_column].apply(lambda x: len(str(x)) if pd.notna(x) else 0)

        # Helper to convert describe() dict values, replacing NaN with None for valid JSON
        def _clean_stats(stats_dict: dict) -> dict:
            cleaned = {}
            for k, v in stats_dict.items():
                if pd.isna(v):
                    cleaned[k] = None
                elif isinstance(v, (np.floating, float)):
                    cleaned[k] = round(float(v), 4)
                elif isinstance(v, (np.integer, int)):
                    cleaned[k] = int(v)
                else:
                    cleaned[k] = v
            return cleaned

        overall_desc = df_temp["text_length"].describe().to_dict()
        overall_stats = _clean_stats(overall_desc)

        by_category_stats = {}
        grouped = df_temp.groupby("category_name")["text_length"]
        for i, (cat, group) in enumerate(grouped):
            if i >= 20:
                break
            by_category_stats[str(cat)] = _clean_stats(group.describe().to_dict())

        # Generate boxplot figure
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_temp, x="category_name", y="text_length", ax=ax)
        ax.set_xlabel("Category")
        ax.set_ylabel("Text Length (characters)")
        ax.set_title("Text Length Distribution by Category")
        plt.tight_layout()

        # Set pending figure and close fig locally
        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "overall_stats": overall_stats,
            "by_category_stats": by_category_stats
        }

        session.store_result(
            "describe_data_tool",
            {"text_column": text_column},
            summary
        )

        return summary

    @tool
    def inspect_data_tool(
        n: int = 10
    ) -> Dict[str, Any]:
        """Peeks at the first n rows of session.dataframe to inspect category and text contents.

        Args:
            n: Number of initial rows to inspect (default 10, max 30).
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded in session.dataframe."}

        if n < 1 or n > 30:
            return {"error": f"n must be between 1 and 30, got {n}."}

        df = session.dataframe
        total_rows_in_dataframe = len(df)
        head_df = df.head(n)

        inspected_rows = []
        for idx, row in head_df.iterrows():
            cat = row["category_name"] if "category_name" in row and pd.notna(row["category_name"]) else "N/A"
            raw_text = str(row["text"]) if "text" in row and pd.notna(row["text"]) else ""
            
            truncated_text = raw_text[:100] + "..." if len(raw_text) > 100 else raw_text
            inspected_rows.append({
                "original_index": int(idx),
                "category_name": str(cat),
                "text": truncated_text
            })

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows_in_dataframe": total_rows_in_dataframe,
            "inspected_rows_count": len(inspected_rows),
            "rows": inspected_rows
        }

        full_report_df = pd.DataFrame(inspected_rows)

        session.store_result(
            "inspect_data_tool",
            {"n": n},
            summary,
            full_report=full_report_df
        )

        return summary

    return [
        check_missing_tool,
        check_duplicates_tool,
        tokenize_tool,
        sample_data_tool,
        list_files_tool,
        describe_data_tool,
        inspect_data_tool
    ]
