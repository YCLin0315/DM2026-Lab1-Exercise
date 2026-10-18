from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Flags near-constant terms by variance in session.feature_matrix.

        Computes population variance for each term across all documents without
        converting the sparse matrix to a dense array. Terms with variance > threshold
        are kept, while terms with variance <= threshold are flagged. Does not modify
        session.feature_matrix or session.feature_names.

        Args:
            threshold: Variance threshold (default 0.0). Terms with variance > threshold are kept.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix loaded in session.feature_matrix."}

        X = session.feature_matrix
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        # Convert to CSC format for fast column-wise computations
        X_csc = X.tocsc()

        # Compute mean and mean of squares per column (axis=0) efficiently
        mean = np.ravel(X_csc.mean(axis=0))
        mean_sq = np.ravel(X_csc.power(2).mean(axis=0))
        variances = mean_sq - np.square(mean)
        # Numerical stability clip to prevent tiny negative variances due to floating point precision
        variances = np.maximum(variances, 0.0)

        feature_names = list(session.feature_names)
        n_terms_total = len(feature_names)

        # Strictly greater than threshold logic: keep if var > threshold, flag if var <= threshold
        kept_indices = [i for i, v in enumerate(variances) if v > threshold]
        flagged_indices = [i for i, v in enumerate(variances) if v <= threshold]

        kept_terms = [feature_names[i] for i in kept_indices]
        flagged_terms = [feature_names[i] for i in flagged_indices]

        # Build full_report DataFrame sorted descending by variance
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "status": "success",
            "threshold": threshold,
            "n_terms_total": n_terms_total,
            "n_terms_kept": len(kept_terms),
            "n_terms_flagged": len(flagged_terms),
            "flagged_terms": flagged_terms[:20],
            "kept_terms_sample": kept_terms[:20]
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=report_df
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Calculates Pearson correlation between terms and a target category.

        Measures linear correlation between feature counts and a target class binary vector.
        Terms with absolute correlation |r| > threshold are kept, and |r| <= threshold are flagged.
        Does not convert sparse matrix to dense array.

        Args:
            target_class: Target category name to compute correlation against.
            threshold: Minimum absolute correlation threshold (default 0.0). Terms with |r| > threshold are kept.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix loaded in session.feature_matrix."}

        if session.labels is None or session.categories is None:
            return {"error": "No class labels loaded in session."}

        if target_class not in session.categories:
            return {
                "error": f"Target class '{target_class}' not found in session.categories.",
                "valid_categories": list(session.categories)
            }

        X = session.feature_matrix
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        X_csc = X.tocsc()

        # Binary indicator vector y (1 if target_class, 0 otherwise)
        y = (np.asarray(session.labels) == target_class).astype(float)
        n = X_csc.shape[0]

        # Vectorized population Pearson r computation on sparse CSC matrix
        # mean_x per column
        mean_x = np.ravel(X_csc.mean(axis=0))
        # var_x per column
        mean_x_sq = np.ravel(X_csc.power(2).mean(axis=0))
        var_x = np.maximum(mean_x_sq - np.square(mean_x), 0.0)

        mean_y = np.mean(y)
        var_y = np.mean(y**2) - mean_y**2
        var_y = max(var_y, 0.0)

        # Covariance = E[XY] - E[X]E[Y]
        # E[XY] using sparse product X.T.dot(y)
        mean_xy = np.ravel(X_csc.T.dot(y)) / n
        cov_xy = mean_xy - mean_x * mean_y

        # Standard deviations
        std_x = np.sqrt(var_x)
        std_y = np.sqrt(var_y)

        # Pearson r with zero-variance protection
        denom = std_x * std_y
        r_values = np.zeros_like(cov_xy)
        valid_mask = denom > 0
        r_values[valid_mask] = cov_xy[valid_mask] / denom[valid_mask]

        feature_names = list(session.feature_names)
        n_terms_total = len(feature_names)

        abs_r = np.abs(r_values)
        kept_indices = [i for i, val in enumerate(abs_r) if val > threshold]
        flagged_indices = [i for i, val in enumerate(abs_r) if val <= threshold]

        flagged_terms = [feature_names[i] for i in flagged_indices]

        # Compute top positive and top negative terms by Pearson r
        df_corr = pd.DataFrame({
            "term": feature_names,
            "pearson_r": r_values
        })

        # Top 10 positive (highest r)
        pos_sorted = df_corr.sort_values(by="pearson_r", ascending=False).head(10)
        top_positive_terms = [
            {"term": row["term"], "pearson_r": round(float(row["pearson_r"]), 4)}
            for _, row in pos_sorted.iterrows()
        ]

        # Top 10 negative (lowest r)
        neg_sorted = df_corr.sort_values(by="pearson_r", ascending=True).head(10)
        top_negative_terms = [
            {"term": row["term"], "pearson_r": round(float(row["pearson_r"]), 4)}
            for _, row in neg_sorted.iterrows()
        ]

        # Build full_report sorted by |r| descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": r_values,
            "abs_r": abs_r
        }).sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "threshold": threshold,
            "n_terms_total": n_terms_total,
            "n_terms_kept": len(kept_indices),
            "n_terms_flagged": len(flagged_indices),
            "flagged_terms": flagged_terms[:20],
            "top_positive_terms": top_positive_terms,
            "top_negative_terms": top_negative_terms
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Calculates Spearman rank correlation between terms and a target category.

        Measures monotonic rank correlation between feature counts and a target class binary vector.
        Terms with absolute rank correlation |r_s| > threshold are kept, and |r_s| <= threshold are flagged.
        Extracts columns individually as 1D arrays to rank without converting the full matrix to dense.

        Args:
            target_class: Target category name to compute rank correlation against.
            threshold: Minimum absolute rank correlation threshold (default 0.0). Terms with |r_s| > threshold are kept.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix loaded in session.feature_matrix."}

        if session.labels is None or session.categories is None:
            return {"error": "No class labels loaded in session."}

        if target_class not in session.categories:
            return {
                "error": f"Target class '{target_class}' not found in session.categories.",
                "valid_categories": list(session.categories)
            }

        X = session.feature_matrix
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        X_csc = X.tocsc()
        n_docs, n_terms = X_csc.shape

        # Binary indicator vector y (1 if target_class, 0 otherwise)
        y = (np.asarray(session.labels) == target_class).astype(float)
        y_ranked = rankdata(y)

        feature_names = list(session.feature_names)
        r_s_values = np.zeros(n_terms)

        for i in range(n_terms):
            # Extract column i as a 1D dense array of length n_docs (efficient for column slices)
            col = X_csc.getcol(i).toarray().ravel()
            col_ranked = rankdata(col)
            
            # Compute Pearson correlation on the ranks
            # If standard deviation of ranks is 0 (constant column), correlation is 0
            std_col = np.std(col_ranked)
            std_y = np.std(y_ranked)
            if std_col > 0 and std_y > 0:
                r_s_values[i] = np.corrcoef(col_ranked, y_ranked)[0, 1]
            else:
                r_s_values[i] = 0.0

        abs_rs = np.abs(r_s_values)
        kept_indices = [i for i, val in enumerate(abs_rs) if val > threshold]
        flagged_indices = [i for i, val in enumerate(abs_rs) if val <= threshold]

        flagged_terms = [feature_names[i] for i in flagged_indices]

        # Compute top positive and top negative terms by Spearman r
        df_corr = pd.DataFrame({
            "term": feature_names,
            "spearman_r": r_s_values
        })

        # Top 10 positive (highest r_s)
        pos_sorted = df_corr.sort_values(by="spearman_r", ascending=False).head(10)
        top_positive_terms = [
            {"term": row["term"], "spearman_r": round(float(row["spearman_r"]), 4)}
            for _, row in pos_sorted.iterrows()
        ]

        # Top 10 negative (lowest r_s)
        neg_sorted = df_corr.sort_values(by="spearman_r", ascending=True).head(10)
        top_negative_terms = [
            {"term": row["term"], "spearman_r": round(float(row["spearman_r"]), 4)}
            for _, row in neg_sorted.iterrows()
        ]

        # Build full_report sorted by |r_s| descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "spearman_r": r_s_values,
            "abs_rs": abs_rs
        }).sort_values(by="abs_rs", ascending=False).drop(columns=["abs_rs"]).reset_index(drop=True)

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "status": "success",
            "target_class": target_class,
            "threshold": threshold,
            "n_terms_total": n_terms,
            "n_terms_kept": len(kept_indices),
            "n_terms_flagged": len(flagged_indices),
            "flagged_terms": flagged_terms[:20],
            "top_positive_terms": top_positive_terms,
            "top_negative_terms": top_negative_terms
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df
        )

        return summary

    return [
        variance_filter_tool,
        pearson_filter_tool,
        spearman_filter_tool
    ]
