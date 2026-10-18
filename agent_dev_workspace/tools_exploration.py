from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_tools(session):
    @tool
    def cosine_similarity_tool(doc_a: int, doc_b: int) -> dict:
        """Compute the cosine similarity between two documents from their DTM row indices.

        Args:
            doc_a: Row index of the first document.
            doc_b: Row index of the second document.
        """
        if session.dataframe is None or session.feature_matrix is None:
            return {"error": "Dataset and DTM must be loaded/built first."}
        
        n_rows = session.feature_matrix.shape[0]
        if doc_a < 0 or doc_b < 0 or doc_a >= n_rows or doc_b >= n_rows:
            return {"error": f"Document indices must be between 0 and {n_rows - 1}."}
        
        vec_a = session.feature_matrix[doc_a]
        vec_b = session.feature_matrix[doc_b]
        
        sim = float(cosine_similarity(vec_a, vec_b)[0, 0])
        
        result_id = session.next_result_id("cosine")
        summary = {
            "status": "success",
            "result_id": result_id,
            "doc_a": doc_a,
            "doc_b": doc_b,
            "similarity": sim
        }
        session.store_result("cosine_similarity_tool", {"doc_a": doc_a, "doc_b": doc_b}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Compute the feature-vs-feature Pearson correlation matrix over the top-20 terms by variance, plot a heatmap, and store the result."""
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "DTM must be built first."}
        
        mat = session.feature_matrix
        names = session.feature_names
        
        # Compute variance efficiently per column for sparse matrix
        mean_vals = np.array(mat.mean(axis=0)).ravel()
        mean_sq_vals = np.array(mat.power(2).mean(axis=0)).ravel()
        variances = mean_sq_vals - mean_vals ** 2
        
        # Select top up-20 terms by variance
        n_top = min(20, len(names))
        top_indices = np.argsort(variances)[::-1][:n_top]
        
        selected_names = [names[i] for i in top_indices]
        
        sub_mat = mat[:, top_indices].toarray()
        corr_matrix = np.corrcoef(sub_mat, rowvar=False)
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]])
        elif corr_matrix.ndim == 1:
            corr_matrix = np.array([[1.0]])
        
        # Handle potential NaNs
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        # Ensure diagonal is 1.0
        np.fill_diagonal(corr_matrix, 1.0)
        
        rounded_corr = np.round(corr_matrix, 4).tolist()
        
        # Build heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=selected_names, yticklabels=selected_names, ax=ax, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("Feature Correlation Matrix (Top Terms by Variance)")
        plt.tight_layout()
        
        session.pending_figure = fig
        plt.close(fig)
        
        result_id = session.next_result_id("feature_corr")
        summary = {
            "status": "success",
            "result_id": result_id,
            "terms": selected_names,
            "correlation_matrix": rounded_corr
        }
        
        import pandas as pd
        full_report = pd.DataFrame(rounded_corr, index=selected_names, columns=selected_names)
        
        session.store_result("feature_correlation_matrix_tool", {}, summary, full_report=full_report)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
