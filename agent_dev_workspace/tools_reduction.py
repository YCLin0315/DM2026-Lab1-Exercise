"""Dimensionality reduction tools for text data."""

from typing import Dict, List, Literal, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
import umap


def make_tools(session):
    """Create dimensionality reduction tools bound to a shared session."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
    ) -> dict:
        """Reduce the document-term matrix to 2D using PCA, t-SNE, or UMAP and plot the result.

        Args:
            method: Reduction technique ('pca', 'tsne', or 'umap'). Default is 'pca'.
            random_state: Seed for reproducible projections. Default is 42.
            perplexity: Perplexity parameter for t-SNE (must be less than n_documents). Default is 30.0.
            n_neighbors: Number of neighbors for UMAP (must be less than n_documents). Default is 15.

        Returns:
            A summary dictionary with reduction metadata, coordinates preview, and explained variance (if PCA).
        """
        if session.feature_matrix is None:
            return {"error": "Feature matrix is not built yet. Run build_dtm_tool first."}

        n_docs = session.feature_matrix.shape[0]
        if n_docs == 0:
            return {"error": "Feature matrix contains 0 documents."}

        if session.labels is None or len(session.labels) == 0:
            return {"error": "Labels are not set. Document labels must be set before reducing dimensions."}

        if len(session.labels) != n_docs:
            return {"error": f"Labels length ({len(session.labels)}) does not match document count ({n_docs})."}

        if method == "tsne" and perplexity >= n_docs:
            return {"error": f"perplexity ({perplexity}) must be smaller than the number of documents ({n_docs})."}

        if method == "umap" and n_neighbors >= n_docs:
            return {"error": f"n_neighbors ({n_neighbors}) must be smaller than the number of documents ({n_docs})."}

        # Select top 100 terms by variance computed directly on the sparse matrix
        # Var(X) = E[X^2] - (E[X])^2
        mean_x = np.asarray(session.feature_matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(session.feature_matrix.power(2).mean(axis=0)).ravel()
        variances = mean_sq - (mean_x ** 2)

        n_terms = session.feature_matrix.shape[1]
        k = min(100, n_terms)
        top_k_indices = np.argsort(variances)[::-1][:k]

        # Convert only the selected top-k columns to dense
        dense_subset = session.feature_matrix[:, top_k_indices].toarray()

        explained_variance_ratio: Optional[List[float]] = None

        if method == "pca":
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(dense_subset)
            explained_variance_ratio = [round(float(v), 4) for v in reducer.explained_variance_ratio_]
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
            coords = reducer.fit_transform(dense_subset)
        elif method == "umap":
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(dense_subset)
        else:
            return {"error": f"Unknown method: {method}. Must be 'pca', 'tsne', or 'umap'."}

        # Build 2D scatter plot colored by label
        fig, ax = plt.subplots(figsize=(8, 6))
        labels_arr = np.asarray(session.labels)
        unique_labels = np.unique(labels_arr)

        for lab in unique_labels:
            mask = labels_arr == lab
            ax.scatter(coords[mask, 0], coords[mask, 1], label=str(lab), alpha=0.7)

        ax.set_title(f"2D Reduction using {method.upper()}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Category")
        fig.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        # Store artifacts and report
        df_coords = pd.DataFrame(coords, columns=["dim_1", "dim_2"])
        df_coords["category"] = labels_arr
        session.artifacts["reduced_coordinates"] = df_coords

        result_id = session.next_result_id("reduction")
        preview = [
            [round(float(c[0]), 4), round(float(c[1]), 4)]
            for c in coords[:5]
        ]

        summary = {
            "result_id": result_id,
            "method": method,
            "n_samples": n_docs,
            "coordinates_preview": preview,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        args = {
            "method": method,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors,
        }
        session.store_result("reduce_dimensions_tool", args, summary, full_report=df_coords)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes category labels from session.dataframe["category_name"].

        Uses scikit-learn's LabelBinarizer to encode labels in alphabetical order.
        Adds the binarized labels as a new column 'bin_category' in session.dataframe.

        Returns:
            A summary dictionary with categories, document counts per category, and encoding notes.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session.dataframe."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "Column 'category_name' not found in session.dataframe."}

        df = session.dataframe.copy()
        labels = df["category_name"].values

        encoder = LabelBinarizer()
        binary_matrix = encoder.fit_transform(labels)
        classes = list(encoder.classes_)

        # Store each row as a plain Python list of ints
        bin_lists = binary_matrix.tolist()
        df["bin_category"] = bin_lists
        session.dataframe = df

        counts = df["category_name"].value_counts().to_dict()
        category_counts_list = [{"category": cat, "count": int(cnt)} for cat, cnt in counts.items()]

        report_df = pd.DataFrame([
            {"category": cat, "count": int(cnt)}
            for cat, cnt in counts.items()
        ])

        note = (
            "Single binary column (1 = second category alphabetically)"
            if len(classes) == 2
            else f"One-hot encoded across {len(classes)} categories."
        )

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "status": "success",
            "classes": classes,
            "category_counts": category_counts_list,
            "note": note
        }

        session.store_result(
            "binarize_labels_tool",
            {},
            summary,
            full_report=report_df
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
