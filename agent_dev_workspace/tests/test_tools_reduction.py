"""Tests for tools_reduction.py (reduce_dimensions_tool)."""

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def _setup_session_with_dtm():
    """Helper to set up a clean session loaded with sample_fixture.csv and fitted DTM."""
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade if t.name == "load_dataset_tool")
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    # Load 8 fixture rows:
    # Row 0: "always alpha alpha alpha beta",     label="catA"
    # Row 1: "always alpha alpha beta",           label="catA"
    # Row 2: "always alpha beta",                 label="catA"
    # Row 3: "always gamma gamma gamma delta",    label="catB"
    # Row 4: "always gamma gamma delta",          label="catB"
    # Row 5: "always gamma delta",                label="catB"
    # Row 6: "always gamma delta",                label="catB"
    # Row 7: "",                                  label="catA"
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })
    build_dtm_tool.invoke({})
    return session


def test_reduce_dimensions_pca_known_answer_7():
    """Known Answer #7: PCA reduction to 2 components.

    Derivation from TEST_FIXTURE.md #7:
    The fixture documents have 5 terms: alpha, always, beta, delta, gamma.
    Row counts:
    Doc 0: alpha=3, always=1, beta=1, delta=0, gamma=0 -> PC1= 2.3669, PC2= 0.8871
    Doc 1: alpha=2, always=1, beta=1, delta=0, gamma=0 -> PC1= 1.7039, PC2= 0.2642
    Doc 2: alpha=1, always=1, beta=1, delta=0, gamma=0 -> PC1= 1.0408, PC2=-0.3587
    Doc 3: alpha=0, always=1, beta=0, delta=1, gamma=3 -> PC1=-2.0759, PC2= 1.0129
    Doc 4: alpha=0, always=1, beta=0, delta=1, gamma=2 -> PC1=-1.4554, PC2= 0.3405
    Doc 5: alpha=0, always=1, beta=0, delta=1, gamma=1 -> PC1=-0.8349, PC2=-0.3319
    Doc 6: alpha=0, always=1, beta=0, delta=1, gamma=1 -> PC1=-0.8349, PC2=-0.3319
    Doc 7: alpha=0, always=0, beta=0, delta=0, gamma=0 -> PC1= 0.0894, PC2=-1.4822
    Explained variance ratio: [0.7532, 0.1965]
    """
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({"method": "pca"})
    assert "error" not in res
    assert res["method"] == "pca"
    assert res["n_samples"] == 8

    # Check explained variance ratio: [0.7532, 0.1965]
    expected_evr = [0.7532, 0.1965]
    assert np.allclose(res["explained_variance_ratio"], expected_evr, atol=1e-3)

    # Check full coordinates stored in artifacts
    coords_df = session.artifacts.get("reduced_coordinates")
    assert coords_df is not None
    coords = coords_df[["dim_1", "dim_2"]].values

    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ])
    assert np.allclose(coords, expected_coords, atol=1e-3)

    # Verify pending_figure was set
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne_known_answer_8():
    """Known Answer #8: t-SNE reduction with perplexity=2, random_state=42.

    Derivation from TEST_FIXTURE.md #8:
    Doc 0: Dim 1= 1330.9991, Dim 2=   32.1039
    Doc 1: Dim 1=  951.2459, Dim 2=  393.5588
    Doc 2: Dim 1=  530.8209, Dim 2=  682.9791
    Doc 3: Dim 1=-1558.6448, Dim 2= -314.1625
    Doc 4: Dim 1= -915.5153, Dim 2= -346.3907
    Doc 5: Dim 1= -280.0313, Dim 2=-1261.2115
    Doc 6: Dim 1= -280.0313, Dim 2=-1261.2115
    Doc 7: Dim 1= -701.3674, Dim 2= 1465.3973
    """
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42,
    })
    assert "error" not in res
    assert res["method"] == "tsne"

    coords_df = session.artifacts.get("reduced_coordinates")
    assert coords_df is not None
    coords = coords_df[["dim_1", "dim_2"]].values

    expected_coords = np.array([
        [1330.9991, 32.1039],
        [951.2459, 393.5588],
        [530.8209, 682.9791],
        [-1558.6448, -314.1625],
        [-915.5153, -346.3907],
        [-280.0313, -1261.2115],
        [-280.0313, -1261.2115],
        [-701.3674, 1465.3973],
    ])
    assert np.allclose(coords, expected_coords, atol=1e-3)
    assert session.pending_figure is not None


def test_reduce_dimensions_umap_known_answer_9():
    """Known Answer #9: UMAP reduction with n_neighbors=3, random_state=42.

    Derivation from TEST_FIXTURE.md #9:
    Doc 0: Dim 1=  6.6811, Dim 2= -6.8218
    Doc 1: Dim 1=  6.9907, Dim 2= -7.2245
    Doc 2: Dim 1=  7.7022, Dim 2= -7.3712
    Doc 3: Dim 1= 10.2228, Dim 2= -6.2310
    Doc 4: Dim 1=  9.8759, Dim 2= -5.6537
    Doc 5: Dim 1=  9.3831, Dim 2= -6.5748
    Doc 6: Dim 1=  9.2153, Dim 2= -5.8297
    Doc 7: Dim 1=  8.5860, Dim 2= -7.2622
    """
    session = _setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42,
    })
    assert "error" not in res
    assert res["method"] == "umap"

    coords_df = session.artifacts.get("reduced_coordinates")
    assert coords_df is not None
    coords = coords_df[["dim_1", "dim_2"]].values

    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ])
    assert np.allclose(coords, expected_coords, atol=1e-3)
    assert session.pending_figure is not None


def test_reduce_dimensions_validation_errors():
    """Check validation error conditions before modifying any state."""
    # 1. Feature matrix not built yet
    session1 = SessionState()
    tools1 = make_reduction_tools(session1)
    reduce_tool1 = tools1[0]
    res1 = reduce_tool1.invoke({})
    assert "error" in res1
    assert "Feature matrix is not built" in res1["error"]

    # 2. Labels not set
    session2 = _setup_session_with_dtm()
    session2.labels = None
    tools2 = make_reduction_tools(session2)
    reduce_tool2 = tools2[0]
    res2 = reduce_tool2.invoke({})
    assert "error" in res2
    assert "Labels are not set" in res2["error"]

    # 3. Perplexity >= n_documents (8 documents in fixture)
    session3 = _setup_session_with_dtm()
    tools3 = make_reduction_tools(session3)
    reduce_tool3 = tools3[0]
    res3 = reduce_tool3.invoke({"method": "tsne", "perplexity": 8.0})
    assert "error" in res3
    assert "perplexity" in res3["error"]

    # 4. n_neighbors >= n_documents (8 documents in fixture)
    session4 = _setup_session_with_dtm()
    tools4 = make_reduction_tools(session4)
    reduce_tool4 = tools4[0]
    res4 = reduce_tool4.invoke({"method": "umap", "n_neighbors": 8})
    assert "error" in res4
    assert "n_neighbors" in res4["error"]


def test_binarize_labels_tool_sample_fixture():
    """Test binarize_labels_tool with sample_fixture.csv (two categories: catA, catB)."""
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Set placeholder feature_matrix to verify it's unchanged afterwards
    session.feature_matrix = "placeholder_dtm"

    tools = make_reduction_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})
    assert res.get("status") == "success"
    assert res["classes"] == ["catA", "catB"]
    assert "single binary column" in res["note"].lower()

    # Check document counts per category: 4 for catA, 4 for catB
    counts_dict = {item["category"]: item["count"] for item in res["category_counts"]}
    assert counts_dict == {"catA": 4, "catB": 4}

    # Verify session.dataframe has bin_category column and 8 rows
    df = session.dataframe
    assert len(df) == 8
    assert "bin_category" in df.columns

    # Rows 0, 1, 2, 7 are catA -> [0]
    # Rows 3, 4, 5, 6 are catB -> [1]
    expected_bins = [[0], [0], [0], [1], [1], [1], [1], [0]]
    assert df["bin_category"].tolist() == expected_bins

    # Verify DTM state was not cleared (since row count didn't change)
    assert session.feature_matrix == "placeholder_dtm"


def test_binarize_labels_tool_three_categories():
    """Test binarize_labels_tool with 3 categories on a custom small DataFrame."""
    import pandas as pd
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["doc1", "doc2", "doc3"],
        "category_name": ["apple", "banana", "cherry"]
    })

    tools = make_reduction_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    res = binarize_tool.invoke({})
    assert res.get("status") == "success"
    assert res["classes"] == ["apple", "banana", "cherry"]
    assert "one-hot encoded across 3 categories" in res["note"].lower()

    # Alphabetical order: apple, banana, cherry
    # apple -> [1, 0, 0]
    # banana -> [0, 1, 0]
    # cherry -> [0, 0, 1]
    expected_bins = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    assert session.dataframe["bin_category"].tolist() == expected_bins


def test_binarize_labels_tool_errors():
    """Test error conditions for binarize_labels_tool."""
    import pandas as pd
    session = SessionState()
    tools = make_reduction_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    # 1. No dataframe loaded
    res1 = binarize_tool.invoke({})
    assert "error" in res1

    # 2. Missing category_name column
    session.dataframe = pd.DataFrame({"text": ["abc"]})
    res2 = binarize_tool.invoke({})
    assert "error" in res2
    assert "category_name" in res2["error"]
