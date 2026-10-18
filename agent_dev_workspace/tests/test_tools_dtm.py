import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools

def test_build_dtm_tool_known_answer_4():
    # Instantiate session state once
    session = SessionState()

    # Load premade load_dataset_tool to load sample fixture
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load the fixture dataset
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res.get("n_documents") == 8

    # Instantiate student tool using the same session
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # Invoke build_dtm_tool with default settings
    res = build_dtm_tool.invoke({})

    # Check status and result fields
    assert res.get("status") == "success"
    assert res.get("n_documents") == 8
    assert res.get("n_terms") == 5
    assert res.get("non_zero") == 21
    assert res.get("total_elements") == 40
    assert pytest.approx(res.get("sparsity_pct"), abs=1e-3) == 47.5

    # Assert session feature_names match vocabulary order
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocab

    # Assert vectorizer stored in artifacts and matrix dimensions
    assert "count_vectorizer" in session.artifacts
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)

    # Assert session categories and labels originating from category_name
    assert session.categories == ["catA", "catB"]
    assert len(session.labels) == 8


def test_build_dtm_tool_missing_category_name():
    session = SessionState()
    # Create session dataframe without category_name
    session.dataframe = pd.DataFrame({"text": ["hello world"]})

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    res = build_dtm_tool.invoke({})
    assert "error" in res
    assert res["error"] == "Missing 'category_name' column in session.dataframe."

    # Verify no session attributes/artifacts were created or modified
    assert session.feature_matrix is None
    assert "count_vectorizer" not in session.artifacts


def test_term_frequency_tool_known_answer_14():
    # Independent fresh session for this test
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student tools
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # Run term frequency tool
    res = term_frequency_tool.invoke({})

    assert res.get("status") == "success"
    assert res.get("total_terms") == 5
    assert res.get("returned_terms") == 5

    tf = res.get("term_frequencies")
    # Assert exact term totals matching known answer 14:
    # alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    assert tf["alpha"] == 6
    assert tf["always"] == 7
    assert tf["beta"] == 3
    assert tf["delta"] == 4
    assert tf["gamma"] == 7


def test_term_frequency_tool_before_dtm():
    session = SessionState()
    tools = make_tools(session)
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    res = term_frequency_tool.invoke({})
    assert "error" in res
    assert res["error"] == "Document-term matrix not built yet. Run build_dtm_tool first."


def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # Run heatmap tool with defaults (20 terms, 20 documents - returns all 8 docs x 5 terms)
    res = dtm_heatmap_tool.invoke({})

    assert res.get("status") == "success"
    assert res.get("n_documents") == 8
    assert res.get("n_terms") == 5
    assert res.get("terms") == ["alpha", "always", "beta", "delta", "gamma"]
    assert res.get("documents") == [f"Doc {i}" for i in range(8)]

    expected_matrix = [
        [3, 1, 1, 0, 0],  # Doc 0
        [2, 1, 1, 0, 0],  # Doc 1
        [1, 1, 1, 0, 0],  # Doc 2
        [0, 1, 0, 1, 3],  # Doc 3
        [0, 1, 0, 1, 2],  # Doc 4
        [0, 1, 0, 1, 1],  # Doc 5
        [0, 1, 0, 1, 1],  # Doc 6
        [0, 0, 0, 0, 0],  # Doc 7
    ]
    assert res.get("matrix_slice") == expected_matrix

    # Verify session.pending_figure was set
    assert session.pending_figure is not None

    # Test sliced sub-block call (n_terms=2, n_documents=3)
    slice_res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert slice_res.get("n_documents") == 3
    assert slice_res.get("n_terms") == 2
    assert slice_res.get("terms") == ["alpha", "always"]
    assert slice_res.get("matrix_slice") == [
        [3, 1],
        [2, 1],
        [1, 1]
    ]


def test_dtm_heatmap_tool_before_dtm():
    session = SessionState()
    tools = make_tools(session)
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    res = dtm_heatmap_tool.invoke({})
    assert "error" in res
    assert res["error"] == "Document-term matrix not built yet. Run build_dtm_tool first."
