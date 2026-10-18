import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def test_cosine_similarity():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    
    # Known answer 16:
    # Row 0 text: "always alpha alpha alpha beta" -> counts: [3, 1, 1, 0, 0] (alpha, always, beta, delta, gamma)
    # Row 1 text: "always alpha alpha beta"           -> counts: [2, 1, 1, 0, 0]
    # Row 3 text: "always gamma gamma gamma delta"    -> counts: [0, 1, 0, 1, 3]
    # Row 4 text: "always gamma gamma delta"          -> counts: [0, 1, 0, 1, 2]
    # Row 7 text: "" (empty text)                     -> counts: [0, 0, 0, 0, 0]
    
    res_01 = cosine_sim_tool.invoke({"doc_a": 0, "doc_b": 1})
    assert res_01["similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    res_03 = cosine_sim_tool.invoke({"doc_a": 0, "doc_b": 3})
    assert res_03["similarity"] == pytest.approx(0.0909, abs=1e-3)
    
    res_34 = cosine_sim_tool.invoke({"doc_a": 3, "doc_b": 4})
    assert res_34["similarity"] == pytest.approx(0.9847, abs=1e-3)
    
    res_07 = cosine_sim_tool.invoke({"doc_a": 0, "doc_b": 7})
    assert res_07["similarity"] == pytest.approx(0.0, abs=1e-3)
    
    # Test error cases: negative index and out-of-range index
    res_neg = cosine_sim_tool.invoke({"doc_a": -1, "doc_b": 0})
    assert "error" in res_neg
    
    res_out = cosine_sim_tool.invoke({"doc_a": 0, "doc_b": 100})
    assert "error" in res_out

def test_feature_correlation_matrix():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    exploration_tools = make_exploration_tools(session)
    feat_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")
    
    # Known answer 19:
    # Per-document counts across 8 rows:
    # - alpha:  3, 2, 1, 0, 0, 0, 0, 0 (variance: 1.1875)
    # - gamma:  0, 0, 0, 3, 2, 1, 1, 0 (variance: 1.109375)
    # - delta:  0, 0, 0, 1, 1, 1, 1, 0 (variance: 0.25)
    # - beta:   1, 1, 1, 0, 0, 0, 0, 0 (variance: 0.234375)
    # - always: 1, 1, 1, 1, 1, 1, 1, 0 (variance: 0.109375)
    
    res = feat_corr_tool.invoke({})
    assert res["status"] == "success"
    assert res["terms"] == ["alpha", "gamma", "delta", "beta", "always"]
    
    mat = res["correlation_matrix"]
    
    # Check diagonal is 1.0 and shape/symmetry
    for i in range(len(mat)):
        assert mat[i][i] == pytest.approx(1.0, abs=1e-3)
        for j in range(len(mat)):
            assert mat[i][j] == pytest.approx(mat[j][i], abs=1e-3)
            
    # Check specific known values from #19:
    # alpha vs beta index mapping: alpha is 0, beta is 3
    idx_alpha = 0
    idx_beta = 3
    assert mat[idx_alpha][idx_beta] == pytest.approx(0.8885, abs=1e-3)
    
    # gamma vs delta index mapping: gamma is 1, delta is 2
    idx_gamma = 1
    idx_delta = 2
    assert mat[idx_gamma][idx_delta] == pytest.approx(0.8307, abs=1e-3)
    
    # Check pending figure is set
    assert session.pending_figure is not None
