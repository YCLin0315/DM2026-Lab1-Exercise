import pytest
from scipy import sparse
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools

def test_variance_filter_tool_known_answer_5():
    # Known Answer 5 fixture per-term counts across 8 documents (in 8-row order):
    # alpha:  [3, 2, 1, 0, 0, 0, 0, 0] -> variance = 1.1875
    # always: [1, 1, 1, 1, 1, 1, 1, 0] -> variance = 0.109375
    # beta:   [1, 1, 1, 0, 0, 0, 0, 0] -> variance = 0.234375
    # delta:  [0, 0, 0, 1, 1, 1, 1, 0] -> variance = 0.25
    # gamma:  [0, 0, 0, 3, 2, 1, 1, 0] -> variance = 1.109375

    session = SessionState()

    # Load premade dataset tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using CountVectorizer defaults over sample fixture
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer()
    session.feature_matrix = vec.fit_transform(session.dataframe["text"])
    session.feature_names = vec.get_feature_names_out().tolist()

    # Instantiate variance_filter_tool
    tools = make_tools(session)
    variance_filter_tool = next(t for t in tools if t.name == "variance_filter_tool")

    # Run tool with threshold=0.15
    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res.get("status") == "success"

    # Fetch stored full_report to check individual term variances
    full_report = session.results_store[res["result_id"]]["full_report"]
    variances_by_term = dict(zip(full_report["term"], full_report["variance"]))

    # Assert variances against known answer 5
    assert variances_by_term["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert variances_by_term["always"] == pytest.approx(0.109375, abs=1e-3)
    assert variances_by_term["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert variances_by_term["delta"] == pytest.approx(0.25, abs=1e-3)
    assert variances_by_term["gamma"] == pytest.approx(1.109375, abs=1e-3)

    # Assert threshold=0.15 filtering results:
    # "always" has var 0.109375 <= 0.15, so it is flagged.
    # alpha, beta, delta, gamma all have var > 0.15, so they are kept.
    assert res.get("n_terms_total") == 5
    assert res.get("n_terms_kept") == 4
    assert res.get("n_terms_flagged") == 1
    assert res.get("flagged_terms") == ["always"]
    assert set(res.get("kept_terms_sample")) == {"alpha", "beta", "delta", "gamma"}

    # Assert session.feature_matrix shape remains unchanged (read-only behavior)
    assert session.feature_matrix.shape == (8, 5)
    assert len(session.feature_names) == 5


def test_variance_filter_tool_no_dtm():
    session = SessionState()
    tools = make_tools(session)
    variance_filter_tool = next(t for t in tools if t.name == "variance_filter_tool")

    res = variance_filter_tool.invoke({"threshold": 0.15})
    assert "error" in res
    assert res["error"] == "No document-term matrix loaded in session.feature_matrix."


def test_pearson_filter_tool_known_answer_6():
    # Known Answer 6:
    # catB binary vector across 8 documents: [0, 0, 0, 1, 1, 1, 1, 0]
    # Per-term Pearson r with catB:
    # alpha: -0.6882, always: 0.3780, beta: -0.7746, delta: 1.0000, gamma: 0.8307
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer()
    session.feature_matrix = vec.fit_transform(session.dataframe["text"])
    session.feature_names = vec.get_feature_names_out().tolist()

    # Use session.set_labels to properly populate categories as well
    session.set_labels(session.dataframe["category_name"])

    tools = make_tools(session)
    pearson_filter_tool = next(t for t in tools if t.name == "pearson_filter_tool")

    # Run with target_class="catB" and threshold=0.0
    res = pearson_filter_tool.invoke({
        "target_class": "catB",
        "threshold": 0.0
    })

    assert res.get("status") == "success"

    full_report = session.results_store[res["result_id"]]["full_report"]
    pearson_by_term = dict(zip(full_report["term"], full_report["pearson_r"]))

    assert pearson_by_term["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert pearson_by_term["always"] == pytest.approx(0.3780, abs=1e-3)
    assert pearson_by_term["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert pearson_by_term["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert pearson_by_term["gamma"] == pytest.approx(0.8307, abs=1e-3)

    # Test threshold=0.5 flagging (flags terms with |r| <= 0.5: only "always")
    res_thresh = pearson_filter_tool.invoke({
        "target_class": "catB",
        "threshold": 0.5
    })
    assert res_thresh.get("status") == "success"
    assert set(res_thresh.get("flagged_terms")) == {"always"}

    # Check top positive and negative terms (items are dicts, check ["term"])
    assert res.get("top_positive_terms")[0]["term"] == "delta"
    assert res.get("top_negative_terms")[0]["term"] == "beta"

    # Test unknown target_class returns valid_categories
    res_invalid = pearson_filter_tool.invoke({
        "target_class": "nonexistent",
        "threshold": 0.0
    })
    assert "error" in res_invalid
    assert "valid_categories" in res_invalid


def test_spearman_filter_tool_known_answer_6():
    # Known Answer 6 for Spearman rank correlation:
    # catB target vector across 8 documents: [0, 0, 0, 1, 1, 1, 1, 0]
    # Per-term counts across 8 documents:
    # alpha:  [3, 2, 1, 0, 0, 0, 0, 0]
    # always: [1, 1, 1, 1, 1, 1, 1, 0]
    # beta:   [1, 1, 1, 0, 0, 0, 0, 0]
    # delta:  [0, 0, 0, 1, 1, 1, 1, 0]
    # gamma:  [0, 0, 0, 3, 2, 1, 1, 0]
    # Note: alpha (-0.7500 vs -0.6882) and gamma (0.9363 vs 0.8307) differ from
    # their Pearson values, showing that rank transformation is applied.
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer()
    session.feature_matrix = vec.fit_transform(session.dataframe["text"])
    session.feature_names = vec.get_feature_names_out().tolist()

    session.set_labels(session.dataframe["category_name"])

    tools = make_tools(session)
    spearman_filter_tool = next(t for t in tools if t.name == "spearman_filter_tool")

    res = spearman_filter_tool.invoke({
        "target_class": "catB",
        "threshold": 0.0
    })

    assert res.get("status") == "success"

    full_report = session.results_store[res["result_id"]]["full_report"]
    spearman_by_term = dict(zip(full_report["term"], full_report["spearman_r"]))

    assert spearman_by_term["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert spearman_by_term["always"] == pytest.approx(0.3780, abs=1e-3)
    assert spearman_by_term["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert spearman_by_term["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert spearman_by_term["gamma"] == pytest.approx(0.9363, abs=1e-3)

    # Test threshold=0.5 flagging (only always has |r_s| <= 0.5)
    res_thresh = spearman_filter_tool.invoke({
        "target_class": "catB",
        "threshold": 0.5
    })
    assert res_thresh.get("status") == "success"
    assert set(res_thresh.get("flagged_terms")) == {"always"}

    # Check top positive and negative terms
    assert res.get("top_positive_terms")[0]["term"] == "delta"
    assert res.get("top_negative_terms")[0]["term"] == "beta"

    # Test unknown target_class returns valid_categories
    res_invalid = spearman_filter_tool.invoke({
        "target_class": "nonexistent",
        "threshold": 0.0
    })
    assert "error" in res_invalid
    assert "valid_categories" in res_invalid
