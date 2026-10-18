import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_check_missing_tool_known_answer_2():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res.get("n_documents") == 8

    # Instantiate student tool
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # Run check_missing_tool
    res = check_missing_tool.invoke({})

    assert res.get("status") == "success"
    assert res.get("total_rows") == 8
    # Known answer 2: Exactly 1 row has missing/empty text (the last row, index 7)
    assert res.get("missing_count") == 1
    assert res.get("missing_indices") == [7]


def test_check_missing_tool_no_dataset():
    session = SessionState()
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})
    assert "error" in res
    assert res["error"] == "No dataset loaded in session.dataframe."


def test_check_duplicates_tool_known_answer_3():
    # Fresh session for checking without dropping
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Check duplicates without dropping (drop=False)
    res = check_duplicates_tool.invoke({"drop": False})

    assert res.get("status") == "success"
    assert res.get("total_rows_before") == 8
    # Known answer 3: Exactly 1 duplicate row extra copy (rows 5 and 6 are exact copies)
    assert res.get("duplicate_count") == 1
    assert res.get("duplicate_group_indices") == [5, 6]  # 0-indexed rows 5 and 6
    assert res.get("dropped") is False
    assert res.get("total_rows_after") == 8


def test_check_duplicates_tool_drop_true():
    # Fresh independent session for dropping duplicates
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Set placeholder state to test that drop=True actually clears them
    session.feature_matrix = "dummy_matrix"
    session.labels = ["catA"] * 8
    session.set_labels(["catA", "catB"])
    session.artifacts["count_vectorizer"] = "dummy_vectorizer"

    # Confirm placeholders are set before calling tool
    assert session.feature_matrix is not None
    assert session.labels is not None
    assert session.categories == ["catA", "catB"]
    assert "count_vectorizer" in session.artifacts

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Drop duplicates (drop=True)
    drop_res = check_duplicates_tool.invoke({"drop": True})

    assert drop_res.get("status") == "success"
    assert drop_res.get("total_rows_before") == 8
    assert drop_res.get("duplicate_count") == 1
    assert drop_res.get("dropped") is True
    # Known answer 3: Dropping with keep=False leaves exactly 6 rows (both copies gone)
    assert drop_res.get("total_rows_after") == 6
    assert len(session.dataframe) == 6

    # Verify state reset behavior on drop
    assert session.feature_matrix is None
    assert session.labels is None
    assert session.categories == []
    assert "count_vectorizer" not in session.artifacts

    # Verify index reset from 0 to 5
    assert list(session.dataframe.index) == list(range(6))


def test_check_duplicates_tool_no_dataset():
    session = SessionState()
    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({})
    assert "error" in res
    assert res["error"] == "No dataset loaded in session.dataframe."


def test_tokenize_tool_known_answer_10():
    # 1. Initialize single session state for test setup
    session = SessionState()

    # 2. Load dataset via premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Instantiate tokenize_tool from tools_data using the same session
    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    # 4. Invoke tokenize_tool
    res = tokenize_tool.invoke({"text_column": "text", "output_column": "unigrams"})

    # 5. Assert summary stats
    assert res["status"] == "success"
    assert res["total_documents"] == 8
    assert res["total_tokens"] == 27  # 5 + 4 + 3 + 5 + 4 + 3 + 3 + 0

    # 6. Known answer 10 expected token lists per document (rows 0 to 7)
    expected_tokens_per_doc = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    actual_unigrams = session.dataframe["unigrams"].tolist()
    assert actual_unigrams == expected_tokens_per_doc


def test_sample_data_tool_known_answer_13():
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    # Invoke sample_data_tool with n=4, random_state=42
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res.get("status") == "success"
    assert res.get("total_rows_before") == 8
    assert res.get("sampled_rows") == 4
    assert res.get("total_rows_after") == 4

    # Known answer 13: n=4, random_state=42 returns exactly rows [1, 5, 0, 7]
    # Fixture rows derivation:
    # Row 1: "always alpha alpha beta", catA
    # Row 5: "always gamma delta", catB
    # Row 0: "always alpha alpha alpha beta", catA
    # Row 7: empty text, catA
    expected_sampled_indices = [1, 5, 0, 7]
    assert res.get("sampled_indices") == expected_sampled_indices

    # Assert session.dataframe row count and index reset
    assert len(session.dataframe) == 4
    assert list(session.dataframe.index) == [0, 1, 2, 3]


def test_sample_data_tool_errors():
    # 1. Error when no dataset loaded
    session = SessionState()
    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    res_no_data = sample_data_tool.invoke({"n": 4})
    assert "error" in res_no_data
    assert res_no_data["error"] == "No dataset loaded in session.dataframe."

    # 2. Error when n exceeds total rows (n=10 on 8 rows fixture)
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    res_exceeds = sample_data_tool.invoke({"n": 10})
    assert "error" in res_exceeds
    assert res_exceeds["error"] == "Sample size n=10 exceeds dataset size of 8 rows."


def test_list_files_tool_known_answer_11():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Known answer 11: subdirectory="newdataset" returns directories: [], files: ["Reddit-stock-sentiment.csv"]
    # Item-by-item derivation from repo structure:
    # newdataset/ directory contains only one file: "Reddit-stock-sentiment.csv" and no subdirectories.
    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res.get("status") == "success"
    assert res.get("subdirectory") == "newdataset"
    assert res.get("directories") == []
    assert res.get("files") == ["Reddit-stock-sentiment.csv"]


def test_list_files_tool_errors_and_root():
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # 1. Non-existent directory returns error
    res_nonexistent = list_files_tool.invoke({"subdirectory": "non_existent_folder_xyz"})
    assert "error" in res_nonexistent

    # 2. Path resolving outside repo root (e.g. "..") returns security error
    res_outside = list_files_tool.invoke({"subdirectory": ".."})
    assert "error" in res_outside

    # 3. Repository root listing (subdirectory="") should not include hidden files starting with "."
    res_root = list_files_tool.invoke({"subdirectory": ""})
    assert res_root.get("status") == "success"

    directories = res_root.get("directories", [])
    files = res_root.get("files", [])

    for d in directories:
        assert not d.startswith("."), f"Directory '{d}' starts with '.'"
    for f in files:
        assert not f.startswith("."), f"File '{f}' starts with '.'"


def test_describe_data_tool_known_answer_18():
    # Known Answer 18 fixture row text_length values (in 8-row order):
    # Row 0 ("always alpha alpha alpha beta"): 29
    # Row 1 ("always alpha alpha beta"): 23
    # Row 2 ("always alpha beta"): 17
    # Row 3 ("always gamma gamma gamma delta"): 30
    # Row 4 ("always gamma gamma delta"): 24
    # Row 5 ("always gamma delta"): 18
    # Row 6 ("always gamma delta"): 18
    # Row 7 (""): 0

    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({"text_column": "text"})

    assert res.get("status") == "success"

    # Assert overall_stats against known answer 18
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == pytest.approx(0.0, abs=1e-3)
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == pytest.approx(30.0, abs=1e-3)

    # Assert by_category_stats against known answer 18
    cat_a = res["by_category_stats"]["catA"]
    assert cat_a["count"] == 4
    assert cat_a["mean"] == pytest.approx(17.25, abs=1e-3)
    assert cat_a["std"] == pytest.approx(12.5, abs=1e-3)
    assert cat_a["min"] == pytest.approx(0.0, abs=1e-3)
    assert cat_a["25%"] == pytest.approx(12.75, abs=1e-3)
    assert cat_a["50%"] == pytest.approx(20.0, abs=1e-3)
    assert cat_a["75%"] == pytest.approx(24.5, abs=1e-3)
    assert cat_a["max"] == pytest.approx(29.0, abs=1e-3)

    cat_b = res["by_category_stats"]["catB"]
    assert cat_b["count"] == 4
    assert cat_b["mean"] == pytest.approx(22.5, abs=1e-3)
    assert cat_b["std"] == pytest.approx(5.7446, abs=1e-3)
    assert cat_b["min"] == pytest.approx(18.0, abs=1e-3)
    assert cat_b["25%"] == pytest.approx(18.0, abs=1e-3)
    assert cat_b["50%"] == pytest.approx(21.0, abs=1e-3)
    assert cat_b["75%"] == pytest.approx(25.5, abs=1e-3)
    assert cat_b["max"] == pytest.approx(30.0, abs=1e-3)

    # Verify side-effects
    assert "text_length" not in session.dataframe.columns
    assert session.pending_figure is not None


def test_describe_data_tool_missing_category_name():
    import pandas as pd
    session = SessionState()
    session.dataframe = pd.DataFrame({"text": ["hello", "world"]})

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({"text_column": "text"})
    assert "error" in res
    assert res["error"] == "Column 'category_name' not found in session.dataframe."


def test_describe_data_tool_single_document_category():
    import pandas as pd
    session = SessionState()
    session.dataframe = pd.DataFrame({
        "text": ["hello"],
        "category_name": ["catSingle"]
    })

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({"text_column": "text"})
    assert res.get("status") == "success"
    cat_single = res["by_category_stats"]["catSingle"]
    assert cat_single["count"] == 1
    assert cat_single["std"] is None


def test_inspect_data_tool_fixture_and_truncation():
    """Test inspect_data_tool on sample_fixture.csv and truncation logic on custom DataFrame.

    Item-by-item derivations from sample_fixture.csv:
    Total rows = 8.
    Default call (n=10):
      Returns all 8 rows. total_rows_in_dataframe = 8, inspected_rows_count = 8.
    Call with n=3:
      Row 0: original_index=0, category_name="catA", text="always alpha alpha alpha beta"
      Row 1: original_index=1, category_name="catA", text="always alpha alpha beta"
      Row 2: original_index=2, category_name="catA", text="always alpha beta"
      No fixture text contains "..." because all fixture texts are <= 30 chars (< 100 max).
    """
    import pandas as pd

    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    df_before = session.dataframe.copy()

    tools = make_tools(session)
    inspect_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # 1. Default call (n=10 default parameter)
    res_default = inspect_tool.invoke({})
    assert res_default["status"] == "success"
    assert res_default["total_rows_in_dataframe"] == 8
    assert res_default["inspected_rows_count"] == 8
    assert len(res_default["rows"]) == 8

    # 2. Call with n=3
    res_n3 = inspect_tool.invoke({"n": 3})
    assert res_n3["status"] == "success"
    assert res_n3["total_rows_in_dataframe"] == 8
    assert res_n3["inspected_rows_count"] == 3

    expected_rows_n3 = [
        {"original_index": 0, "category_name": "catA", "text": "always alpha alpha alpha beta"},
        {"original_index": 1, "category_name": "catA", "text": "always alpha alpha beta"},
        {"original_index": 2, "category_name": "catA", "text": "always alpha beta"},
    ]
    assert res_n3["rows"] == expected_rows_n3

    # Check that no text in returned fixture rows ends with "..."
    for row in res_default["rows"]:
        assert not row["text"].endswith("...")

    # Assert session.dataframe remains unchanged (read-only)
    pd.testing.assert_frame_equal(session.dataframe, df_before)

    # 3. Truncation test on custom DataFrame
    custom_session = SessionState()
    str_150 = "x" * 150
    str_100 = "y" * 100

    custom_session.dataframe = pd.DataFrame({
        "text": [str_150, str_100],
        "category_name": ["cat1", "cat2"]
    })

    custom_tools = make_tools(custom_session)
    custom_inspect_tool = next(t for t in custom_tools if t.name == "inspect_data_tool")

    res_trunc = custom_inspect_tool.invoke({"n": 2})
    rows_trunc = res_trunc["rows"]

    # 150-char text cut to 100 + "..." -> len = 103
    assert len(rows_trunc[0]["text"]) == 103
    assert rows_trunc[0]["text"] == ("x" * 100) + "..."

    # 100-char text unchanged -> len = 100
    assert len(rows_trunc[1]["text"]) == 100
    assert rows_trunc[1]["text"] == str_100


def test_inspect_data_tool_errors():
    # 1. No dataset loaded error
    session_empty = SessionState()
    tools_empty = make_tools(session_empty)
    inspect_empty = next(t for t in tools_empty if t.name == "inspect_data_tool")

    res_no_data = inspect_empty.invoke({"n": 5})
    assert "error" in res_no_data
    assert res_no_data["error"] == "No dataset loaded in session.dataframe."

    # 2. Parameter boundary errors (n=0 and n=31)
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    inspect_tool = next(t for t in tools if t.name == "inspect_data_tool")

    res_n0 = inspect_tool.invoke({"n": 0})
    assert "error" in res_n0
    assert res_n0["error"] == "n must be between 1 and 30, got 0."

    res_n31 = inspect_tool.invoke({"n": 31})
    assert "error" in res_n31
    assert res_n31["error"] == "n must be between 1 and 30, got 31."
