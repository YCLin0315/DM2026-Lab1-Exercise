from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool():
    # Create a single session instance for state tracking
    session = SessionState()

    # Load premade tools from premade_tools/
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Invoke tool on sample_fixture.csv with text and label columns
    result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Assert against Known Answer 1 from TEST_FIXTURE.md
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}
