import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def test_mine_patterns_known_answer_17():
    """Test known answer #17 from TEST_FIXTURE.md.

    Fixture Data Breakdown for mine_patterns_tool (known answer 17):
    Category catA (rows 0, 1, 2, 7 -> 4 docs, row 7 is empty text):
      - Local CountVectorizer(stop_words="english") fits on catA texts:
        Words present: 'always' (dropped as English stop word), 'alpha', 'beta'.
        catA local vocabulary: ['alpha', 'beta'].
      - filtering_method="tfidf":
        Mean TF-IDF scores: 'alpha' ~ 0.6376, 'beta' ~ 0.3676.
        20th percentile cut leaves surviving term: ['alpha'].
      - FPGrowth with min_sup=1:
        Pattern: {'pattern': ['alpha'], 'support': 3}.
      - FPGrowth with min_sup=4:
        Support is 3, which is < min_sup=4 -> 0 patterns found.

    Category catB (rows 3, 4, 5, 6 -> 4 docs):
      - Local CountVectorizer(stop_words="english") fits on catB texts:
        Words present: 'always' (dropped as English stop word), 'delta', 'gamma'.
        catB local vocabulary: ['delta', 'gamma'].
      - filtering_method="tfidf":
        Mean TF-IDF scores: 'gamma' ~ 0.8143, 'delta' ~ 0.5444.
        20th percentile cut leaves surviving term: ['gamma'].
      - FPGrowth with min_sup=1:
        Pattern: {'pattern': ['gamma'], 'support': 4}.
      - FPGrowth with min_sup=4:
        Support is 4, which is >= min_sup=4 -> Pattern: {'pattern': ['gamma'], 'support': 4}.

    Filtering Methods "variance" & "term_frequency":
      - Both methods keep 5th-95th percentile. Across 2 terms per category, 5th-95th
        percentile band excludes both terms -> 0 terms survive filtering -> returns note.
    """
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_tool = next(t for t in premade if t.name == "load_dataset_tool")
    load_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # 1. tfidf + fpgrowth + min_sup=1 for catA
    res_a1 = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_a1["n_patterns"] == 1
    assert res_a1["patterns"][0]["pattern"] == ["alpha"]
    assert res_a1["patterns"][0]["support"] == 3

    # 2. tfidf + fpgrowth + min_sup=4 for catA (support 3 < 4 -> 0 patterns)
    res_a4 = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_a4["n_patterns"] == 0

    # 3. tfidf + fpgrowth + min_sup=1 for catB
    res_b1 = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_b1["n_patterns"] == 1
    assert res_b1["patterns"][0]["pattern"] == ["gamma"]
    assert res_b1["patterns"][0]["support"] == 4

    # 4. tfidf + fpgrowth + min_sup=4 for catB (support 4 >= 4 -> 1 pattern)
    res_b4 = mine_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4,
    })
    assert res_b4["n_patterns"] == 1
    assert res_b4["patterns"][0]["pattern"] == ["gamma"]
    assert res_b4["patterns"][0]["support"] == 4

    # 5. filtering_method="variance" -> 0 terms survive
    res_var = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_var["n_patterns"] == 0
    assert "note" in res_var
    assert res_var["note"] == "no terms survived filtering"

    # 6. filtering_method="term_frequency" -> 0 terms survive
    res_tf = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert res_tf["n_patterns"] == 0
    assert "note" in res_tf
    assert res_tf["note"] == "no terms survived filtering"


def test_mine_patterns_known_answer_20():
    """Test known answer #20 from TEST_FIXTURE.md.

    Fixture Data Breakdown for mine_patterns_tool (known answer 20):
    Using tfidf filtering:
      catA surviving terms: ['alpha'] (support 3)
      catB surviving terms: ['gamma'] (support 4)

    Algorithm "topk" (k=1, 2, 3):
      - With only 1 surviving term, there is only 1 possible pattern.
      - Top-k returns exactly 1 pattern for k=1, k=2, and k=3 alike.

    Algorithm "maxfpgrowth" (min_sup=1, min_sup=4):
      - Maximal frequent pattern mining on a single-item dataset yields 0 maximal
        patterns (PAMI MaxFPGrowth behavior).
      - Returns 0 patterns at min_sup=1 and min_sup=4.
    """
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_tool = next(t for t in premade if t.name == "load_dataset_tool")
    load_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Algorithm "topk" with k=1, k=2, k=3
    for k_val in [1, 2, 3]:
        res_topk_a = mine_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val,
        })
        assert res_topk_a["n_patterns"] == 1
        assert res_topk_a["patterns"][0]["pattern"] == ["alpha"]
        assert res_topk_a["patterns"][0]["support"] == 3

        res_topk_b = mine_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "topk",
            "k": k_val,
        })
        assert res_topk_b["n_patterns"] == 1
        assert res_topk_b["patterns"][0]["pattern"] == ["gamma"]
        assert res_topk_b["patterns"][0]["support"] == 4

    # Algorithm "maxfpgrowth" with min_sup=1 and min_sup=4
    for sup_val in [1, 4]:
        res_max_a = mine_tool.invoke({
            "category_name": "catA",
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": sup_val,
        })
        assert res_max_a["n_patterns"] == 0

        res_max_b = mine_tool.invoke({
            "category_name": "catB",
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": sup_val,
        })
        assert res_max_b["n_patterns"] == 0


def test_mine_patterns_min_sup_floor():
    """Test min_sup floor validation error for long documents (>50 distinct terms)."""
    session = SessionState()
    # Construct synthetic text with > 50 distinct terms per document
    long_text = " ".join([f"term{i}" for i in range(60)])
    large_df = pd.DataFrame({
        "text": [long_text] * 50,
        "category_name": ["cat_large"] * 50,
    })
    session.dataframe = large_df

    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res = mine_tool.invoke({
        "category_name": "cat_large",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 5,
    })

    assert "error" in res
    assert "min_sup must be at least 10" in res["error"]


def test_mine_patterns_short_documents_min_sup_one():
    """Test that short documents (<= 50 distinct terms per document) accept min_sup=1."""
    session = SessionState()
    premade, _ = load_workspace_tools("premade_tools", session)
    load_tool = next(t for t in premade if t.name == "load_dataset_tool")
    load_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    tools = make_tools(session)
    mine_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # catA has short texts averaging well under 50 distinct terms; min_sup=1 should be valid
    res = mine_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1,
    })
    assert "error" not in res
    assert res["n_patterns"] >= 0
