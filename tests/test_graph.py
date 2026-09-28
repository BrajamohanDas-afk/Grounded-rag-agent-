"""End-to-end graph tests with the scripted LLM — every branch of the agent."""
from grounded_rag.schemas import CitedAnswer, RouteDecision, SearchPlan
from tests.conftest import answer_with, grade_all, grade_containing, judge_all

LFP = ("LFP is safer and lasts for more charge cycles.",
       "it is safer and lasts for more charge cycles than nickel-based chemistries")
BETZ = ("A turbine can capture at most 59.3 percent of wind energy.",
        "no wind turbine can capture more than 59.3 percent of the kinetic energy in wind")


def test_happy_path(svc, llm):
    llm.handlers["SearchPlan"] = SearchPlan(queries=["LFP stationary storage"])
    llm.handlers["CitedAnswer"] = answer_with(LFP)
    res = svc.ask("Why is LFP used for stationary storage?")
    assert res["grounding_score"] == 1.0
    assert res["answer"].startswith(LFP[0]) and "[1]" in res["answer"]
    assert res["citations"][0]["source"] == "battery_storage.md"
    assert len(res["queries"]) == 2                              # original + planned, run in parallel
    assert sum(t.startswith("retrieve(") for t in res["trace"]) == 2


def test_parallel_multi_query_comparison(svc, llm):
    llm.handlers["SearchPlan"] = SearchPlan(queries=["LFP chemistry", "Betz limit wind turbine"])
    llm.handlers["CitedAnswer"] = answer_with(LFP, BETZ)
    res = svc.ask("Compare LFP batteries with the Betz limit")
    assert res["grounding_score"] == 1.0
    assert {c["source"] for c in res["citations"]} == {"battery_storage.md", "wind_power.md"}


def test_chitchat_skips_retrieval(svc, llm):
    llm.handlers["RouteDecision"] = RouteDecision(route="chitchat", reason="greeting")
    res = svc.ask("hello!")
    assert res["route"] == "chitchat" and "Ask me" in res["answer"]
    assert not llm.schema_calls("SearchPlan")


def test_abstains_when_nothing_relevant(svc, llm):
    llm.handlers["GradeBatch"] = grade_all(False)
    llm.handlers["SearchPlan"] = [SearchPlan(queries=["nuclear capacity factor"]), SearchPlan(queries=["reactor uptime"])]
    res = svc.ask("What is the capacity factor of nuclear plants?")
    assert "couldn't find" in res["answer"] and res["citations"] == []
    assert any(t.startswith("rewrite") for t in res["trace"])      # it tried again before giving up
    assert not llm.schema_calls("CitedAnswer")                    # never asked to answer without evidence


def test_repair_loop_fixes_bad_quote(svc, llm):
    bad = answer_with((LFP[0], "LFP is the safest battery ever invented by humans"))
    llm.handlers["CitedAnswer"] = [bad, answer_with(LFP)]
    res = svc.ask("Why LFP chemistry?")
    assert res["grounding_score"] == 1.0
    second_prompt = llm.schema_calls("CitedAnswer")[1]
    assert "FAILED VERIFICATION" in second_prompt and "quote not found" in second_prompt


def test_unfixable_claims_are_dropped(svc, llm):
    llm.handlers["CitedAnswer"] = answer_with(LFP, ("LFP costs 10 dollars.", "it is safer and lasts for more charge cycles"))
    res = svc.ask("Tell me about LFP chemistry")
    assert "10 dollars" not in res["answer"] and LFP[0] in res["answer"]
    assert res["grounding_score"] == 0.5
    assert len(llm.schema_calls("CitedAnswer")) == 2               # tried to repair once


def test_judge_can_reject(svc, llm):
    llm.handlers["CitedAnswer"] = answer_with(LFP)
    llm.handlers["SupportVerdicts"] = judge_all(False)
    res = svc.ask("Why LFP chemistry?")
    assert res["grounding_score"] == 0.0 and res["citations"] == []


def test_insufficient_triggers_second_search(svc, llm):
    llm.handlers["GradeBatch"] = grade_containing("wind")
    llm.handlers["SearchPlan"] = [SearchPlan(queries=[]), SearchPlan(queries=["Betz limit"])]
    llm.handlers["CitedAnswer"] = [
        lambda p: CitedAnswer(insufficient_evidence=True, missing_information="no Betz info"),
        answer_with(BETZ)]
    res = svc.ask("What limits wind turbine efficiency?")
    assert res["grounding_score"] == 1.0
    assert "Betz limit" in res["queries"]


def test_memory_contextualizes_follow_up(svc, llm):
    llm.handlers["CitedAnswer"] = answer_with(LFP)
    svc.ask("Why is LFP chemistry used?", thread_id="t1")
    llm.handlers["StandaloneQuestion"] = lambda p: __import__("grounded_rag.schemas", fromlist=["x"]).StandaloneQuestion(
        question="How much energy does LFP chemistry store per kilogram?")
    res = svc.ask("how much energy does it store?", thread_id="t1")
    ctx_prompt = llm.schema_calls("StandaloneQuestion")[-1]
    assert "Why is LFP chemistry used?" in ctx_prompt                # history was passed in
    assert res["standalone_question"].startswith("How much energy does LFP")
    assert [m["role"] for m in svc.history("t1")] == ["human", "ai", "human", "ai"]
    assert res["trace"][0].startswith("contextualize:")              # trace was reset per turn


def test_streaming_emits_node_events(svc, llm):
    llm.handlers["CitedAnswer"] = answer_with(LFP)
    events = list(svc.stream("Why LFP chemistry?"))
    nodes = [e["node"] for e in events if e["type"] == "step"]
    assert nodes[:3] == ["contextualize", "route", "plan"] and "verify" in nodes
    assert events[-1]["type"] == "result" and events[-1]["grounding_score"] == 1.0


def test_graph_renders_mermaid(svc):
    m = svc.mermaid()
    assert "retrieve" in m and "verify" in m
