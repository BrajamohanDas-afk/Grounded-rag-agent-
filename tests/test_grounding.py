from langchain_core.documents import Document

from grounded_rag.grounding import locate_quote, numbers_in, render, verify_answer
from grounded_rag.schemas import Citation, CitedAnswer, CitedSentence

SRC = {"S1": Document(page_content="The Betz limit states that no wind turbine can capture more "
                                   "than 59.3 percent of the kinetic energy in wind.",
                      metadata={"source": "wind.md", "chunk_id": "a" * 16}),
       "S2": Document(page_content="Battery pack prices fell by roughly 90 percent between 2010 and 2023.",
                      metadata={"source": "bat.md", "page": 4, "chunk_id": "b" * 16})}


def S(text, *cites):
    return CitedSentence(text=text, citations=[Citation(source_id=s, quote=q) for s, q in cites])


def test_locate_quote_exact_fuzzy_and_miss():
    t = SRC["S1"].page_content
    assert locate_quote("no wind turbine can capture more than 59.3 percent", t, 90)[0]
    assert locate_quote("no wind turbine could capture more than 59.3 percent", t, 90)[0]   # 1 word off
    assert not locate_quote("turbines capture most of the wind energy", t, 90)[0]
    assert not locate_quote("wind", t, 90)[0]                                              # too short


def test_numbers():
    assert numbers_in("fell 90% between 2010 and 2023, to $1,200.") == {"90", "2010", "2023", "1200"}


def test_verify_all_failure_modes():
    ans = CitedAnswer(sentences=[
        S("Turbines capture at most 59.3% of wind energy.", ("S1", "no wind turbine can capture more than 59.3 percent")),
        S("Prices fell 95 percent.", ("S2", "Battery pack prices fell by roughly 90 percent")),   # wrong number
        S("Turbines are cheap.", ("S1", "wind turbines are very cheap to build today")),          # fake quote
        S("Offshore is windy.", ("S9", "offshore wind farms benefit from stronger winds")),       # unknown id
        S("Storage is growing."),                                                                  # no citation
    ])
    reps = verify_answer(ans, SRC)
    assert [r["supported"] for r in reps] == [True, False, False, False, False]
    assert "95" in reps[1]["problems"][0]


def test_render_renumbers_and_drops():
    ans = CitedAnswer(sentences=[
        S("Prices fell about 90 percent from 2010 to 2023.", ("S2", "prices fell by roughly 90 percent between 2010 and 2023")),
        S("The Betz limit is 59.3 percent.", ("S1", "no wind turbine can capture more than 59.3 percent")),
        S("Made up.", ("S1", "this sentence is not in the source at all")),
    ])
    reps = verify_answer(ans, SRC)
    text, cites = render(reps, SRC, False, "")
    assert text.startswith("Prices fell about 90 percent from 2010 to 2023.[1] The Betz limit is 59.3 percent.[2]")
    assert "Made up" not in text and "1 statement(s) were removed" in text
    assert cites[0]["label"] == "bat.md, p.4" and cites[1]["source_id"] == "S1"


def test_find_span_for_highlighting():
    from grounded_rag.grounding import find_span
    text = "Lithium iron phosphate (LFP) chemistry is\n  increasingly popular for stationary storage."
    s = find_span(text, "chemistry is increasingly popular")
    assert text[s[0]:s[1]].split() == ["chemistry", "is", "increasingly", "popular"]
    assert find_span(text, "completely unrelated words about turbines") is None
