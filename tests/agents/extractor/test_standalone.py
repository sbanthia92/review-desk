from reviewdesk.agents.extractor import ExtractorAgent
from reviewdesk.contracts import AddClaim
from reviewdesk.testing.fakes import REPORT_DOC, FakeLLM, make_context


async def test_standalone_restatement_is_kept_and_cleaned():
    quote = "Remote staff filed 12% fewer support tickets than office staff."
    thesis = "Therefore remote work causes higher productivity."
    llm = FakeLLM(
        {
            "extractor.extract": {
                "claims": [
                    {
                        "quote": quote,
                        "standalone": "In the Q3 survey,  remote staff filed\n12% fewer tickets.",
                        "type": "factual",
                        "importance": 0.8,
                    },
                    {"quote": thesis, "standalone": thesis, "type": "thesis", "importance": 1},
                    {"quote": "Of 1,200 employees surveyed", "type": "factual", "importance": 0.4},
                ]
            }
        }
    )
    result = await ExtractorAgent().run(make_context(REPORT_DOC, llm=llm))
    claims = {u.claim.text: u.claim for u in result.ledger_updates if isinstance(u, AddClaim)}
    restated = claims[quote]
    assert restated.standalone == "In the Q3 survey, remote staff filed 12% fewer tickets."
    assert restated.checkable_text == restated.standalone
    assert restated.span.text_of(REPORT_DOC.text) == quote  # the quote still round-trips
    assert claims[thesis].standalone == ""  # identical to the quote: not repeated
    plain = claims["Of 1,200 employees surveyed"]
    assert plain.standalone == "" and plain.checkable_text == plain.text
