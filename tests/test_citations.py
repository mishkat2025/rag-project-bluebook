"""Citation validation.

The Phase 6 criterion is that every page the answer cites is a page the
retriever returned. These tests pin the parsing (which is where a checker
quietly stops checking) and the two judgements the report makes: validity, and
how many factual sentences carry a citation at all.
"""
import pytest

from src.validation import citations


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("answer,expected", [
    ("The fee is Tk. 15,000 [Page 179].", [179]),
    ("Admission needs a GPA of 3.00 [Page 176] [Page 177].", [176, 177]),
    ("Both are listed [Pages 176, 177].", [176, 177]),
    ("A range is cited [Pages 111-113].", [111, 113]),
    ("Lowercase is accepted [page 22].", [22]),
    ("Spacing is tolerated [ Page  216 ].", [216]),
    ("No citation at all.", []),
])
def test_page_numbers_are_parsed_out_of_the_answer(answer, expected):
    assert citations.extract_pages(answer) == expected


def test_a_repeated_citation_is_counted_once():
    answer = "A [Page 176]. B [Page 176]. C [Page 217]."

    assert citations.extract_pages(answer) == [176, 217]


def test_stripping_removes_markers_and_leaves_the_prose():
    answer = "The fee is Tk. 15,000 [Page 179]."

    assert citations.strip_citations(answer) == "The fee is Tk. 15,000 ."
    assert "179" not in citations.strip_citations(answer)


def test_a_bare_bracketed_number_is_not_a_citation():
    """Only [Page N] counts -- a stray [3] must not be read as page 3."""
    assert citations.extract_pages("See item [3].") == []


# ---------------------------------------------------------------------------
# Validity
# ---------------------------------------------------------------------------

def test_an_answer_citing_only_retrieved_pages_is_valid():
    report = citations.validate("A fact [Page 176].", [176, 217])

    assert report.valid
    assert report.accuracy == 1.0
    assert report.invalid_pages == []


def test_a_page_that_was_never_retrieved_is_caught():
    """The failure this module exists for: a plausible page nobody returned."""
    report = citations.validate("A fact [Page 41].", [176, 217])

    assert not report.valid
    assert report.invalid_pages == [41]
    assert report.accuracy == 0.0


def test_accuracy_is_the_fraction_of_citations_that_check_out():
    report = citations.validate(
        "A [Page 176]. B [Page 41]. C [Page 217]. D [Page 999].",
        [176, 217],
    )

    assert report.invalid_pages == [41, 999]
    assert report.accuracy == pytest.approx(0.5)


def test_an_answer_that_cites_nothing_is_valid_but_not_dense():
    """It makes no page claim, so there is nothing to be wrong about.

    Whether it *should* have cited something is citation_density's question,
    and that is what the eval harness reports alongside accuracy.
    """
    report = citations.validate(
        "The one-time admission fee is fifteen thousand taka.", [179]
    )

    assert report.valid
    assert report.accuracy == 1.0
    assert report.citation_density == 0.0


# ---------------------------------------------------------------------------
# Sentence-level density
# ---------------------------------------------------------------------------

def test_every_factual_sentence_carrying_a_citation_scores_one():
    answer = (
        "The one-time admission fee is Tk. 15,000 [Page 179]. "
        "Attendance must be at least 80 percent of classes [Page 179]."
    )

    report = citations.validate(answer, [179])

    assert report.factual_sentences == 2
    assert report.citation_density == 1.0
    assert report.uncited_sentences == []


def test_an_uncited_factual_sentence_is_reported():
    answer = (
        "The one-time admission fee is Tk. 15,000 [Page 179]. "
        "The application processing charge is a separate payment entirely."
    )

    report = citations.validate(answer, [179])

    assert report.factual_sentences == 2
    assert report.citation_density == pytest.approx(0.5)
    assert len(report.uncited_sentences) == 1


def test_a_sentence_reporting_missing_information_is_not_counted():
    """It is honest, and it has no page to cite.

    Counting it as uncited would penalise exactly the behaviour Phase 6 is
    trying to produce -- saying plainly that the bulletin does not cover part
    of a question.
    """
    answer = (
        "The one-time admission fee is Tk. 15,000 [Page 179]. "
        "The bulletin does not provide a separate fee for late applications."
    )

    report = citations.validate(answer, [179])

    assert report.factual_sentences == 1
    assert report.citation_density == 1.0


def test_a_short_fragment_is_not_treated_as_a_factual_claim():
    report = citations.validate("Yes. No.", [179])

    assert report.factual_sentences == 0
    assert report.citation_density == 1.0


# ---------------------------------------------------------------------------
# Abbreviations that precede a capitalised word
#
# The lookbehinds in _SENTENCE_END exist so "Dr. Taskeed Jabid" and "Tk.
# 15,000" are not read as sentence boundaries. Every case above happens not to
# exercise them: "Tk. 15,000" is followed by a digit, not a capital letter, so
# the lookahead never fires regardless of whether the lookbehind works. These
# cases only fire the lookahead, and would have caught the regression where
# the lookbehinds were silently inert (a raw backspace byte had been saved in
# place of the two-character regex escape \b, so every "(?<!\bXyz)" matched
# unconditionally) -- see PROGRESS.md Session 9.
# ---------------------------------------------------------------------------

def test_an_abbreviated_title_before_a_capitalised_name_does_not_split():
    answer = (
        "Dr. Taskeed Jabid is the chairperson of the department [Page 22]. "
        "He also teaches CSE 101 [Page 23]."
    )

    report = citations.validate(answer, [22, 23])

    assert report.factual_sentences == 2
    assert report.uncited_sentences == []


def test_a_bsc_abbreviation_before_a_capitalised_word_does_not_split():
    answer = (
        "The Bachelor of Science (B. Sc.) in Civil Engineering requires "
        "156.5 credits in total [Page 162]."
    )

    report = citations.validate(answer, [162])

    assert report.factual_sentences == 1
    assert report.uncited_sentences == []


def test_a_tk_amount_before_a_capitalised_word_does_not_split():
    answer = "The fee is Tk. Five hundred per semester [Page 180]."

    report = citations.validate(answer, [180])

    assert report.factual_sentences == 1
    assert report.uncited_sentences == []
