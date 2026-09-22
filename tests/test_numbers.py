"""Number fidelity.

The Phase 6 criterion is 1.00 -- every number in the answer appears in the
evidence. A wrong fee or a wrong CGPA floor is the most damaging thing this
chatbot can emit, because it is the part a reader acts on without checking.

Most of these tests are about the one concession to typesetting: this bulletin
writes thousands with a comma *and a space* inside the digits (``Tk.15, 000/-``,
``45, 000``), so a literal substring test would reject a correctly copied
value. Normalisation removes separators from both sides. It stops there --
decimals are not normalised, because ``3.0`` and ``3.00`` are different claims
about a CGPA requirement.
"""
import pytest

from src.validation import numbers


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("The fee is Tk. 15,000.", ["15000"]),
    ("A CGPA of 3.00 is required.", ["3.00"]),
    ("Attendance must be 80%.", ["80"]),
    ("140 credits over 4 years.", ["140", "4"]),
    ("No numbers here.", []),
])
def test_numbers_are_extracted(text, expected):
    assert numbers.extract(text) == expected


def test_page_citations_are_not_read_as_claims():
    """Otherwise every cited page would look like an invented number."""
    assert numbers.extract("The fee is Tk. 15,000 [Page 179].") == ["15000"]
    assert numbers.extract("Two sources [Page 176] [Page 177].") == []


def test_the_bulletins_comma_space_thousands_normalise():
    assert numbers.normalise("Tk.15, 000/-") == "Tk.15000/-"
    assert numbers.normalise("45, 000") == "45000"
    assert numbers.normalise("15,000") == "15000"


def test_a_repeated_value_is_listed_once():
    assert numbers.extract("3.00 here and 3.00 there") == ["3.00"]


# ---------------------------------------------------------------------------
# Regression: normalisation must not weld separate numbers together
# ---------------------------------------------------------------------------

GRADING_TABLE = "\n".join([
    "Numerical Scores Letter Grade", "Grade Point",
    "90 - below 97", "A", "4.00",
    "87 - below 90", "A-", "3.70",
    "83 - below 87", "B+", "3.30",
])


def test_a_serialised_table_does_not_merge_adjacent_numbers():
    """The bug this guards: ``3.70`` then a newline then ``83``.

    The first separator rule dropped anything between two digits, which turned
    the grading table's consecutive cells into ``3.7083``. Three correct
    answers were then withheld as inventions, so the guard was refusing good
    answers while reporting perfect fidelity.
    """
    assert "3.7083" not in numbers.normalise(GRADING_TABLE)
    assert numbers.appears_in("3.70", numbers.normalise(GRADING_TABLE))


def test_a_grade_point_copied_from_the_table_is_supported():
    report = numbers.validate("An A- carries a grade point of 3.70 [Page 216].",
                              GRADING_TABLE)

    assert report.valid
    assert report.unsupported == []


def test_a_grade_point_not_in_the_table_is_still_caught():
    report = numbers.validate("An A- carries a grade point of 3.75 [Page 216].",
                              GRADING_TABLE)

    assert report.unsupported == ["3.75"]


def test_two_space_separated_numbers_stay_separate():
    assert numbers.normalise("140 500") == "140 500"
    assert numbers.extract("140 500") == ["140", "500"]


def test_grouped_thousands_still_collapse():
    assert numbers.normalise("1,000,000") == "1000000"
    assert numbers.normalise("Tk.15, 000/-") == "Tk.15000/-"


# ---------------------------------------------------------------------------
# Whole-number matching
# ---------------------------------------------------------------------------

def test_a_value_must_match_as_a_whole_number():
    assert not numbers.appears_in("15000", "the total is 150000 taka")
    assert not numbers.appears_in("3", "a GPA of 3.05")
    assert numbers.appears_in("3", "3 credits")


def test_a_leading_period_that_is_not_a_decimal_point_does_not_block_a_match():
    """``Tk.15000`` must still match ``15000``."""
    assert numbers.appears_in("15000", "a fee of Tk.15000/- is charged")


def test_a_decimal_tail_is_not_matched_out_of_a_longer_decimal():
    assert not numbers.appears_in("00", "a GPA of 3.00")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_a_number_copied_from_the_evidence_is_supported():
    report = numbers.validate(
        "The one-time admission fee is Tk. 15,000 [Page 179].",
        "A one-time, non-refundable admission fee of Tk.15, 000/- is charged.",
    )

    assert report.valid
    assert report.fidelity == 1.0
    assert report.unsupported == []


def test_an_invented_number_is_caught():
    """The failure this module exists for."""
    report = numbers.validate(
        "The one-time admission fee is Tk. 16,500 [Page 179].",
        "A one-time admission fee of Tk.15, 000/- is charged.",
    )

    assert not report.valid
    assert report.unsupported == ["16500"]
    assert report.fidelity == 0.0


def test_a_recomputed_total_is_caught():
    """Rule 2 of the answer prompt: never combine numbers. This enforces it."""
    report = numbers.validate(
        "Together they cost Tk. 16,000 [Page 179].",
        "An admission fee of Tk.15, 000/- and a form charge of Tk. 1, 000.",
    )

    assert report.unsupported == ["16000"]


def test_fidelity_is_the_fraction_of_grounded_numbers():
    report = numbers.validate(
        "It needs 140 credits and a CGPA of 2.75.",
        "The degree requires 140 credits with a minimum CGPA of 2.00.",
    )

    assert report.numbers == ["140", "2.75"]
    assert report.unsupported == ["2.75"]
    assert report.fidelity == pytest.approx(0.5)


def test_an_answer_with_no_numbers_is_perfectly_faithful():
    report = numbers.validate("Yes, the programme is offered.", "Some evidence.")

    assert report.valid
    assert report.fidelity == 1.0


def test_a_number_the_user_typed_is_not_counted_as_invented():
    """"How many credits is CSE 101?" puts 101 in the answer, not the model."""
    report = numbers.validate(
        "CSE 101 carries 3 credits.",
        "The course carries 3 credits.",
        question="How many credits is CSE 101?",
    )

    assert report.valid
    assert report.from_question_only == ["101"]


def test_a_value_in_neither_the_evidence_nor_the_question_is_unsupported():
    report = numbers.validate(
        "CSE 101 carries 4 credits.",
        "The course carries 3 credits.",
        question="How many credits is CSE 101?",
    )

    assert not report.valid
    assert report.unsupported == ["4"]


def test_decimals_are_not_normalised_away():
    """A CGPA stated to two places must be reproduced to two places."""
    report = numbers.validate("A CGPA of 3.0 is required.",
                              "A minimum CGPA of 3.00 is required.")

    assert not report.valid
    assert report.unsupported == ["3.0"]
