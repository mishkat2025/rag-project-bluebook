"""Deterministic number checking. No LLM involved.

A university bulletin is mostly numbers -- fees, CGPA floors, credit counts,
attendance percentages -- and a wrong one is the most damaging thing this
chatbot can produce, because it is the part a reader will act on without
double-checking. The Phase 6 criterion is 1.00: every number in the answer
appears verbatim in the evidence the generator was given.

"Verbatim" needs one concession to how this PDF is typeset. The bulletin
writes thousands as ``Tk.15, 000/-`` and ``45, 000`` -- a comma *and a space*
inside the digits -- so a literal substring test would reject an answer saying
``Tk. 15,000`` even though it copied the value correctly. Normalisation
removes separators from both sides and compares the digits. It does NOT
normalise decimals: ``3.00`` and ``3.0`` stay different, because a CGPA
requirement stated to two places should be reproduced to two places.

Two things are deliberately not counted as claims:

* **page citations** -- ``[Page 176]`` is stripped before extraction, or every
  cited page would read as an invented number;
* **numbers the user supplied** -- "How many credits is CSE 101?" puts 101 in
  the answer through no fault of the model's. The question is treated as an
  allowed source alongside the evidence, and the two are reported separately.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.validation.citations import strip_citations

#: A number, matched AFTER :func:`normalise` has removed the thousands
#: separators -- so by this point a value is an unbroken run of digits with an
#: optional decimal tail.
NUMBER = re.compile(r"\d+(?:\.\d+)?")

#: The thousands separator, and only that: a comma, plus any spaces following
#: it, sitting between a digit and exactly three more digits. ``Tk.15, 000/-``
#: and ``15,000`` both become ``15000``.
#:
#: The first version of this was ``(?<=\d)[,\s]+(?=\d)`` -- drop anything
#: between two digits that is only there for readability. It is wrong on the
#: one thing this corpus is full of. A serialised table puts each cell on its
#: own line, so the grading table on page 216 runs ``A-``, ``3.70``,
#: ``83 - below 87`` down consecutive lines, and collapsing the newline
#: between ``3.70`` and ``83`` produced ``3.7083``. The grade point then no
#: longer appeared as a whole number, and the guard withheld three *correct*
#: answers as if the model had invented their values. Requiring the comma
#: keeps the bulletin's ``45, 000`` working and cannot weld two separate
#: numbers together.
_SEPARATORS = re.compile(r"(?<=\d),[ 	 ]*(?=\d{3}(?!\d))")

#: Years, and the small integers that appear in ordinary prose ("one of the 4
#: faculties") rather than as bulletin data. Nothing is exempted by default --
#: this is here as the documented place to exempt something, and it is empty
#: because exempting values is how a fidelity metric becomes decorative.
_EXEMPT: frozenset[str] = frozenset()


@dataclass
class NumberReport:
    """Which numbers an answer asserts, and where each one came from."""

    numbers: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    from_question_only: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.unsupported

    @property
    def fidelity(self) -> float:
        """Fraction of the answer's numbers that appear in the evidence.

        1.0 for an answer containing no numbers: it asserts no value that
        could be wrong.
        """
        if not self.numbers:
            return 1.0

        return (len(self.numbers) - len(self.unsupported)) / len(self.numbers)


def normalise(text: str) -> str:
    """Drop thousands separators from inside numbers.

    ``Tk.15, 000/-`` -> ``Tk.15000/-``, so it matches an answer's ``15,000``.
    Nothing else is touched -- see :data:`_SEPARATORS` for why the wider rule
    was wrong.
    """
    return _SEPARATORS.sub("", text or "")


def extract(text: str) -> list[str]:
    """Every distinct number in ``text``, normalised, in order of appearance.

    Citation markers are removed first.
    """
    found: list[str] = []

    for raw in NUMBER.findall(normalise(strip_citations(text))):
        value = normalise(raw).strip(".,")

        if not value or value in _EXEMPT or value in found:
            continue

        found.append(value)

    return found


def appears_in(value: str, haystack: str) -> bool:
    """Whether ``value`` occurs in ``haystack`` as a whole number.

    Substring alone is not enough: ``15000`` occurs inside ``150000``, and
    ``3`` inside ``3.05``. A match must not be flanked by a digit, must not
    continue into a longer decimal, and must not begin after a decimal point
    (so ``00`` does not "match" the tail of ``3.00``). A leading ``.`` that is
    not itself preceded by a digit is ordinary punctuation -- the bulletin
    writes fees as ``Tk.15, 000/-`` -- and does not block a match.
    """
    start = 0

    while True:
        index = haystack.find(value, start)

        if index == -1:
            return False

        before = haystack[index - 1] if index else ""
        two_before = haystack[index - 2] if index > 1 else ""
        after_index = index + len(value)
        after = haystack[after_index] if after_index < len(haystack) else ""

        # A "." only blocks the match when it is a decimal point, which means
        # a digit sits in front of it.
        blocked_before = before.isdigit() or (before == "." and two_before.isdigit())
        blocked_after = after.isdigit() or (after == "." and
                                            haystack[after_index + 1:
                                                     after_index + 2].isdigit())

        if not blocked_before and not blocked_after:
            return True

        start = index + 1


def validate(answer: str, context: str, question: str = "") -> NumberReport:
    """Check every number in ``answer`` against ``context``.

    ``question`` is a secondary allowed source: a value the user typed is not
    something the model invented. Those are reported in
    :attr:`NumberReport.from_question_only` rather than silently accepted, so
    the distinction stays visible in the eval output.
    """
    evidence = normalise(context or "")
    asked = normalise(question or "")

    report = NumberReport(numbers=extract(answer))

    for value in report.numbers:
        if appears_in(value, evidence):
            continue

        if asked and appears_in(value, asked):
            report.from_question_only.append(value)
            continue

        report.unsupported.append(value)

    return report
