"""The manuscript states that no measured number in it is typed by hand.

That is a claim about the source files, so it is checkable, and it decays
silently the moment someone pastes a figure into a sentence during a revision.
This test is the enforcement: prose may carry the frozen constants that define
the protocol, and nothing else.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SECTIONS = sorted((PAPER / "sections").glob("*.tex"))
SOURCES = [PAPER / "main.tex"] + SECTIONS

# Structural lines carry no claims, and generated inputs are checked elsewhere.
STRUCTURAL = re.compile(r"^\s*(\\label|\\begin|\\end|\\generated|\\input|%)")
NUMBER = re.compile(r"(?<![\w\\])(\d+\.\d+|\d+\\%|\d{1,3}(?:,\d{3})+)")

# Every entry here is a constant of the frozen protocol or of a stated analysis
# threshold: a definition the reader needs in order to interpret the evidence,
# not a quantity the evidence produced. Measured values must arrive as macros.
PROTOCOL_CONSTANTS = {
    r"60\%",
    r"15\%",
    r"10\%",
    r"90\%",
    r"95\%",
    r"5\%",
    r"40\%",
    "0.05",
    "0.95",
    "0.90",
    "0.9",
}


def numbered(path):
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if STRUCTURAL.match(line):
            continue
        for match in NUMBER.finditer(line):
            yield index, match.group(1)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_prose_quotes_no_measured_number_directly(path):
    offenders = [
        f"{path.name}:{line} contains {value!r}"
        for line, value in numbered(path)
        if value not in PROTOCOL_CONSTANTS
    ]
    assert not offenders, (
        "measured values must reach the manuscript as generated macros:\n  "
        + "\n  ".join(offenders)
    )


def test_the_guard_would_notice_a_pasted_number(tmp_path):
    """A test that cannot fail protects nothing, so make it fail on demand."""
    sample = tmp_path / "sample.tex"
    sample.write_text("The error falls to $3.142$ at ninety days.\n", encoding="utf-8")
    assert [value for _, value in numbered(sample)] == ["3.142"]


def test_the_guard_ignores_the_frozen_constants(tmp_path):
    sample = tmp_path / "sample.tex"
    sample.write_text("Splits are 60\\%, 15\\%, 10\\% and 15\\%.\n", encoding="utf-8")
    assert all(value in PROTOCOL_CONSTANTS for _, value in numbered(sample))
