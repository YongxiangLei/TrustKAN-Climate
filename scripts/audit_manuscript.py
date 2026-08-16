"""Audit manuscript for unresolved placeholders and unsupported quantitative claims.

This is a lightweight pre-submission guardrail, not a semantic proof checker.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

PLACEHOLDERS = [r"\\todo\{", r"\bTBD\b", r"INSERT", r"PLACEHOLDER"]
NUMERIC_CLAIM = re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percentage points|RMSE|MAE|AURC|p\s*[<=>])", re.I)
CITE = re.compile(r"\\cite\{[^}]+\}")


def tex_files(root: Path):
    return sorted(root.rglob("*.tex"))


def main(root: str, strict: bool = False):
    base = Path(root)
    issues = []
    for path in tex_files(base):
        text = path.read_text(encoding="utf-8")
        for pat in PLACEHOLDERS:
            for m in re.finditer(pat, text, flags=re.I):
                line = text.count("\n", 0, m.start()) + 1
                issues.append(("PLACEHOLDER", path, line, m.group(0)))
        for lineno, line_text in enumerate(text.splitlines(), 1):
            if NUMERIC_CLAIM.search(line_text) and not any(tok in line_text for tok in ["TBD", "table", "begin{equation}", "end{equation}"]):
                # Quantitative prose should normally point to an experiment artifact via table/figure reference.
                if "\\ref{" not in line_text and "\\eqref{" not in line_text:
                    issues.append(("NUMERIC_CLAIM_CHECK", path, lineno, line_text.strip()))
    print(f"Audited {len(tex_files(base))} TeX files")
    for kind, path, line, detail in issues:
        print(f"{kind}: {path}:{line}: {detail}")
    print(f"Total flags: {len(issues)}")
    if strict and issues:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="paper")
    p.add_argument("--strict", action="store_true")
    a = p.parse_args()
    main(a.root, a.strict)
