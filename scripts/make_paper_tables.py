"""Generate the manuscript's CET tables directly from the frozen ledgers.

The manuscript rule is that no numerical value is typed by hand, so every table
in the results section is emitted here from `results/aggregated` and
`results/reliability/aggregated`. Runs whose code fingerprint differs from the
current one are dropped rather than silently averaged in.

This module is deliberately outside the `code_sha256` set (which covers the
runners and `src/`), so regenerating tables never invalidates a completed run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401  # repository-root import setup

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Study:
    """Where one study's ledgers live and which model it is arguing about.

    The v2 architecture had to be evaluated without disturbing the published
    artifacts, so each study writes to its own experiment names. Collecting the
    paths here keeps the generator from being rewritten per study and keeps the
    two sets of tables from ever being mixed.
    """

    benchmark: Path
    reliability: Path
    comparisons: Path
    ablations: Path
    ablation_runs: Path
    extremes: Path
    proposed: str
    # Where to read rows the current campaign did not produce. The classical
    # shard costs about twenty-five hours, almost all of it in the SVR
    # hyperparameter search, so the v2 campaign is neural-only and quotes those
    # two rows from the frozen campaign instead. Importing them is only sound
    # because the source edit is demonstrably inert for models it does not
    # name, which scripts/verify_baseline_inertness.py establishes bit for bit;
    # the imported fingerprint is disclosed in the caption and the provenance.
    imported: Path | None = None


STUDIES = {
    "v1": Study(
        benchmark=ROOT / "results" / "aggregated" / "cet_full_runs.csv",
        reliability=ROOT / "results" / "reliability" / "aggregated"
        / "cet_reliability_full_runs.csv",
        comparisons=ROOT / "results" / "statistical_tests" / "cet_primary"
        / "primary_comparisons_index.json",
        ablations=ROOT / "results" / "ablations" / "aggregated"
        / "ablations_cet_full_runs.csv",
        ablation_runs=ROOT / "results" / "ablations" / "runs" / "ablations_cet_full",
        extremes=ROOT / "results" / "extremes" / "cet_extremes_runs.csv",
        proposed="trustkan",
    ),
    "v2": Study(
        benchmark=ROOT / "results" / "aggregated" / "cet_v2_neural_runs.csv",
        reliability=ROOT / "results" / "reliability" / "aggregated"
        / "cet_reliability_v2_runs.csv",
        comparisons=ROOT / "results" / "statistical_tests" / "cet_v2"
        / "primary_comparisons_index.json",
        ablations=ROOT / "results" / "ablations" / "aggregated"
        / "ablations_cet_v2_runs.csv",
        ablation_runs=ROOT / "results" / "ablations" / "runs" / "ablations_cet_v2",
        extremes=ROOT / "results" / "extremes_v2" / "cet_extremes_runs.csv",
        proposed="trustkan_v2",
        imported=ROOT / "results" / "aggregated" / "cet_full_runs.csv",
    ),
}
# Which study the committed tables belong to. It is the default so that running
# a generator with no arguments cannot quietly replace the manuscript's tables
# with another study's; the superseded campaign stays reachable through
# --study v1, which is what keeps the published artifacts reproducible.
MANUSCRIPT_STUDY = "v2"
# Set once by main. Module-level so the table builders do not each have to
# thread the study through, and frozen per invocation so they cannot disagree
# about which ledgers they are reading.
STUDY = STUDIES[MANUSCRIPT_STUDY]
HORIZONS = (1, 7, 30, 90)
MODEL_LABELS = {
    "persistence": "Persistence",
    "svr": "SVR",
    "random_forest": "Random forest",
    "mlp": "MLP",
    "lstm": "LSTM",
    "gru": "GRU",
    "tcn": "TCN",
    "transformer": "Transformer",
    "kan": "KAN (plain)",
    "trustkan": "TrustKAN (published)",
    "trustkan_dilated": "TrustKAN, wide stem",
    "trustkan_v2": "TrustKAN v2 (ours)",
}
MODEL_ORDER = list(MODEL_LABELS)
# Fitted but not neural. Persistence is excluded: it is trivial rather than
# classical, and the manuscript already treats it as the weakest bar available.
CLASSICAL_MODELS = ("svr", "random_forest")


def load_ok(path: Path, expected_fingerprint: str, *, has_status: bool = True) -> pd.DataFrame:
    """Load a ledger and keep only runs produced by the current code.

    A ledger regenerated before a source change still parses cleanly and still
    reports plausible numbers, so majority-voting on the fingerprint is unsafe:
    an entirely stale ledger looks unanimous. The fingerprint is therefore
    checked against the live source tree, and a ledger that contributes nothing
    current is a hard error rather than an empty table.

    `has_status` is false for ledgers that re-score already-successful runs and
    so carry no status column; it is named at the call site rather than
    inferred, because silently skipping the filter on a ledger that should have
    one is the failure this gate exists to prevent.
    """
    frame = pd.read_csv(path)
    ok = frame.copy() if not has_status else frame[frame.status.eq("ok")].copy()
    if ok.empty:
        raise ValueError(f"No successful runs in {path}")
    current = ok[ok.code_sha256.eq(expected_fingerprint)]
    if current.empty:
        found = ", ".join(sorted({f[:12] for f in ok.code_sha256}))
        raise SystemExit(
            f"{path.relative_to(ROOT)} holds no run matching the current code "
            f"fingerprint {expected_fingerprint[:12]} (found: {found}). "
            "Re-run the campaign's --collect-only step before regenerating tables."
        )
    stale = len(ok) - len(current)
    if stale:
        print(f"  dropping {stale} run(s) from superseded code fingerprints")
    return current


def load_imported(path: Path, live_fingerprint: str, models, *, has_status: bool = True) -> pd.DataFrame:
    """Load rows the current campaign did not produce, and insist they are old.

    This is the one place a superseded fingerprint may enter a table, so the
    conditions are tightened rather than relaxed. The rows must come from a
    single fingerprint, that fingerprint must differ from the live one -- if it
    matched, the row belongs in the current ledger and importing it would hide
    a gap -- and the fingerprint travels into the caption and the provenance
    file so a reader is told which runs are being quoted from where.
    """
    frame = pd.read_csv(path)
    ok = frame[frame.model.isin(models)].copy()
    if has_status:
        ok = ok[ok.status.eq("ok")]
    if ok.empty:
        return ok
    found = sorted({str(value) for value in ok.code_sha256})
    if len(found) != 1:
        raise SystemExit(f"{path.name} mixes fingerprints for imported rows: {found}")
    if found[0] == live_fingerprint:
        raise SystemExit(
            f"{path.name} carries the live fingerprint {live_fingerprint[:12]} for "
            f"{sorted(set(ok.model))}; those runs belong in the current ledger"
        )
    return ok


def fmt(value, digits=3):
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def oxford(items: list[str]) -> str:
    # Model labels are capitalized for table rows but appear mid-sentence here,
    # except for acronyms, which must keep their case.
    lowered = [
        item if item.isupper() else item[0].lower() + item[1:] for item in items
    ]
    if len(lowered) <= 2:
        return " and ".join(lowered)
    return ", ".join(lowered[:-1]) + f", and {lowered[-1]}"


def display_path(path: Path) -> str:
    """Repository-relative when possible, so a relative --outdir cannot abort a run."""
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT):
        return resolved.relative_to(ROOT).as_posix()
    return resolved.as_posix()


def write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"  wrote {display_path(path)}")


def benchmark_table(bench: pd.DataFrame, out: Path, imported: pd.DataFrame | None = None) -> None:
    imported_note = ""
    if imported is not None and not imported.empty:
        names = sorted(set(imported.model))
        fingerprint = str(imported.code_sha256.iloc[0])[:12]
        labels = [MODEL_LABELS.get(m, m) for m in names]
        joined = " and ".join(labels) if len(labels) <= 2 else (
            ", ".join(labels[:-1]) + f", and {labels[-1]}"
        )
        imported_note = (
            f" {joined}"
            + (" is" if len(names) == 1 else " are")
            + " quoted from the superseded campaign under code fingerprint "
            + rf"\texttt{{{fingerprint}}}, marked $\dagger$: refitting them costs "
            r"about twenty-five hours and the source edit provably cannot reach "
            r"them, since every architecture shared by the two campaigns "
            r"reproduces its stored predictions byte for byte under the current "
            r"code."
        )
        bench = pd.concat([bench, imported], ignore_index=True)
    stats = bench.groupby(["model", "horizon"]).rmse.agg(["mean", "std", "size"])
    dagger = set() if imported is None else set(imported.model)
    best = {h: stats.xs(h, level="horizon")["mean"].idxmin() for h in HORIZONS}
    # The dashes have to be accounted for in the caption rather than left to be
    # read as a modelling choice, and the count must come from the ledger so it
    # cannot drift as the remaining shards land.
    present = set(stats.index)
    absent = sorted(
        MODEL_LABELS[model]
        for model in MODEL_ORDER
        if model in stats.index.get_level_values("model")
        for horizon in HORIZONS
        if (model, horizon) not in present
    )
    note = (
        ""
        if not absent
        else " Dashes mark long-horizon fits of "
        + oxford(sorted(set(absent)))
        + " that are still running on CPU; no comparison in this paper depends "
        "on them."
    )
    # Which horizons the proposed model wins is a property of the ledger, so the
    # caption is derived rather than asserted; the same generator serves the
    # published study, where the answer is none.
    won = [h for h in HORIZONS if best.get(h) == STUDY.proposed]
    label = MODEL_LABELS[STUDY.proposed]
    if not won:
        verdict = f"{label} never attains it."
    elif len(won) == len(HORIZONS):
        verdict = f"{label} attains it at every horizon."
    else:
        verdict = f"{label} attains it at " + oxford([f"$h={h}$" for h in won]) + "."
    lines = [
        "% Generated by scripts/make_paper_tables.py -- do not edit by hand.",
        # Ten models against four horizons with a deviation in every cell does
        # not fit an IEEEtran column, so the headline table spans both.
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{CET test RMSE (\degC), mean $\pm$ standard deviation over "
        r"five seeds. Deterministic baselines use a single run. Bold marks the "
        r"lowest mean at each horizon, which is a ranking of point estimates and "
        r"not a significance statement; Table~\ref{tab:cet_paired} settles which "
        rf"differences survive correction. {verdict}{note}{imported_note}}}",
        r"\label{tab:cet_rmse}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & $h=1$ & $h=7$ & $h=30$ & $h=90$ \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        if model not in stats.index.get_level_values("model"):
            continue
        cells = []
        for horizon in HORIZONS:
            if (model, horizon) not in stats.index:
                cells.append("--")
                continue
            row = stats.loc[(model, horizon)]
            cell = fmt(row["mean"])
            if row["size"] > 1 and pd.notna(row["std"]):
                cell += rf"\,$\pm$\,{row['std']:.3f}"
            if best[horizon] == model:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        name = MODEL_LABELS[model] + (r"$^{\dagger}$" if model in dagger else "")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(out, lines)


def reliability_table(rel: pd.DataFrame, out: Path) -> None:
    nominal = float(rel.nominal_coverage.iloc[0])
    stats = rel.groupby("horizon").agg(
        marginal=("conformal_marginal_coverage", "mean"),
        joint=("conformal_joint_coverage", "mean"),
        simultaneous=("simultaneous_joint_coverage", "mean"),
        width=("conformal_mean_width", "mean"),
        score=("conformal_interval_score", "mean"),
    )
    lines = [
        "% Generated by scripts/make_paper_tables.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Conformal calibration on CET at nominal coverage {nominal:.2f}, "
        r"averaged over five seeds. Marginal coverage holds, as split conformal "
        r"guarantees; per-origin joint coverage collapses with lead time.}",
        r"\label{tab:cet_calibration}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Quantity & $h=1$ & $h=7$ & $h=30$ & $h=90$ \\",
        r"\midrule",
    ]
    rows = [
        ("Marginal coverage", "marginal", 3),
        ("Joint coverage", "joint", 3),
        ("Simultaneous coverage", "simultaneous", 3),
        (r"Mean width (\degC)", "width", 2),
        (r"Interval score (\degC)", "score", 2),
    ]
    for label, column, digits in rows:
        cells = [fmt(stats.loc[h, column], digits) if h in stats.index else "--" for h in HORIZONS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(out, lines)


def selective_table(rel: pd.DataFrame, out: Path) -> None:
    """Rank the reliability score against the components it is fused from.

    Which of the three AURC rows is best is the whole question this table
    answers, so both the bold and the caption are read off the ledger. Bolding
    a fixed row and asserting the outcome in words was correct for the frozen
    weighting and would have silently misreported the fitted one, which is the
    class of error the no-hand-typed-numbers rule exists to prevent.
    """
    stats = rel.groupby("horizon").agg(
        rmse=("rmse", "mean"),
        fused=("fused_aurc", "mean"),
        width=("width_only_aurc", "mean"),
        shift=("shift_only_aurc", "mean"),
        auroc=("error_detection_auroc", "mean"),
        spearman=("reliability_spearman", "mean"),
    )
    components = ("fused", "width", "shift")
    best = {h: min(components, key=lambda c: stats.loc[h, c]) for h in stats.index}
    horizons = [h for h in HORIZONS if h in stats.index]
    fused_wins = [h for h in horizons if best[h] == "fused"]
    beaten = [h for h in horizons if stats.loc[h, "fused"] > stats.loc[h, "width"]]
    if not beaten:
        verdict = (
            "The fused score is never beaten by either component it is built "
            "from, which the calibration-selected weight guarantees on the "
            "calibration split and which is confirmed here on test."
        )
    elif len(beaten) == len(horizons):
        verdict = (
            "The fused score loses to its own width-only component at every horizon."
        )
    else:
        verdict = (
            "The fused score loses to its own width-only component at "
            + oxford([f"$h={h}$" for h in beaten])
            + "."
        )
    if fused_wins:
        verdict += (
            " It is the best of the three at "
            + oxford([f"$h={h}$" for h in fused_wins])
            + "."
        )
    chance = [h for h in horizons if abs(stats.loc[h, "auroc"] - 0.5) < 0.05]
    if chance:
        verdict += (
            " Top-error AUROC is at chance at "
            + oxford([f"$h={h}$" for h in chance])
            + " and rises with lead time."
        )
    lines = [
        "% Generated by scripts/make_paper_tables.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Selective prediction on CET, averaged over five seeds. Lower AURC "
        rf"is better and bold marks the best of the three scores. {verdict}}}",
        r"\label{tab:cet_selective}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Quantity & $h=1$ & $h=7$ & $h=30$ & $h=90$ \\",
        r"\midrule",
    ]
    rows = [
        ("RMSE, no abstention", "rmse", 3),
        ("AURC, fused score", "fused", 3),
        ("AURC, width only", "width", 3),
        ("AURC, shift only", "shift", 3),
        ("Top-error AUROC", "auroc", 3),
        (r"Error--reliability $\rho$", "spearman", 3),
    ]
    for label, column, digits in rows:
        cells = []
        for horizon in HORIZONS:
            if horizon not in stats.index:
                cells.append("--")
                continue
            cell = fmt(stats.loc[horizon, column], digits)
            if column in components and best[horizon] == column:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(out, lines)


def comparison_table(index_path: Path, out: Path) -> None:
    """Tabulate the pre-specified paired-bootstrap family.

    Both directions are printed. A table that counted only the comparator's
    wins would let a favourable mean difference read as a win in a row whose
    interval covers zero, which is the reading Holm correction exists to
    prevent; the inconclusive count has to be as visible as the other two.
    """
    rows = json.loads(index_path.read_text(encoding="utf-8"))
    label = MODEL_LABELS.get(STUDY.proposed, STUDY.proposed)
    total = sum(row["n_pairs"] for row in rows)
    won = sum(row["a_better"] for row in rows)
    lost = sum(row["b_better"] for row in rows)
    tied = sum(row["inconclusive"] for row in rows)
    family = sorted({row["family_size"] for row in rows})
    family_text = (
        f"{family[0]}" if len(family) == 1 else oxford([str(f) for f in family])
    )
    if won == 0:
        verdict = f"{label} wins none of the {total}"
    else:
        verdict = f"{label} wins {won} of the {total}"
    verdict += f", loses {lost}, and is inconclusive on {tied}."
    lines = [
        "% Generated by scripts/make_paper_tables.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Pre-specified paired comparisons on CET. $\Delta$RMSE is the mean "
        rf"difference ({label} minus the comparator; negative favours {label}). The "
        r"last three columns count, of five seed-matched pairs, how many resolve in "
        r"each direction under a circular moving-block bootstrap over forecast "
        r"origins with Holm correction applied across each comparator's family of "
        rf"{family_text} comparisons. {verdict}}}",
        r"\label{tab:cet_paired}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Comparator & Horizon & $\Delta$RMSE & Ours & Comparator & Tied \\",
        r"\midrule",
    ]
    for i, row in enumerate(rows):
        if i and rows[i - 1]["model_b"] != row["model_b"]:
            lines.append(r"\midrule")
        comparator = MODEL_LABELS.get(row["model_b"], row["model_b"])
        name = comparator if i == 0 or rows[i - 1]["model_b"] != row["model_b"] else ""
        lines.append(
            f"{name} & $h={row['horizon']}$ & "
            f"{row['mean_rmse_difference']:+.3f} & "
            f"{row['a_better']} & {row['b_better']} & {row['inconclusive']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(out, lines)


TRAINED_ABLATIONS = {
    "A0": "Full model",
    "A1": "KAN encoder $\\to$ budget-matched MLP",
    "A2": "No quantile head",
    "A9": "Mean-pooled readout (superseded)",
    "A10": "Local stem, field spans three steps",
    "A11": "Last-state readout, no global aggregation",
}


def ablation_verdict(stats: pd.DataFrame) -> str:
    """What each single-element removal cost, from the ablation ledger.

    A0 is the full model, so every other row is A0 plus one thing taken away
    and the caption is a list of those costs. Stating them rather than naming
    one of them keeps the caption honest whichever way the numbers fall.
    """
    ids = set(stats.index.get_level_values("ablation_id"))
    if "A0" not in ids:
        return ""
    horizons = [h for h in HORIZONS if ("A0", h) in stats.index]
    spread = stats.loc["A0", "std"].max()

    def gap(identifier: str) -> float | None:
        available = [h for h in horizons if (identifier, h) in stats.index]
        if not available:
            return None
        return max(
            stats.loc[(identifier, h), "mean"] - stats.loc[("A0", h), "mean"]
            for h in available
        )

    parts = []
    for identifier, phrase in (
        ("A1", "replacing the KAN mapping with a budget-matched MLP"),
        ("A10", "restoring the symmetrically padded local stem"),
        ("A11", "reading the final state alone"),
        ("A9", "restoring the superseded mean-pooled readout"),
    ):
        worst = gap(identifier)
        if worst is None:
            continue
        parts.append(
            rf"{phrase} ({identifier}) costs up to {worst:.3f}\,\degC{{}}"
        )
    if not parts:
        return ""
    return (
        "Against the full model, "
        + oxford(parts)
        + rf", against an across-seed standard deviation of at most {spread:.3f}."
    )


def trained_ablation_table(abl: pd.DataFrame, out: Path) -> None:
    stats = abl.groupby(["ablation_id", "horizon"]).rmse.agg(["mean", "std"])
    lines = [
        "% Generated by scripts/make_paper_tables.py -- do not edit by hand.",
        # Six columns with a spelled-out variant name do not fit an IEEEtran
        # column, so the two ablation tables span both.
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Trained ablations on CET, test RMSE (\degC) as mean $\pm$ standard "
        r"deviation over five seeds. Each variant removes exactly one element of the "
        rf"design. {ablation_verdict(stats)}}}",
        r"\label{tab:cet_ablation}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"ID & Variant & $h=1$ & $h=7$ & $h=30$ & $h=90$ \\",
        r"\midrule",
    ]
    for identifier, label in TRAINED_ABLATIONS.items():
        if identifier not in stats.index.get_level_values("ablation_id"):
            continue
        cells = []
        for horizon in HORIZONS:
            if (identifier, horizon) not in stats.index:
                cells.append("--")
                continue
            row = stats.loc[(identifier, horizon)]
            cells.append(rf"{row['mean']:.3f}\,$\pm$\,{row['std']:.3f}")
        lines.append(f"{identifier} & {label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(out, lines)


def evaluation_ablation_table(out: Path):
    """Aggregate the A3--A8 recomputations stored inside each A0 record.

    Returns the per-horizon means so that prose quoting one of these gains
    draws on the same aggregation as the table rather than a second one.
    """
    rows, adaptive = [], []
    for path in sorted(STUDY.ablation_runs.glob("ablation_A0_h*_s*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "ok":
            continue
        payload = json.loads(record["ablation_json"])
        a8 = payload["A8_static_vs_adaptive_conformal"]
        adaptive.append(
            {
                "horizon": int(record["horizon"]),
                "static_dev": a8["static_rolling_deviation"],
                "adaptive_dev": a8["adaptive_rolling_deviation"],
                "static_width": a8["static_mean_width"],
                "adaptive_width": a8["adaptive_mean_width"],
            }
        )
        rows.append(
            {
                "horizon": int(record["horizon"]),
                "A3": payload["A3_no_conformal"]["coverage_gain"],
                "A4": payload["A4_no_embedding_shift"]["aurc_gain_from_shift"],
                "A5": payload["A5_no_interval_width"]["aurc_gain_from_width"],
                "A6": payload["A6_no_fusion"]["aurc_gain_from_fusion"],
                "A7": payload["A7_no_abstention"]["rmse_reduction"],
                "A8": payload["A8_static_vs_adaptive_conformal"][
                    "rolling_deviation_improvement"
                ],
            }
        )
    if not rows:
        print("  skipping evaluation-side ablation table; no A0 records found")
        return None
    stats = pd.DataFrame(rows).groupby("horizon").mean()
    adaptive_macros(pd.DataFrame(adaptive), out.with_name("cet_adaptive.tex"))

    def eval_verdict(stats: pd.DataFrame, labels: dict) -> str:
        """Which components earn their place, by the sign of their own gain.

        The pre-registered rule is that a component whose ablation shows no
        incremental value is withdrawn, so the caption applies that rule to the
        numbers instead of restating the outcome of a previous campaign.
        """
        kept, dropped = [], []
        for identifier, (name, _measured) in labels.items():
            column = stats[identifier]
            (kept if (column > 0).all() else dropped).append(f"{name} ({identifier})")
        sentences = []
        if kept:
            verb = "gain" if len(kept) > 1 else "gains"
            sentences.append(f"{oxford(kept)} {verb} at every horizon.")
        if dropped:
            verb = "do not" if len(dropped) > 1 else "does not"
            sentences.append(f"{oxford(dropped)} {verb}.")
        return " ".join(sentence[0].upper() + sentence[1:] for sentence in sentences)

    labels = {
        "A3": ("Conformal correction", "coverage gain over raw quantiles"),
        "A4": ("Embedding shift", "AURC gain over width-only"),
        "A5": ("Interval width", "AURC gain over shift-only"),
        "A6": ("Fusion", "AURC gain over best single component"),
        "A7": ("Abstention", "RMSE reduction on retained set"),
        "A8": ("Adaptive conformal", "rolling-deviation gain over static"),
    }
    lines = [
        "% Generated by scripts/make_paper_tables.py -- do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Evaluation-side ablations on CET, recomputed from the A0 artifacts "
        r"and averaged over five seeds. Each entry is the gain attributable to the "
        rf"named component, so positive favours keeping it. {eval_verdict(stats, labels)}}}",
        r"\label{tab:cet_eval_ablation}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"ID & Component (gain measured) & $h=1$ & $h=7$ & $h=30$ & $h=90$ \\",
        r"\midrule",
    ]
    for identifier, (name, measured) in labels.items():
        cells = [
            fmt(stats.loc[h, identifier], 3) if h in stats.index else "--"
            for h in HORIZONS
        ]
        lines.append(
            f"{identifier} & {name}, {measured} & " + " & ".join(cells) + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(out, lines)
    return stats


EXTREMES_LEDGER = ROOT / "results" / "extremes" / "cet_extremes_runs.csv"


def extremes_verdict(best: dict) -> str:
    label = MODEL_LABELS[STUDY.proposed]
    won = [h for h in HORIZONS if best.get(h) == STUDY.proposed]
    if not won:
        return f"{label} attains it nowhere."
    if len(won) == len(HORIZONS):
        return f"{label} attains it at every horizon."
    return f"{label} attains it at " + oxford([f"$h={h}$" for h in won]) + "."


def extremes_table(frame: pd.DataFrame, out: Path) -> None:
    """Test-set RMSE restricted to the pre-registered extreme subsets."""
    stats = frame.groupby(["model", "horizon"]).either_rmse.mean()
    counts = frame.groupby("horizon")[["cold_n_origins", "warm_n_origins"]].first()
    best = {
        h: stats.xs(h, level="horizon").idxmin()
        for h in HORIZONS
        if h in stats.index.get_level_values("horizon")
    }
    lines = [
        "% Generated by scripts/make_paper_tables.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Test RMSE (\degC) on the pre-registered extreme subsets, defined by "
        r"the 5th and 95th percentiles of the training targets only and labelled from "
        r"each origin's first lead. Bold marks the lowest mean at each horizon. "
        rf"{extremes_verdict(best)}}}",
        r"\label{tab:cet_extremes}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & $h=1$ & $h=7$ & $h=30$ & $h=90$ \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        if model not in stats.index.get_level_values("model"):
            continue
        cells = []
        for horizon in HORIZONS:
            if (model, horizon) not in stats.index:
                cells.append("--")
                continue
            cell = fmt(stats.loc[(model, horizon)])
            if best.get(horizon) == model:
                cell = rf"\textbf{{{cell}}}"
            cells.append(cell)
        lines.append(f"{MODEL_LABELS[model]} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        "Cold origins & "
        + " & ".join(str(int(counts.loc[h, "cold_n_origins"])) for h in HORIZONS)
        + r" \\",
        "Warm origins & "
        + " & ".join(str(int(counts.loc[h, "warm_n_origins"])) for h in HORIZONS)
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    write(out, lines)


def cost_verdict(stats: pd.DataFrame) -> str:
    """Where the proposed model sits on cost, read off the ledger.

    The point of the table is to close the explanation that the proposed model
    was starved of or lavished with compute, so the sentence has to be able to
    report either answer, including the awkward one where it is the slowest.
    """
    proposed = STUDY.proposed
    if proposed not in stats.index:
        return ""
    label = MODEL_LABELS[proposed]
    ratio = stats.train[proposed] / stats.train.min()
    slowest = stats.train.idxmax() == proposed
    largest = stats.parameters.idxmax() == proposed
    size = "the largest" if largest else "not the largest"
    speed = "the slowest" if slowest else "not the slowest"
    return (
        f"{label} is {size} and {speed} model in the table, training "
        rf"{ratio:.1f}$\times$ slower than the cheapest neural baseline; the "
        r"cost of the wide stem and the attention readout is stated here rather "
        r"than left implicit."
    )


def efficiency_table(bench: pd.DataFrame, out: Path) -> None:
    """Parameter count, training time, and inference latency per model.

    Only the models that ran on one accelerator are tabulated. The classical
    baselines were sharded to CPU, so their wall-clock figures measure a
    different device and would invite a comparison the ledger cannot support;
    they are named in the caption instead of being listed with times.
    """
    devices = sorted(set(bench.device.dropna()))
    neural = bench[bench.device.str.startswith("cuda", na=False)]
    if neural.empty:
        print("  skipping efficiency table; no accelerator runs in the ledger")
        return
    accelerators = sorted(set(neural.cuda_device_name.dropna()))
    excluded = sorted(set(bench.model) - set(neural.model))
    stats = neural.groupby("model").agg(
        parameters=("parameters", "mean"),
        train=("train_seconds", "mean"),
        latency=("inference_ms", "mean"),
        runs=("seed", "size"),
    )
    caption = (
        r"Computational cost of the neural models, averaged over all horizons and "
        r"seeds on a single "
        + (accelerators[0] if accelerators else "accelerator")
        + r". Training time is per run to the early-stopping point; inference "
        r"latency is per forecast origin. "
        + (
            oxford([MODEL_LABELS.get(m, m) for m in excluded])
            + (" was" if len(excluded) == 1 else " were")
            + " not run on the accelerator and "
            + ("is" if len(excluded) == 1 else "are")
            + " omitted, since a wall-clock time measured on different hardware "
            "would not be comparable. "
            if excluded
            else ""
        )
        + cost_verdict(stats)
    )
    lines = [
        "% Generated by scripts/make_paper_tables.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:cet_efficiency}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & Parameters & Train (s) & Latency (ms) \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        if model not in stats.index:
            continue
        row = stats.loc[model]
        lines.append(
            f"{MODEL_LABELS[model]} & {int(row.parameters):,} & "
            f"{fmt(row.train, 1)} & {fmt(row.latency, 3)}" + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(out, lines)
    if len(devices) > 1:
        print(f"  efficiency table covers {len(neural)} accelerator runs of {len(bench)}")


def dataset_macros(rel: pd.DataFrame, out: Path) -> None:
    """Describe the CET panel from the artifacts rather than from the config.

    The split sizes and the calendar span are properties of what was actually
    windowed and evaluated, so they are read back off a stored artifact.
    """
    import numpy as np

    row = rel[rel.horizon.eq(1)].iloc[0]
    total = int(row.n_train + row.n_validation + row.n_calibration + row.n_test)
    with np.load(ROOT / row.artifact_path, allow_pickle=False) as source:
        times = pd.to_datetime(source["target_time"].reshape(-1))
    last = times.max()
    first = last - pd.Timedelta(days=total - 1)
    lines = [
        "% Generated by scripts/make_paper_tables.py -- do not edit by hand.",
        rf"\newcommand{{\cetFirstYear}}{{{first.year}}}",
        rf"\newcommand{{\cetLastYear}}{{{last.year}}}",
        rf"\newcommand{{\cetSpanYears}}{{{(last - first).days / 365.25:.0f}}}",
        rf"\newcommand{{\cetSamples}}{{{total:,}}}",
        rf"\newcommand{{\cetTestStart}}{{{times.min().date()}}}",
        rf"\newcommand{{\cetTestEnd}}{{{last.date()}}}",
        rf"\newcommand{{\cetTestSamples}}{{{int(row.n_test):,}}}",
    ]
    write(out, lines)


def adaptive_macros(frame: pd.DataFrame, out: Path) -> None:
    """Macros for the A8 static-versus-adaptive conformal comparison."""
    stats = frame.groupby("horizon").mean()
    names = {1: "One", 7: "Seven", 30: "Thirty", 90: "Ninety"}
    lines = ["% Generated by scripts/make_paper_tables.py -- do not edit by hand."]
    for horizon, word in names.items():
        if horizon not in stats.index:
            continue
        row = stats.loc[horizon]
        drop = 100 * (row.static_dev - row.adaptive_dev) / row.static_dev
        cost = 100 * (row.adaptive_width - row.static_width) / row.static_width
        lines.append(rf"\newcommand{{\adaptiveDevDrop{word}}}{{{drop:.1f}\%}}")
        lines.append(rf"\newcommand{{\adaptiveWidthCost{word}}}{{{cost:.1f}\%}}")
    write(out, lines)


def derived_quantities(rel: pd.DataFrame, out: Path) -> None:
    """Emit LaTeX macros for figures quoted in prose.

    The results text needs percentage improvements and gaps that are functions
    of the table entries. Deriving them here keeps the manuscript's rule that no
    number is typed by hand, and keeps prose and tables from drifting apart.
    """
    stats = rel.groupby("horizon").agg(
        rmse=("rmse", "mean"),
        fused=("fused_aurc", "mean"),
        width=("width_only_aurc", "mean"),
        selected=("fused_selected_rmse", "mean"),
        coverage=("conformal_marginal_coverage", "mean"),
        simul_width=("simultaneous_mean_width", "mean"),
        conf_width=("conformal_mean_width", "mean"),
    )
    names = {1: "One", 7: "Seven", 30: "Thirty", 90: "Ninety"}
    lines = ["% Generated by scripts/make_paper_tables.py -- do not edit by hand."]

    def macro(name: str, value: str) -> None:
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    for horizon, word in names.items():
        if horizon not in stats.index:
            continue
        row = stats.loc[horizon]
        macro(f"widthGain{word}", f"{100 * (row.rmse - row.width) / row.rmse:.1f}\\%")
        macro(f"fusedGain{word}", f"{100 * (row.rmse - row.selected) / row.rmse:.1f}\\%")
        macro(f"fusedPenalty{word}", f"{row.fused - row.width:.3f}")
        macro(f"widthRatio{word}", f"{row.simul_width / row.conf_width:.1f}")
    deviation = (stats["coverage"] - float(rel.nominal_coverage.iloc[0])).abs().max()
    macro("maxCoverageDeviation", f"{100 * deviation:.1f}")
    seed_sd = pd.concat([
        rel.groupby("horizon").fused_aurc.std(),
        rel.groupby("horizon").width_only_aurc.std(),
    ])
    macro("aurcSeedSdLow", f"{seed_sd.min():.3f}")
    macro("aurcSeedSdHigh", f"{seed_sd.max():.3f}")
    # What the weight selection concluded is a result, not a setting, so the
    # range it landed in and how it was chosen are read from the ledger.
    if "fusion_weight_width" in rel:
        weights = rel.fusion_weight_width
        macro("fusionWeightLow", f"{weights.min():.2f}")
        macro("fusionWeightHigh", f"{weights.max():.2f}")
        macro("fusionWeightMean", f"{weights.mean():.2f}")
        macro("fusionShiftHigh", f"{1 - weights.min():.2f}")
        chosen = sorted({str(value) for value in rel["fusion_weight_selection"].dropna()})
        macro("fusionSelection", chosen[0] if len(chosen) == 1 else "mixed")
        macro("fusionEqualRuns", f"{int((weights == 0.5).sum())}")
        macro("fusionRuns", f"{len(weights)}")
    write(out, lines)


def provenance(
    bench: pd.DataFrame,
    rel: pd.DataFrame,
    imported: pd.DataFrame | None,
    out: Path,
) -> None:
    """Record which artifacts produced the tables so reviewers can re-derive them."""
    payload = {
        "benchmark": {
            "ledger": str(STUDY.benchmark.relative_to(ROOT).as_posix()),
            "ledger_sha256": hashlib.sha256(STUDY.benchmark.read_bytes()).hexdigest(),
            "runs": int(len(bench)),
            "code_sha256": str(bench.code_sha256.iloc[0]),
            "dataset_sha256": str(bench.dataset_sha256.iloc[0]),
        },
        "reliability": {
            "ledger": str(STUDY.reliability.relative_to(ROOT).as_posix()),
            "ledger_sha256": hashlib.sha256(STUDY.reliability.read_bytes()).hexdigest(),
            "runs": int(len(rel)),
            "code_sha256": str(rel.code_sha256.iloc[0]),
            "nominal_coverage": float(rel.nominal_coverage.iloc[0]),
        },
    }
    if imported is not None and not imported.empty:
        payload["imported"] = {
            "ledger": str(STUDY.imported.relative_to(ROOT).as_posix()),
            "ledger_sha256": hashlib.sha256(STUDY.imported.read_bytes()).hexdigest(),
            "models": sorted(set(imported.model)),
            "runs": int(len(imported)),
            "code_sha256": str(imported.code_sha256.iloc[0]),
            "reason": (
                "refitting the classical shard costs about twenty-five hours; the "
                "source edit is inert for models it does not name, established "
                "bit for bit by scripts/verify_baseline_inertness.py"
            ),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {display_path(out)}")


def defect_macros(bench: pd.DataFrame, abl: pd.DataFrame, out: Path) -> None:
    """How far the superseded readout fell behind the trivial baseline.

    The manuscript describes the defect as leaving the model behind
    persistence. That comparison is only auditable where a persistence run
    exists, and its verdict is not uniform across horizons, so the count and
    the scope are derived here rather than asserted in the text.
    """
    words = {1: "one day", 7: "seven days", 30: "thirty days", 90: "ninety days"}
    persistence = bench[bench.model.eq("persistence")].groupby("horizon").rmse.mean()
    defect = abl[abl.ablation_id.eq("A9")].groupby("horizon").rmse.mean()
    shared = [h for h in HORIZONS if h in persistence.index and h in defect.index]
    behind = [h for h in shared if defect[h] > persistence[h]]
    lines = [
        "% Generated by scripts/make_paper_tables.py -- do not edit by hand.",
        rf"\newcommand{{\defectBehindCount}}{{{len(behind)}}}",
        rf"\newcommand{{\defectMeasuredCount}}{{{len(shared)}}}",
        rf"\newcommand{{\defectBehindHorizons}}{{{oxford([words[h] for h in behind])}}}",
    ]
    if behind:
        gaps = [defect[h] - persistence[h] for h in behind]
        lines.append(rf"\newcommand{{\defectWorstGap}}{{{max(gaps):.2f}}}")
    write(out, lines)


GHCN_PANEL = ROOT / "configs" / "datasets" / "ghcn_frozen.yaml"
GHCN_WINDOWS = ROOT / "results" / "dataset_audits" / "ghcn_full_windows.json"


def ghcn_macros(out: Path) -> None:
    """Describe the frozen GHCN panel from its audit, not from memory.

    Every macro here is a property of the *panel*: how many stations were
    selected, over what years, and how many observations and windows they hold.
    None is a property of a model, because the GHCN campaign has not been run.
    Nothing in this file may ever describe an outcome; the manuscript states
    that no station-generalization result is claimed, and a macro that
    contributed one would make that statement false.

    The protocol being unexecuted is also why there is no run ledger to read.
    There is a window audit, which is the artifact that fixes what the panel
    would supply, and the dataset table must quote the same quantity for GHCN
    as it does for CET: windowed forecast origins, not raw observations.
    """
    import yaml

    panel = yaml.safe_load(GHCN_PANEL.read_text(encoding="utf-8"))
    period = panel["dataset"]["period"]
    stations = panel["stations"]
    lines = [
        "% Generated by scripts/make_paper_tables.py -- do not edit by hand.",
        "% Frozen, unexecuted GHCN protocol: panel description only, no results.",
        rf"\newcommand{{\ghcnStations}}{{{len(stations)}}}",
        rf"\newcommand{{\ghcnFirstYear}}{{{str(period['start'])[:4]}}}",
        rf"\newcommand{{\ghcnLastYear}}{{{str(period['end'])[:4]}}}",
        rf"\newcommand{{\ghcnObservations}}"
        rf"{{{sum(int(s['observations']) for s in stations):,}}}",
    ]
    if GHCN_WINDOWS.exists():
        audit = json.loads(GHCN_WINDOWS.read_text(encoding="utf-8"))
        shortest = min(r["horizon"] for r in audit["station_window_counts"])
        origins = sum(
            sum(r["windows"].values())
            for r in audit["station_window_counts"]
            if r["horizon"] == shortest
        )
        lines.append(rf"\newcommand{{\ghcnOrigins}}{{{origins:,}}}")
    else:
        # Better an explicit gap in the table than a number with no artifact.
        lines.append(r"\newcommand{\ghcnOrigins}{pending audit}")
    write(out, lines)


def prose_macros(bench, rel, abl, eval_stats, ext, out: Path) -> None:
    """Macros for the measured quantities that sentences quote directly.

    Section~\\ref{sec:results} opens by asserting that no number in it is
    transcribed by hand. That assertion holds only if the figures embedded in
    running text come from the same ledgers as the tables beside them, so every
    such figure is derived here instead of being typed into the prose.
    """
    names = {1: "One", 7: "Seven", 30: "Thirty", 90: "Ninety"}
    lines = ["% Generated by scripts/make_paper_tables.py -- do not edit by hand."]

    def macro(name: str, value: str) -> None:
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    # Whether the accuracy deficit is larger than the noise that produced it.
    macro(
        "benchSeedSdTrust",
        f"{bench[bench.model.eq(STUDY.proposed)].groupby('horizon').rmse.std().max():.3f}",
    )

    # Which classical baselines finish ahead of the proposed model at the long
    # horizons. This is named rather than asserted because it is the claim most
    # exposed to the ledger changing: the classical shards were the last to
    # land. If none qualifies the macro is not written, so the manuscript fails
    # to build rather than keeping a sentence the evidence no longer supports.
    rmse = bench.groupby(["model", "horizon"]).rmse.mean().unstack()
    long_horizons = [h for h in (30, 90) if h in rmse.columns]
    ahead = [
        MODEL_LABELS.get(model, model)
        for model in CLASSICAL_MODELS
        if model in rmse.index
        and all(rmse.loc[model, h] < rmse.loc[STUDY.proposed, h] for h in long_horizons)
    ]
    if ahead and long_horizons:
        macro("classicalAheadLong", oxford(sorted(ahead)))

    stats = rel.groupby("horizon").agg(
        marginal=("conformal_marginal_coverage", "mean"),
        joint=("conformal_joint_coverage", "mean"),
        simultaneous=("simultaneous_joint_coverage", "mean"),
        auroc=("error_detection_auroc", "mean"),
    )
    macro("coverageLow", f"{stats.marginal.min():.3f}")
    macro("coverageHigh", f"{stats.marginal.max():.3f}")
    macro("nominalCoverage", f"{float(rel.nominal_coverage.iloc[0]):.2f}")
    for horizon, word in names.items():
        if horizon not in stats.index:
            continue
        row = stats.loc[horizon]
        macro(f"jointCoverage{word}", f"{row.joint:.3f}")
        macro(f"simulCoverage{word}", f"{row.simultaneous:.3f}")
        macro(f"errorAuroc{word}", f"{row.auroc:.3f}")
    macro("errorAurocLow", f"{stats.auroc.min():.3f}")
    macro("errorAurocHigh", f"{stats.auroc.max():.3f}")
    macro(
        "reliabilityRhoMax",
        f"{rel.groupby('horizon').reliability_spearman.mean().abs().max():.3f}",
    )

    if abl is not None:
        mean = abl.groupby(["ablation_id", "horizon"]).rmse.mean().unstack()
        spread = abl.groupby(["ablation_id", "horizon"]).rmse.std().unstack()
        if {"A0", "A1"} <= set(mean.index):
            macro(
                "ablationKanGapMax",
                f"{(mean.loc['A1'] - mean.loc['A0']).abs().max():.3f}",
            )
        if {"A0", "A9"} <= set(mean.index):
            gap = mean.loc["A9"] - mean.loc["A0"]
            for horizon, word in names.items():
                if horizon in gap.index:
                    macro(f"defectGap{word}", f"{gap[horizon]:.2f}")
            first = min(names)
            macro("defectSeedSdOne", f"{spread.loc['A9', first]:.3f}")
            macro("correctedSeedSdOne", f"{spread.loc['A0', first]:.3f}")
        # Each single-element removal is charged separately, which is the point
        # of A10 and A11 existing at all.
        for identifier, key in (("A10", "Stem"), ("A11", "Readout"), ("A9", "Pooled")):
            if {"A0", identifier} <= set(mean.index):
                gap = mean.loc[identifier] - mean.loc["A0"]
                macro(f"ablation{key}GapMax", f"{gap.max():.3f}")
                macro(f"ablation{key}GapMin", f"{gap.min():.3f}")
                for horizon, word in names.items():
                    if horizon in gap.index:
                        macro(f"ablation{key}{word}", f"{gap[horizon]:.3f}")

    if eval_stats is not None and "A7" in eval_stats:
        macro("abstainGainLow", f"{eval_stats.A7.min():.3f}")
        macro("abstainGainHigh", f"{eval_stats.A7.max():.3f}")

    if STUDY.comparisons.exists():
        rows = json.loads(STUDY.comparisons.read_text(encoding="utf-8"))
        # The text speaks of the comparator's advantage as a penalty, so the
        # magnitude is what it needs; the sign is carried by Table 2.
        gaps = {
            (str(r["model_b"]), int(r["horizon"])): abs(float(r["mean_rmse_difference"]))
            for r in rows
        }
        # Every comparator in the family gets macros, so a sentence about one of
        # them cannot outlive its removal from the family.
        comparators = (
            ("transformer", "Transformer"),
            ("kan", "Kan"),
            ("trustkan_dilated", "Dilated"),
            ("trustkan", "Published"),
        )
        for model, key in comparators:
            for horizon, word in names.items():
                if (model, horizon) in gaps:
                    macro(f"paired{key}{word}", f"{gaps[(model, horizon)]:.3f}")

        # The verdict itself is generated, not narrated. Writing "wins" or
        # "indistinguishable" by hand would let the prose survive a change of
        # evidence, which is exactly what these comparisons exist to prevent.
        for comparator, key in comparators:
            family = [r for r in rows if str(r["model_b"]) == comparator]
            if not family:
                continue
            wins = sum(int(r["a_better"]) for r in family)
            losses = sum(int(r["b_better"]) for r in family)
            total = sum(int(r["n_pairs"]) for r in family)
            macro(f"paired{key}Wins", str(wins))
            macro(f"paired{key}Losses", str(losses))
            macro(f"paired{key}Total", str(total))
            macro(
                f"paired{key}Tied",
                str(sum(int(r["inconclusive"]) for r in family)),
            )
            # Which leads the wins land on is the mechanism claim for the
            # readout, so it is counted rather than described.
            won_at = [int(r["horizon"]) for r in family if int(r["a_better"])]
            macro(
                f"paired{key}WinHorizons",
                oxford([f"$h={h}$" for h in sorted(set(won_at))]) if won_at else "none",
            )
            if wins == 0 and losses == 0:
                verdict = "statistically indistinguishable from"
            elif wins > 0 and losses == 0:
                verdict = "better than" if wins == total else "better than or level with"
            elif losses > 0 and wins == 0:
                verdict = "worse than" if losses == total else "level with or worse than"
            else:
                verdict = "mixed against"
            macro(f"paired{key}Verdict", verdict)

    if ext is not None:
        either = ext.groupby(["model", "horizon"]).either_rmse.mean()
        complement = ext.groupby(["model", "horizon"]).complement_rmse.mean()
        counts = ext.groupby("horizon")[["cold_n_origins", "warm_n_origins"]].first()
        last = max(names)
        proposed = STUDY.proposed
        if (proposed, last) in either.index and ("transformer", last) in either.index:
            macro(
                "extremeGapNinety",
                f"{either[(proposed, last)] - either[('transformer', last)]:.3f}",
            )
            macro("extremeTrustNinety", f"{either[(proposed, last)]:.3f}")
            macro("extremeComplementNinety", f"{complement[(proposed, last)]:.3f}")
        warm = counts.warm_n_origins.unique()
        macro("extremeWarmCount", f"{int(warm[0])}" if len(warm) == 1 else "varying")
        macro("extremeColdLow", f"{int(counts.cold_n_origins.min())}")
        macro("extremeColdHigh", f"{int(counts.cold_n_origins.max())}")

    write(out, lines)


def main(outdir: Path, study: str = MANUSCRIPT_STUDY) -> None:
    global STUDY
    STUDY = STUDIES[study]

    from run_cet_benchmark import code_sha256 as benchmark_fingerprint
    from run_cet_reliability import code_sha256 as reliability_fingerprint

    print(f"study {study}: proposed model {STUDY.proposed}")
    print("benchmark ledger")
    bench = load_ok(STUDY.benchmark, benchmark_fingerprint())
    print("reliability ledger")
    rel = load_ok(STUDY.reliability, reliability_fingerprint())
    imported = None
    if STUDY.imported is not None and STUDY.imported.exists():
        print("imported classical rows")
        imported = load_imported(
            STUDY.imported, benchmark_fingerprint(), CLASSICAL_MODELS
        )
        if not imported.empty:
            print(
                f"  quoting {sorted(set(imported.model))} from "
                f"{display_path(STUDY.imported)} under "
                f"{str(imported.code_sha256.iloc[0])[:12]}"
            )
    benchmark_table(bench, outdir / "cet_rmse.tex", imported)
    reliability_table(rel, outdir / "cet_calibration.tex")
    selective_table(rel, outdir / "cet_selective.tex")
    efficiency_table(bench, outdir / "cet_efficiency.tex")
    derived_quantities(rel, outdir / "cet_derived.tex")
    dataset_macros(rel, outdir / "cet_dataset.tex")
    ext = None
    if STUDY.extremes.exists():
        # The extreme subsets re-score the benchmark's stored predictions, so
        # the fingerprint they must match is the benchmark's. Without this the
        # extremes table was the one table that could carry superseded runs.
        print("extremes ledger")
        ext = load_ok(STUDY.extremes, benchmark_fingerprint(), has_status=False)
        if STUDY.imported is not None:
            imported_ext = load_imported(
                STUDY.extremes,
                benchmark_fingerprint(),
                CLASSICAL_MODELS,
                has_status=False,
            )
            if imported_ext.empty:
                frozen_ext = ROOT / "results" / "extremes" / "cet_extremes_runs.csv"
                if frozen_ext.exists():
                    imported_ext = load_imported(
                        frozen_ext,
                        benchmark_fingerprint(),
                        CLASSICAL_MODELS,
                        has_status=False,
                    )
            if not imported_ext.empty:
                print(
                    f"  quoting extremes for {sorted(set(imported_ext.model))} "
                    f"under {str(imported_ext.code_sha256.iloc[0])[:12]}"
                )
                ext = pd.concat([ext, imported_ext], ignore_index=True)
        extremes_table(ext, outdir / "cet_extremes.tex")
    else:
        print("  skipping extremes table; run scripts/run_cet_extremes.py first")
    abl, eval_stats = None, None
    if STUDY.ablations.exists():
        from run_ablations import code_sha256 as ablation_fingerprint

        print("ablation ledger")
        abl = load_ok(STUDY.ablations, ablation_fingerprint())
        trained_ablation_table(abl, outdir / "cet_ablation.tex")
        defect_macros(bench, abl, outdir / "cet_defect.tex")
        eval_stats = evaluation_ablation_table(outdir / "cet_eval_ablation.tex")
    else:
        print("  skipping ablation tables; run scripts/run_ablations.py first")
    if STUDY.comparisons.exists():
        comparison_table(STUDY.comparisons, outdir / "cet_paired.tex")
    else:
        print("  skipping paired table; run scripts/run_primary_comparisons.py first")
    prose_macros(bench, rel, abl, eval_stats, ext, outdir / "cet_prose.tex")
    ghcn_macros(outdir / "ghcn_frozen_panel.tex")
    provenance(bench, rel, imported, outdir / "cet_tables_provenance.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(ROOT / "paper" / "tables"))
    parser.add_argument("--study", choices=sorted(STUDIES), default=MANUSCRIPT_STUDY)
    args = parser.parse_args()
    main(Path(args.outdir), args.study)
