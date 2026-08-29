"""Summarize the corruption sweep and measure the mechanism behind it.

The sweep's most distinctive result is that masking the most recent history
block costs far more than any noise or scattered-missingness level, and that for
some architectures the cost stops growing once the block passes a certain
length. That saturation point is a property of the architecture, not of the
data: it is the number of trailing timesteps the readout can still see.

That number is measured by scripts/run_receptive_field.py, on retrained weights
that reproduce the benchmark ledger, and is read here rather than recomputed.
Measuring it on a freshly initialized model would answer a question about the
architecture; the paper needs the answer for the models it reports, so the
measurement belongs with the runners that can prove which models those are.
Reported beside the sweep, it turns "the model is fragile to recent gaps" into a
statement about how much of the history each architecture actually consumes.

Whether saturation happens at all is a finding, not a premise, so the caption
and the macros are derived from the measurement in both studies.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import _bootstrap  # noqa: F401  # file-path execution
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  # module execution

ROOT = Path(__file__).resolve().parents[1]
AGGREGATED = ROOT / "results" / "robustness" / "aggregated"
LABELS = {
    "trustkan": "TrustKAN (published)",
    "trustkan_dilated": "TrustKAN, wide stem",
    "trustkan_v2": "TrustKAN v2",
    "kan": "KAN (plain)",
    "transformer": "Transformer",
}
# LaTeX macro names cannot carry digits or underscores, so each measured model
# needs a short alphabetic tag. "Trust" stays attached to whichever model the
# study is arguing about, so the manuscript's \robustFieldTrust always names the
# proposed architecture.
SHORT = {
    "kan": "Kan",
    "transformer": "Transformer",
    "trustkan": "Published",
    "trustkan_dilated": "Dilated",
    "trustkan_v2": "TrustTwo",
}
STUDIES = {
    "v1": {
        "grid": AGGREGATED / "cet_robustness_grid.csv",
        "fields": AGGREGATED / "cet_receptive_fields.csv",
        "proposed": "trustkan",
        "labels": {**LABELS, "trustkan": "TrustKAN"},
    },
    "v2": {
        "grid": AGGREGATED / "cet_robustness_v2_grid.csv",
        "fields": AGGREGATED / "cet_receptive_fields_v2.csv",
        "proposed": "trustkan_v2",
        "labels": LABELS,
    },
}
# See scripts/make_paper_tables.py: the default is the manuscript's study so a
# bare invocation cannot swap the committed tables for another campaign's.
MANUSCRIPT_STUDY = "v2"
STUDY = STUDIES[MANUSCRIPT_STUDY]
KIND_LABELS = {
    "noise": "Gaussian noise",
    "random_missing": "Random missing",
    "block_missing": "Recent block missing",
}


def label(model: str) -> str:
    return STUDY["labels"].get(model, model)


def short(model: str) -> str:
    return "Trust" if model == STUDY["proposed"] else SHORT.get(model, model.title())


def receptive_fields() -> pd.DataFrame:
    """One reach per model, from the runner's ledger-verified measurements.

    The runner measures every horizon separately. A reach that varied with the
    horizon would mean the number describes a trained instance rather than the
    architecture, so disagreement is refused rather than averaged away.
    """
    path = STUDY["fields"]
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(ROOT)}; run scripts/run_receptive_field.py"
        )
    frame = pd.read_csv(path)
    spread = frame.groupby("model").receptive_field.nunique()
    if (spread > 1).any():
        raise SystemExit(
            f"receptive field varies with horizon for {list(spread[spread > 1].index)}; "
            "it cannot be reported as a property of the architecture"
        )
    if not frame.reproduced.all():
        raise SystemExit("some receptive-field runs did not reproduce the ledger RMSE")
    rows = frame.groupby("model", as_index=False).agg(
        history=("history", "first"),
        receptive_field=("receptive_field", "first"),
        control_receptive_field=("control_receptive_field", "first"),
        horizons=("horizon", "nunique"),
    )
    rows["history_used"] = rows.receptive_field / rows.history
    for row in rows.itertuples():
        print(
            f"  {label(row.model):<22} sees {row.receptive_field} of {row.history} "
            f"timesteps across {row.horizons} horizons "
            f"(untrained control {row.control_receptive_field})"
        )
    return rows


def swept_models(grid: pd.DataFrame) -> list[str]:
    """Corruption-sweep models in a stable order, the proposed one first."""
    present = set(grid.model.unique())
    ordered = [STUDY["proposed"]] + [m for m in LABELS if m != STUDY["proposed"]]
    return [m for m in ordered if m in present]


def caption(grid: pd.DataFrame, fields: pd.DataFrame, models: list[str]) -> str:
    """State what the sweep found, from the sweep, including whether it saturated.

    The v1 caption asserted that the proposed model's block cost stops growing
    past three days. Under a stem that reaches the whole window it does not, and
    a caption that cannot report the opposite outcome is not a measurement.
    """
    reach = fields.set_index("model").receptive_field
    history = int(fields.history.iloc[0])
    saturation = block_saturation(grid)
    worst = (
        grid[grid.kind.ne("clean")]
        .groupby("model")
        .relative_increase.mean()
        .idxmax()
    )
    saturated = [m for m in models if saturation.get(m, 0.0) > 0.5]
    spans = [m for m in models if int(reach.get(m, 0)) >= history]
    text = (
        r"Robustness on CET: percentage increase in test RMSE over the clean "
        r"forecast, averaged over four horizons and five seeds, on the "
        r"pre-registered corruption grid applied to test histories only. "
        r"Masking the most recent days costs more than any other corruption for "
        rf"every model, and most of all for {label(worst)}. "
    )
    if saturated:
        text += (
            "The cost stops growing past three days for "
            + oxford([label(m) for m in saturated])
            + ", which is where the readout's reach ends. "
        )
    else:
        text += "No model's cost saturates with block length. "
    if spans:
        text += (
            "The bottom row reports the measurement behind that: "
            + oxford([label(m) for m in spans])
            + f" can see all {history} timesteps."
        )
    return text


def oxford(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def robustness_table(grid: pd.DataFrame, fields: pd.DataFrame, out: Path) -> None:
    models = swept_models(grid)
    stats = (
        grid[grid.kind.ne("clean")]
        .groupby(["model", "kind", "level"])
        .relative_increase.mean()
        .mul(100)
    )
    reach = fields.set_index("model").receptive_field
    lines = [
        "% Generated by scripts/analyze_robustness.py -- do not edit by hand.",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption(grid, fields, models)}}}",
        r"\label{tab:cet_robustness}",
        r"\begin{tabular}{ll" + "c" * len(models) + "}",
        r"\toprule",
        r"Corruption & Level & "
        + " & ".join(label(name) for name in models)
        + r" \\",
        r"\midrule",
    ]
    for kind, kind_label in KIND_LABELS.items():
        levels = sorted({level for name, k, level in stats.index if k == kind})
        for index, level in enumerate(levels):
            cells = []
            for name in models:
                key = (name, kind, level)
                cells.append(f"{stats[key]:.1f}" if key in stats.index else "--")
            shown = f"{level:g}" if kind != "block_missing" else f"{int(level)}\\,d"
            first = kind_label if index == 0 else ""
            lines.append(f"{first} & {shown} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")
    lines += [
        r"\midrule",
        r"History visible to the readout & &"
        + " & ".join(
            f" {int(reach[name])} of {int(fields.history.iloc[0])} " for name in models
        )
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {log_path(out)}")


def log_path(path: Path) -> str:
    """Path for logging, which must never be able to abort a write."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def block_saturation(grid: pd.DataFrame) -> pd.Series:
    """Share of runs where lengthening the masked block changes nothing at all.

    Past the readout's receptive field the extra masked steps are unreachable,
    so the forecast is not merely similar but identical. That distinguishes an
    architectural limit from a model that simply weights recent days heavily.
    """
    block = grid[grid.kind.eq("block_missing")]
    wide = block.pivot_table(
        index=["model", "horizon", "model_seed"], columns="level", values="rmse"
    )
    identical = (wide[3.0] - wide[7.0]).abs() < 1e-12
    return identical.groupby(level="model").mean()


def macros(grid: pd.DataFrame, fields: pd.DataFrame, out: Path) -> None:
    indexed = fields.set_index("model")
    reach = indexed.receptive_field
    control = indexed.control_receptive_field
    saturation = block_saturation(grid)
    corrupted = grid[grid.kind.ne("clean")]
    lines = ["% Generated by scripts/analyze_robustness.py -- do not edit by hand."]

    def macro(name: str, value: str) -> None:
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    # Every model with a measured reach gets its field macros, including the
    # ones measured for contrast but not swept; the sweep macros are only
    # defined where a sweep exists.
    for model in indexed.index:
        tag = short(model)
        macro(f"robustField{tag}", f"{int(reach[model])}")
        macro(f"robustControl{tag}", f"{int(control[model])}")
        macro(f"robustShare{tag}", f"{100 * reach[model] / indexed.history[model]:.0f}\\%")
    for model in sorted(set(corrupted.model)):
        tag = short(model)
        subset = corrupted[corrupted.model.eq(model)]
        macro(f"robustSaturated{tag}", f"{100 * saturation.get(model, 0):.0f}\\%")
        block = subset[subset.kind.eq("block_missing")]
        macro(f"robustBlockOne{tag}", pct(block[block.level.eq(1)]))
        macro(f"robustBlockThree{tag}", pct(block[block.level.eq(3)]))
        macro(f"robustBlockSeven{tag}", pct(block[block.level.eq(7)]))
        noise = subset[subset.kind.eq("noise") & subset.level.eq(0.20)]
        macro(f"robustNoise{tag}", pct(noise))
        missing = subset[subset.kind.eq("random_missing") & subset.level.eq(0.40)]
        macro(f"robustMissing{tag}", pct(missing))
    macro("robustHistory", f"{int(fields.history.iloc[0])}")
    macro("robustFieldHorizons", f"{int(fields.horizons.iloc[0])}")
    macro("robustFieldModels", f"{len(indexed)}")
    macro("robustSeeds", f"{grid.model_seed.nunique()}")
    macro("robustRuns", f"{len(grid.groupby(['model', 'horizon', 'model_seed']))}")
    macro("robustLevels", f"{corrupted.groupby(['kind', 'level']).ngroups}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {log_path(out)}")


def pct(frame: pd.DataFrame) -> str:
    return "--" if frame.empty else f"{100 * frame.relative_increase.mean():.1f}\\%"


def main(outdir: Path, study: str = MANUSCRIPT_STUDY) -> None:
    global STUDY
    STUDY = STUDIES[study]
    # A relative --outdir, which is how docs/RESULT_PRODUCTION_WORKFLOW.md
    # invokes this script, used to abort between the two output files and leave
    # the table regenerated and its macros stale.
    outdir = Path(outdir).resolve()
    print(f"study {study}: proposed model {STUDY['proposed']}")
    print("receptive fields")
    fields = receptive_fields()
    grid_path = STUDY["grid"]
    if not grid_path.exists():
        raise SystemExit(
            f"missing {grid_path.relative_to(ROOT)}; run scripts/run_robustness_campaign.py"
        )
    grid = pd.read_csv(grid_path)
    print("corruption grid")
    robustness_table(grid, fields, outdir / "cet_robustness.tex")
    macros(grid, fields, outdir / "cet_robustness_macros.tex")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(ROOT / "paper" / "tables"))
    parser.add_argument("--study", choices=sorted(STUDIES), default=MANUSCRIPT_STUDY)
    args = parser.parse_args()
    main(Path(args.outdir), args.study)
