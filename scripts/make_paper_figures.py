"""Generate the manuscript's figures from the frozen reliability artifacts.

Nothing is recomputed that the runners already stored: risk-coverage curves,
per-origin reliability scores and errors, and the static-versus-adaptive rolling
coverage trajectories are read back and plotted. Curves are averaged across the
five seeds on a common coverage grid so the figures carry the same seed
aggregation as the tables.

Like the table generator, this module sits outside the `code_sha256` set, so
redrawing figures never invalidates a completed run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import _bootstrap  # noqa: F401  # repository-root import setup

ROOT = Path(__file__).resolve().parents[1]
# Each study writes to its own experiment names so the corrected architecture
# could be evaluated without disturbing the published artifacts; the figures
# have to follow the same split or they would mix the two.
STUDY_PATHS = {
    "v1": {
        "reliability": ROOT / "results" / "reliability" / "raw" / "cet_reliability_full",
        "ablations": ROOT / "results" / "ablations" / "raw" / "ablations_cet_full",
        "config": ROOT / "configs" / "ablations.yaml",
        "prefix": "cet_trustkan_reliability",
    },
    "v2": {
        "reliability": ROOT / "results" / "reliability" / "raw" / "cet_reliability_v2",
        "ablations": ROOT / "results" / "ablations" / "raw" / "ablations_cet_v2",
        "config": ROOT / "configs" / "ablations_v2.yaml",
        "prefix": "cet_trustkan_reliability",
    },
}
RELIABILITY_RAW = STUDY_PATHS["v1"]["reliability"]
ABLATION_RAW = STUDY_PATHS["v1"]["ablations"]
ABLATION_CONFIG = STUDY_PATHS["v1"]["config"]
HORIZONS = (1, 7, 30, 90)
SEEDS = (11, 22, 33, 44, 55)
GRID = np.linspace(0.05, 1.0, 96)
NOMINAL = 0.90

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.1,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)
STYLES = {
    "fused": ("Fused reliability", "#1f77b4", "-"),
    "width_only": ("Interval width only", "#d62728", "--"),
    "shift_only": ("Embedding shift only", "#7f7f7f", ":"),
}


def artifact(horizon: int, seed: int) -> Path:
    return RELIABILITY_RAW / f"cet_trustkan_reliability_h{horizon}_s{seed}.npz"


def panel_grid(ylabel: str, xlabel: str):
    """Four horizon panels sharing one caption-level axis label and legend.

    The label goes on each panel rather than on the figure because a figure-wide
    label and an outside legend compete for the same strip of space and collide.
    """
    figure, axes = plt.subplots(1, 4, figsize=(7.1, 2.0), layout="constrained")
    for axis, horizon in zip(axes, HORIZONS):
        axis.set_title(f"$h={horizon}$")
        axis.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    return figure, axes


def shared_legend(figure, axis, ncol: int) -> None:
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False)


def save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp{path.suffix}")
    try:
        figure.savefig(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    plt.close(figure)
    print(f"  wrote {path.relative_to(ROOT)}")


def mean_curve(horizon: int, key: str):
    """Average risk-coverage curves across seeds on the shared coverage grid."""
    curves = []
    for seed in SEEDS:
        path = artifact(horizon, seed)
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as source:
            coverage = np.asarray(source[f"{key}_coverage"], dtype=float)
            risk = np.asarray(source[f"{key}_risk"], dtype=float)
        order = np.argsort(coverage)
        curves.append(np.interp(GRID, coverage[order], risk[order]))
    return np.mean(curves, axis=0) if curves else None


def risk_coverage_figure(out: Path) -> None:
    figure, axes = panel_grid(r"RMSE on retained set ($^{\circ}$C)", "Retained fraction")
    for axis, horizon in zip(axes, HORIZONS):
        for key, (label, colour, dash) in STYLES.items():
            curve = mean_curve(horizon, key)
            if curve is not None:
                axis.plot(GRID, curve, color=colour, linestyle=dash, label=label)
    shared_legend(figure, axes[0], 3)
    save(figure, out)


def reliability_error_figure(out: Path, n_bins: int = 10) -> None:
    """Binned error against reliability decile, with the no-skill reference."""
    figure, axes = panel_grid(r"Mean absolute error ($^{\circ}$C)", "Reliability decile")
    edges = np.linspace(0, 1, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for axis, horizon in zip(axes, HORIZONS):
        for key, (label, colour, dash) in STYLES.items():
            score_field = {"fused": "fused", "width_only": "width", "shift_only": "shift"}[key]
            binned, overall = [], []
            for seed in SEEDS:
                path = artifact(horizon, seed)
                if not path.exists():
                    continue
                with np.load(path, allow_pickle=False) as source:
                    score = np.asarray(source[f"{score_field}_reliability"], dtype=float)
                    error = np.asarray(source["test_error"], dtype=float)
                # Rank-transform to deciles so the x axis is comparable across
                # scores with different and unbounded scales.
                rank = score.argsort().argsort() / max(len(score) - 1, 1)
                index = np.clip(np.digitize(rank, edges) - 1, 0, n_bins - 1)
                binned.append([error[index == b].mean() for b in range(n_bins)])
                overall.append(error.mean())
            if binned:
                axis.plot(centres, np.mean(binned, axis=0), color=colour,
                          linestyle=dash, marker="o", markersize=2.2, label=label)
                axis.axhline(np.mean(overall), color="black", linewidth=0.6, alpha=0.6)
    shared_legend(figure, axes[0], 3)
    save(figure, out)


def coverage_window() -> int:
    """The rolling window the ablation runner used, read rather than assumed."""
    config = yaml.safe_load(ABLATION_CONFIG.read_text(encoding="utf-8"))
    for section in (config.get("adaptive", {}), config):
        if isinstance(section, dict) and "coverage_window" in section:
            return int(section["coverage_window"])
    raise SystemExit(f"no coverage_window in {ABLATION_CONFIG.relative_to(ROOT)}")


def per_origin_rolling(hit: np.ndarray, window: int) -> np.ndarray:
    """Coverage averaged over an origin's lead times, then smoothed over origins.

    The ablation runner stores a rolling window applied to the flattened
    (origin, lead time) grid, which for the long horizons spans barely one
    origin and therefore measures within-origin correlation rather than drift
    over the test period. Collapsing lead times first gives an axis that is
    genuinely time.
    """
    per_origin = hit.astype(float).mean(axis=1)
    out = np.full(len(per_origin), np.nan)
    for i in range(window - 1, len(per_origin)):
        out[i] = per_origin[i - window + 1 : i + 1].mean()
    return out


def rolling_coverage_figure(out: Path, window: int) -> None:
    figure, axes = panel_grid("Rolling coverage", "Test origin")
    for axis, horizon in zip(axes, HORIZONS):
        static, adaptive = [], []
        for seed in SEEDS:
            path = ABLATION_RAW / f"ablation_A0_h{horizon}_s{seed}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as source:
                target = np.asarray(source["test_target"], dtype=float)
                static.append(
                    per_origin_rolling(
                        (target >= source["test_lower_horizonwise"])
                        & (target <= source["test_upper_horizonwise"]),
                        window,
                    )
                )
                adaptive.append(
                    per_origin_rolling(
                        (target >= source["adaptive_lower"]) & (target <= source["adaptive_upper"]),
                        window,
                    )
                )
        if not static:
            continue
        length = min(len(a) for a in static)
        x = np.arange(length)
        axis.plot(x, np.mean([a[:length] for a in static], axis=0),
                  color="#d62728", linestyle="--", label="Static split conformal")
        axis.plot(x, np.mean([a[:length] for a in adaptive], axis=0),
                  color="#1f77b4", label="Adaptive conformal")
        axis.axhline(NOMINAL, color="black", linewidth=0.6, label="Nominal $1-\\alpha=0.90$")
        axis.set_ylim(0.6, 1.02)
    shared_legend(figure, axes[0], 3)
    save(figure, out)


CURVE_RAW = ROOT / "results" / "interpretability" / "raw" / "cet_kan_curves"
CURVE_STABILITY = (
    ROOT / "results" / "interpretability" / "aggregated" / "cet_kan_curves_stability.csv"
)
CURVE_SHAPES = (
    ROOT / "results" / "interpretability" / "aggregated" / "cet_kan_curves_shapes.csv"
)


def kan_curve_figure(out: Path, horizon: int = 1, examples: int = 4) -> bool:
    """Representative learned curves overlaid across seeds, plus the agreement.

    The first panels answer what a reader is invited to do with an
    interpretable model: look at curve k and read a meaning off it. Overlaying
    the five seeds shows whether "curve k" denotes anything stable. The last
    panel contrasts that index-matched agreement with the best-match bound,
    which forgives the latent channels' permutation freedom.
    """
    curves = {}
    for seed in SEEDS:
        path = CURVE_RAW / f"kan_curves_h{horizon}_s{seed}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as source:
                curves[seed] = (
                    np.asarray(source["curves"], dtype=float),
                    np.asarray(source["x_grid"], dtype=float),
                )
    if len(curves) < 2:
        print("  skipping KAN curve figure; need at least two seeds")
        return False
    figure, axes = plt.subplots(1, examples + 1, figsize=(7.1, 2.1), layout="constrained")
    reference = curves[sorted(curves)[0]][0]
    # Show the curves that carry the most structure rather than an arbitrary
    # slice, so the panels are the strongest case for the interpretation.
    ranked = np.argsort(reference.std(axis=1))[::-1]
    chosen = [int(i) for i in ranked[:: max(1, len(ranked) // examples)][:examples]]
    palette = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(curves)))
    for axis, index in zip(axes, chosen):
        for colour, seed in zip(palette, sorted(curves)):
            values, grid = curves[seed]
            axis.plot(grid, values[index], color=colour, linewidth=1.0, label=f"seed {seed}")
        axis.set_title(f"curve {index}")
        axis.set_xlabel("latent input")
    axes[0].set_ylabel("curve output")
    shared_legend(figure, axes[0], len(curves))

    summary = axes[examples]
    if CURVE_STABILITY.is_file():
        table = pd.read_csv(CURVE_STABILITY)
        table = table[table.horizon.eq(horizon)]
        # The within-seed control belongs next to the best-match bar: without
        # it, a best match near one reads as cross-seed reproducibility when it
        # is just a consequence of how few distinct shapes the layer learns.
        series = [
            ("matched", table.matched_correlation_mean, "#d62728"),
            ("best", table.best_match_correlation_mean, "#1f77b4"),
            ("control", table.within_seed_best_match_mean, "#7f7f7f"),
        ]
        for position, (_, values, colour) in enumerate(series):
            summary.scatter(
                np.full(len(values), position),
                values.to_numpy(),
                s=9,
                color=colour,
                alpha=0.7,
                edgecolors="none",
            )
        summary.set_xticks(range(len(series)))
        summary.set_xticklabels([label for label, _, _ in series], fontsize=6)
        summary.set_xlim(-0.5, len(series) - 0.5)
        summary.set_ylim(-0.15, 1.08)
        summary.axhline(0.0, color="black", linewidth=0.6)
    summary.set_title("cross-seed agreement")
    summary.set_ylabel("mean correlation")
    save(figure, out)
    return True


def curve_macros(out: Path) -> None:
    """Macros for the interpretability prose, derived from the curve ledgers."""
    if not (CURVE_STABILITY.is_file() and CURVE_SHAPES.is_file()):
        print("  skipping curve macros; run scripts/analyze_kan_curves.py first")
        return
    table = pd.read_csv(CURVE_STABILITY)
    shapes = pd.read_csv(CURVE_SHAPES)
    names = {1: "One", 7: "Seven", 30: "Thirty", 90: "Ninety"}
    lines = ["% Generated by scripts/make_paper_figures.py -- do not edit by hand."]

    def macro(name: str, value: str) -> None:
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    for horizon, word in names.items():
        subset = table[table.horizon.eq(horizon)]
        if subset.empty:
            continue
        macro(f"curveMatched{word}", f"{subset.matched_correlation_mean.mean():+.3f}")
        macro(f"curveBestMatch{word}", f"{subset.best_match_correlation_mean.mean():.4f}")
        macro(f"curveControl{word}", f"{subset.within_seed_best_match_mean.mean():.4f}")
    macro("curveMatchedAbs", f"{table.matched_correlation_abs_mean.mean():.3f}")
    macro("curveShuffledAbs", f"{table.shuffled_correlation_abs_mean.mean():.3f}")
    macro(
        "curveAbsExcessMax",
        f"{table.matched_abs_excess_over_shuffled.abs().max():.3f}",
    )
    macro("curveMatchedLow", f"{table.matched_correlation_mean.min():+.3f}")
    macro("curveMatchedHigh", f"{table.matched_correlation_mean.max():+.3f}")
    macro("curveBestMatchLow", f"{table.best_match_correlation_mean.min():.4f}")
    macro("curveBestMatchHigh", f"{table.best_match_correlation_mean.max():.4f}")
    macro("curveExcessMax", f"{table.best_match_excess_over_control.abs().max():.4f}")
    macro("curveNonlinearShare", f"{shapes.nonlinear_share_mean.mean():.3f}")
    macro("curveLinearMedian", f"{shapes.linear_r2_median.mean():.3f}")
    macro("curveNearlyLinear", f"{100 * shapes.nearly_linear_fraction.mean():.0f}\\%")
    macro("curveRankMax", f"{int(shapes.effective_shape_rank.max())}")
    macro("curveCount", f"{int(shapes.n_curves.iloc[0]):,}")
    macro("curveRuns", f"{len(shapes)}")
    macro("curveSeedPairs", f"{len(table[table.horizon.eq(1)])}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(ROOT)}")


def rolling_macros(out: Path, window: int) -> None:
    """Emit the per-origin A8 comparison that Figure 3 is drawn from.

    The runner's stored deviation flattens origins and lead times together, so
    the manuscript has to report both definitions. Deriving the per-origin
    numbers here keeps them out of the prose by hand and ties them to the same
    artifacts the figure uses.
    """
    names = {1: "One", 7: "Seven", 30: "Thirty", 90: "Ninety"}
    lines = ["% Generated by scripts/make_paper_figures.py -- do not edit by hand."]
    drops, floors = [], []
    for horizon, word in names.items():
        static, adaptive = [], []
        for seed in SEEDS:
            path = ABLATION_RAW / f"ablation_A0_h{horizon}_s{seed}.npz"
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as source:
                target = np.asarray(source["test_target"], dtype=float)
                static.append(
                    per_origin_rolling(
                        (target >= source["test_lower_horizonwise"])
                        & (target <= source["test_upper_horizonwise"]),
                        window,
                    )
                )
                adaptive.append(
                    per_origin_rolling(
                        (target >= source["adaptive_lower"]) & (target <= source["adaptive_upper"]),
                        window,
                    )
                )
        if not static:
            continue

        def deviation(curves):
            values = np.concatenate([c[np.isfinite(c)] for c in curves])
            return float(np.abs(values - NOMINAL).mean())

        static_dev, adaptive_dev = deviation(static), deviation(adaptive)
        drop = 100 * (static_dev - adaptive_dev) / static_dev
        drops.append(drop)
        length = min(len(a) for a in static)
        floors.append(float(np.nanmin(np.mean([a[:length] for a in static], axis=0))))
        lines.append(rf"\newcommand{{\adaptiveOriginDrop{word}}}{{{drop:.1f}\%}}")
    if drops:
        lines.append(rf"\newcommand{{\adaptiveOriginDropLow}}{{{min(drops):.1f}\%}}")
        lines.append(rf"\newcommand{{\adaptiveOriginDropHigh}}{{{max(drops):.1f}\%}}")
    if floors:
        lines.append(rf"\newcommand{{\staticCoverageFloor}}{{{min(floors):.2f}}}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(ROOT)}")


def select_study(study: str) -> None:
    """Point every reader at one study's artifacts before any figure is drawn."""
    global RELIABILITY_RAW, ABLATION_RAW, ABLATION_CONFIG
    global CURVE_RAW, CURVE_STABILITY, CURVE_SHAPES
    paths = STUDY_PATHS[study]
    RELIABILITY_RAW = paths["reliability"]
    ABLATION_RAW = paths["ablations"]
    ABLATION_CONFIG = paths["config"]
    suffix = "" if study == "v1" else f"_{study}"
    interpretability = ROOT / "results" / "interpretability"
    CURVE_RAW = interpretability / "raw" / f"cet_kan_curves{suffix}"
    CURVE_STABILITY = (
        interpretability / "aggregated" / f"cet_kan_curves{suffix}_stability.csv"
    )
    CURVE_SHAPES = interpretability / "aggregated" / f"cet_kan_curves{suffix}_shapes.csv"


def main(outdir: Path, study: str = "v1") -> None:
    select_study(study)
    if not RELIABILITY_RAW.exists():
        raise SystemExit(f"missing {RELIABILITY_RAW.relative_to(ROOT)}")
    fingerprints = set()
    for horizon in HORIZONS:
        for seed in SEEDS:
            path = artifact(horizon, seed)
            if path.exists():
                with np.load(path, allow_pickle=False) as source:
                    fingerprints.add(str(source["code_sha256"].item()))
    if len(fingerprints) > 1:
        raise SystemExit(
            "reliability artifacts span multiple code fingerprints "
            f"({sorted(f[:12] for f in fingerprints)}); re-run the campaign"
        )
    print(f"code fingerprint {next(iter(fingerprints))[:12]}")
    # Two outputs are LaTeX macro files rather than images, so they belong with
    # the tables. Deriving that directory from --outdir keeps them inside
    # whatever tree is being built, which is what the Overleaf bundle needs.
    tables = outdir.parent / "tables"
    window = coverage_window()
    risk_coverage_figure(outdir / "risk_coverage.pdf")
    reliability_error_figure(outdir / "reliability_vs_error.pdf")
    rolling_coverage_figure(outdir / "rolling_coverage.pdf", window)
    rolling_macros(tables / "cet_rolling.tex", window)
    if kan_curve_figure(outdir / "kan_curves.pdf"):
        curve_macros(tables / "cet_curves.tex")
    (outdir / "figure_provenance.json").write_text(
        json.dumps(
            {
                "code_sha256": sorted(fingerprints)[0],
                "coverage_window": window,
                "reliability_artifacts": str(RELIABILITY_RAW.relative_to(ROOT).as_posix()),
                "ablation_artifacts": str(ABLATION_RAW.relative_to(ROOT).as_posix()),
                "seeds": list(SEEDS),
                "horizons": list(HORIZONS),
                "rolling_coverage_definition": "per-origin lead times collapsed, then smoothed over origins",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {(outdir / 'figure_provenance.json').relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(ROOT / "paper" / "figures"))
    parser.add_argument("--study", choices=sorted(STUDY_PATHS), default="v1")
    args = parser.parse_args()
    main(Path(args.outdir), args.study)
