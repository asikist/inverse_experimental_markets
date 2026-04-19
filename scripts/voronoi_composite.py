"""Regenerate the four CEP and AE Voronoi composite PDFs.

Each composite shows, for a given (round, deals) partition, the per-game
winning model type (GBT = Non-Linear, others = Linear) placed in 2D
buyer/seller valuation-distance space and tessellated with Voronoi cells.
Layout: wide main panel on the left, zoom (-0.5, 0.5)^2 panel on the right,
connected by two dashed "zoom-lens" lines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy.spatial import Voronoi

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
REPO_ROOT = CODE_ROOT.parent
FIGURES_ROOT = REPO_ROOT / "paper" / "epj" / "figures"

MODELS = ["EMH", "CEMH", "OB-RLM", "GBT"]

TASKS = {
    "ce": {
        "result_files": {
            "EMH": "data/results/ce_price/emh.ft",
            "CEMH": "data/results/ce_price/cemh.ft",
            "OB-RLM": "data/results/ce_price/ob_rlm.ft",
            "GBT": "data/results/ce_price/gbt.ft",
        },
        "ape_col": "ce_ape",
        "out_dir": FIGURES_ROOT / "cep_voronoi",
        # Only GBT is considered Non-Linear in CEP.
        "non_linear_models": {"GBT"},
    },
    "ae": {
        "result_files": {
            "EMH": "data/results/allocative_efficiency/emh.ft",
            "CEMH": "data/results/allocative_efficiency/cemh.ft",
            "OB-RLM": "data/results/allocative_efficiency/ob_rlm.ft",
            "GBT": "data/results/allocative_efficiency/gbt.ft",
        },
        "ape_col": "ae_ape",
        "out_dir": FIGURES_ROOT / "voronoi",
        # AE considers both CEMH and GBT as Non-Linear.
        "non_linear_models": {"CEMH", "GBT"},
    },
}

TYPE_FACE = {"Non-Linear": "#9A4FB0", "Linear": "#6AAE7C"}  # purple, green
TYPE_POINT = {"Non-Linear": "#4B1F63", "Linear": "#1F5E2E"}  # darker dots
DASH_COLOR = "#9A9A9A"
DASH_STYLE = (0, (6, 5))  # grayer, more spaced

# Each entry:
#   (stem, title_round, title_deal, round_spec, deal_spec,
#    ce_filename_suffix, ae_filename_suffix)
# The 4th AE composite filename in the paper uses a double underscore before
# "all" (the CE version does not), so the suffixes differ.
QUADRANTS = [
    ("voronoi_scatter_1_0",             "1",  "0",  [1],       [0],   "_all", "_all"),
    ("voronoi_scatter_1__more_1",       "1",  ">1", [1],       "pos", "_all", "_all"),
    ("voronoi_scatter__more_2_0",       ">2", "0",  "round>1", [0],   "_all", "_all"),
    ("voronoi_scatter__more_2__more_1", ">2", ">1", "round>1", "pos", "_all", "__all"),
]


def voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)].tolist()
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def compute_distances(original_df: pd.DataFrame, relevant_rounds: Iterable[int]) -> pd.DataFrame:
    df = original_df.query("round in @relevant_rounds").copy()

    all_a, all_b, games, treatments = [], [], [], []
    for t in df.treatment.unique():
        tdf = df.query("treatment == @t")
        for g in tdf.game.unique():
            sg = tdf.query("game == @g")
            vals = sg.groupby(["side", "id"]).valuation.first().to_frame()
            qa = vals.query("side == 'Seller'").quantile(np.linspace(0, 1, 11))
            qb = vals.query("side == 'Buyer'").quantile(np.linspace(0, 1, 11))
            all_a.append(qa)
            all_b.append(qb)
            games.append(g)
            treatments.append(t)

    bvec = np.asarray(all_b)[..., 0]
    np.random.seed(0)
    bvec = bvec + np.random.randn(*bvec.shape) / 10
    bvec = bvec / np.max(all_b)

    avec = np.asarray(all_a)[..., 0]
    np.random.seed(1)
    avec = avec + np.random.randn(*avec.shape) / 10
    avec = avec / np.max(all_a)

    mb = np.median(bvec, axis=0)
    ma = np.median(avec, axis=0)
    distb = (bvec - mb).sum(axis=1)
    dists = (avec - ma).sum(axis=1)

    out = pd.DataFrame(dict(treatment=treatments, game=games,
                            bdistance=distb, sdistance=dists)).round(4)
    return out


def pick_winners(df: pd.DataFrame, ape_col: str, relevant_rounds: Iterable[int],
                 deal_spec, non_linear_models: set[str]) -> pd.DataFrame:
    q = df.query("round in @relevant_rounds")
    if deal_spec == "pos":
        q = q.query("n_unique_deals_round > 0")
    else:
        q = q.query("n_unique_deals_round in @deal_spec")
    med = (q.groupby(["treatment", "game", "model"], observed=True)[ape_col]
           .median().reset_index())
    med["rank"] = med.groupby(["treatment", "game"])[ape_col].rank(
        ascending=True, method="first")
    winners = med.query("rank == 1").copy()
    winners["model_type"] = winners["model"].map(
        lambda x: "Non-Linear" if x in non_linear_models else "Linear")
    return winners[["treatment", "game", "model", "model_type"]]


def _draw_voronoi(ax, points: np.ndarray, colors: list[str]) -> None:
    vor = Voronoi(points)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    for i, region in enumerate(regions):
        if not region:
            continue
        pts = [vertices[j] for j in region if np.linalg.norm(vertices[j]) < 100]
        if len(pts) < 3:
            continue
        poly = plt.Polygon(pts, facecolor=colors[i], edgecolor="white",
                           linewidth=1.0, alpha=0.5, zorder=1)
        ax.add_patch(poly)


def _render_composite(winners_xy: pd.DataFrame, out_pdf: Path,
                      title_round: str, title_deal: str) -> None:
    pts = winners_xy[["bdistance", "sdistance"]].to_numpy()
    types = winners_xy["model_type"].tolist()
    face = [TYPE_FACE[t] for t in types]
    point = [TYPE_POINT[t] for t in types]

    xmin = pts[:, 0].min() * 1.1
    xmax = pts[:, 0].max() * 1.1
    ymin = pts[:, 1].min() * 1.1
    ymax = pts[:, 1].max() * 1.1

    fig, (ax_main, ax_zoom) = plt.subplots(
        1, 2, figsize=(10, 5.4), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    # Main (wide) panel
    _draw_voronoi(ax_main, pts, face)
    ax_main.scatter(pts[:, 0], pts[:, 1], c=point, s=10, zorder=3)
    ax_main.set_xlim(xmin, xmax)
    ax_main.set_ylim(ymin, ymax)
    ax_main.axhline(0, color="black", linewidth=0.6, zorder=2)
    ax_main.axvline(0, color="black", linewidth=0.6, zorder=2)
    ax_main.set_xlabel("Buyer Valuation Distance")
    ax_main.set_ylabel("Seller Valuation Distance")
    ax_main.add_patch(Rectangle((-0.5, -0.5), 1.0, 1.0, fill=False,
                                edgecolor="gray", linewidth=0.8, zorder=4))

    # Zoom panel
    _draw_voronoi(ax_zoom, pts, face)
    ax_zoom.scatter(pts[:, 0], pts[:, 1], c=point, s=10, zorder=3)
    ax_zoom.set_xlim(-0.5, 0.5)
    ax_zoom.set_ylim(-0.5, 0.5)
    ax_zoom.set_xticks([-0.5, 0, 0.5])
    ax_zoom.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax_zoom.axhline(0, color="black", linewidth=0.6, zorder=2)
    ax_zoom.axvline(0, color="black", linewidth=0.6, zorder=2)

    # Dashed "zoom-lens" connectors
    for xy_main, xy_zoom in [((0.5, 0.5), (-0.5, 0.5)),
                              ((0.5, -0.5), (-0.5, -0.5))]:
        con = ConnectionPatch(xyA=xy_main, xyB=xy_zoom, coordsA="data",
                              coordsB="data", axesA=ax_main, axesB=ax_zoom,
                              linestyle=DASH_STYLE, color=DASH_COLOR,
                              linewidth=1.0, zorder=5)
        fig.add_artist(con)

    for ax in (ax_main, ax_zoom):
        ax.tick_params(labelsize=10)

    # Title sits to the right of the axis title strip, above ax_main
    ax_main.set_title(f"Round: {title_round}, # Deals: {title_deal}",
                      fontsize=12, loc="right", pad=14)

    # Figure-level legend above both panels, with larger markers
    nl = plt.Line2D([0], [0], marker="o", color="none",
                    markerfacecolor=TYPE_POINT["Non-Linear"],
                    markeredgecolor=TYPE_POINT["Non-Linear"],
                    markersize=9, label="Non-Linear")
    lin = plt.Line2D([0], [0], marker="o", color="none",
                     markerfacecolor=TYPE_POINT["Linear"],
                     markeredgecolor=TYPE_POINT["Linear"],
                     markersize=9, label="Linear")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(handles=[nl, lin], loc="upper left",
               bbox_to_anchor=(0.02, 0.99), ncol=2, frameon=False,
               handletextpad=0.4, columnspacing=1.6, fontsize=12)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def load_results(task: str) -> pd.DataFrame:
    frames = []
    for model, rel in TASKS[task]["result_files"].items():
        df = pd.read_feather(CODE_ROOT / rel)
        df = df.copy()
        df["model"] = model
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_task(task: str, original_df: pd.DataFrame) -> None:
    cfg = TASKS[task]
    all_df = load_results(task)

    distances_by_rounds: dict[tuple[int, ...], pd.DataFrame] = {}
    for stem, tr, td, round_spec, deal_spec, ce_suffix, ae_suffix in QUADRANTS:
        rounds = (tuple(sorted(r for r in all_df["round"].unique() if r > 1))
                  if round_spec == "round>1" else tuple(round_spec))
        if rounds not in distances_by_rounds:
            distances_by_rounds[rounds] = compute_distances(original_df, list(rounds))
        dists = distances_by_rounds[rounds]

        winners = pick_winners(all_df, cfg["ape_col"], list(rounds),
                               deal_spec, cfg["non_linear_models"])
        merged = winners.merge(dists, on=["treatment", "game"])
        assert merged[["bdistance", "sdistance"]].duplicated().sum() == 0

        suffix = ae_suffix if task == "ae" else ce_suffix
        out_pdf = cfg["out_dir"] / f"{stem}{suffix}.pdf"
        _render_composite(merged, out_pdf, tr, td)
        print(f"[{task}] wrote {out_pdf}  ({len(merged)} games)")


def main() -> None:
    original_df = pd.read_feather(CODE_ROOT / "data/preprocessed/original_df.ft")
    for task in ("ce", "ae"):
        build_task(task, original_df)


if __name__ == "__main__":
    main()
