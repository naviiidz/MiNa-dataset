import argparse
import json
import os
from collections import defaultdict

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from config import (COCO_MAX_DETS, FIT_MAX_PX, FULL_IMAGES, IOU_THRESHOLDS, MIN_VISIBLE_FRACTION, MODELS, NM_BINS, PATCHES,
                    PIXEL_SIZE_CSV, PX_BINS, RESULTS_DIR, SEEDS, prediction_path)

REC_THRS = np.linspace(0, 1, 101)
EVALS = [("rtdetr", "box"), ("maskrcnn", "box"), ("maskrcnn", "mask")]
EVAL_NAMES = {("rtdetr", "box"): "RT-DETR-L (box)", ("maskrcnn", "box"): "Mask R-CNN R101 (box)",
              ("maskrcnn", "mask"): "Mask R-CNN R101 (mask)"}
COLORS = {("rtdetr", "box"): "#e07b39", ("maskrcnn", "box"): "#7fa7d9", ("maskrcnn", "mask"): "#2f6db5"}


def eq_diameter(area):
    return np.sqrt(4 * np.asarray(area, dtype=float) / np.pi)


def polygons(seg):
    seg = json.loads(seg) if isinstance(seg, str) else seg
    return [s for s in seg if len(s) >= 6]


def full_image_areas():
    areas = {}
    for polymer in os.listdir(FULL_IMAGES):
        path = os.path.join(FULL_IMAGES, polymer, f"{polymer.lower()}.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            coco = json.load(fh)
        names = {i["id"]: i["file_name"] for i in coco["images"]}
        for a in coco["annotations"]:
            areas[(names[a["image_id"]], a["id"])] = a["area"]
    return areas


def load_gt(split):
    with open(os.path.join(PATCHES, split, "annotations.json")) as fh:
        coco = json.load(fh)
    images = {i["id"]: i for i in coco["images"]}
    full = full_image_areas()
    nm_per_px = pd.read_csv(PIXEL_SIZE_CSV).groupby("source_image")["um_per_px"].first() * 1000
    rows = []
    for a in coco["annotations"]:
        im = images[a["image_id"]]
        polys = polygons(a["segmentation"])
        if not polys:
            continue
        rle = mask_utils.merge(mask_utils.frPyObjects(polys, im["height"], im["width"]))
        full_area = full.get((im["source_image"], a["source_annotation_id"]), a["area"])
        rows.append({"image_id": a["image_id"], "ann_id": a["id"], "source_image": im["source_image"],
                     "source_ann": a["source_annotation_id"], "area": float(mask_utils.area(rle)), "bbox": a["bbox"],
                     "rle": rle, "visible": a["area"] / max(full_area, 1e-9), "nm_per_px": nm_per_px[im["source_image"]]})
    gt = pd.DataFrame(rows)
    gt["d_px"] = eq_diameter(gt["area"])
    gt["circularity"] = [circularity(mask_utils.decode(r).astype(bool)) for r in gt["rle"]]
    gt["d_nm"] = gt["d_px"] * gt["nm_per_px"]
    keep = gt.sort_values(["area", "ann_id"], ascending=[False, True]).drop_duplicates(["source_image", "source_ann"]).index
    gt["truncated"] = gt["visible"] < MIN_VISIBLE_FRACTION
    gt["duplicate"] = ~gt.index.isin(keep)
    gt["ignore"] = gt["truncated"] | gt["duplicate"]
    nm = {i["id"]: nm_per_px[i["source_image"]] for i in coco["images"]}
    return gt, sorted(images), nm


def load_dets(model, seed, split):
    with open(prediction_path(model, seed, split)) as fh:
        dets = json.load(fh)
    by_image = defaultdict(list)
    for d in dets:
        seg = d.get("segmentation")
        if seg is not None and isinstance(seg["counts"], list):
            d["segmentation"] = mask_utils.frPyObjects(seg, *seg["size"])
        by_image[d["image_id"]].append(d)
    return {k: sorted(v, key=lambda d: -d["score"]) for k, v in by_image.items()}


def prepare(gt, image_ids, dets, kind, nm):
    data = {}
    for image_id in image_ids:
        g = gt[gt["image_id"] == image_id]
        d = dets.get(image_id, [])
        if kind == "mask":
            dt_obj, gt_obj = [x["segmentation"] for x in d], list(g["rle"])
            dt_area = np.array([mask_utils.area(x["segmentation"]) for x in d], dtype=float)
        else:
            dt_obj, gt_obj = [x["bbox"] for x in d], [list(b) for b in g["bbox"]]
            dt_area = np.array([np.pi / 4 * x["bbox"][2] * x["bbox"][3] for x in d], dtype=float)
        ious = np.asarray(mask_utils.iou(dt_obj, gt_obj, [0] * len(gt_obj)), dtype=float).reshape(len(d), len(gt_obj)) \
            if len(d) and len(gt_obj) else np.zeros((len(d), len(gt_obj)))
        data[image_id] = {"scores": np.array([x["score"] for x in d], dtype=float), "ious": ious,
                          "dt_d_px": eq_diameter(dt_area), "dt_d_nm": eq_diameter(dt_area) * nm[image_id],
                          "dt_boxes": np.array([x["bbox"] for x in d], dtype=float).reshape(-1, 4),
                          "dt_masks": [x.get("segmentation") for x in d], "gt_index": g.index.to_numpy()}
    return data


def match(ious, gt_ig, t):
    n_dt, n_gt = ious.shape
    gt_taken = np.zeros(n_gt, dtype=bool)
    dt_gt = np.full(n_dt, -1)
    for k in range(n_dt):
        ok = (~gt_taken) & (ious[k] >= t)
        if not ok.any():
            continue
        cand = ok & ~gt_ig
        cand = cand if cand.any() else ok
        g = np.flatnonzero(cand)[np.argmax(ious[k, cand])]
        gt_taken[g] = True
        dt_gt[k] = g
    return dt_gt


def matched_ignore(dt_gt, gt_ig):
    return (dt_gt >= 0) & (gt_ig[np.maximum(dt_gt, 0)] if len(gt_ig) else False)


def coco_ap(data, gt, in_bin_gt, det_in_bin, thresholds=IOU_THRESHOLDS):
    aps = []
    for t in thresholds:
        scores, tps, igs, npig = [], [], [], 0
        for image_id, d in data.items():
            idx = d["gt_index"]
            gt_ig = gt.loc[idx, "ignore"].to_numpy() | ~in_bin_gt[idx]
            npig += int((~gt_ig).sum())
            n = min(len(d["scores"]), COCO_MAX_DETS)
            if n == 0:
                continue
            dt_gt = match(d["ious"][:n], gt_ig, t)
            matched = dt_gt >= 0
            ig = matched_ignore(dt_gt, gt_ig) | (~matched & ~det_in_bin(d)[:n])
            scores.append(d["scores"][:n]), tps.append(matched), igs.append(ig)
        if npig == 0:
            return np.nan, np.nan
        s, tp, ig = np.concatenate(scores), np.concatenate(tps), np.concatenate(igs)
        order = np.argsort(-s, kind="mergesort")
        tp, ig = tp[order], ig[order]
        tp_c = np.cumsum(tp & ~ig)
        fp_c = np.cumsum(~tp & ~ig)
        rc = tp_c / npig
        pr = tp_c / np.maximum(tp_c + fp_c, np.spacing(1))
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        inds = np.searchsorted(rc, REC_THRS, side="left")
        q = np.array([pr[i] if i < len(pr) else 0.0 for i in inds])
        aps.append(q.mean())
    return 100 * aps[0], 100 * float(np.mean(aps))


def operating_matches(data, gt):
    out = {}
    for image_id, d in data.items():
        gt_ig = gt.loc[d["gt_index"], "ignore"].to_numpy()
        out[image_id] = match(d["ious"], gt_ig, 0.5) if len(d["scores"]) else np.zeros(0, dtype=int)
    return out


def choose_threshold(data, gt, matches):
    s_all, tp_all, ig_all = [], [], []
    npig = int((~gt["ignore"]).sum())
    for image_id, d in data.items():
        dt_gt = matches[image_id]
        gt_ig = gt.loc[d["gt_index"], "ignore"].to_numpy()
        matched = dt_gt >= 0
        ig = matched_ignore(dt_gt, gt_ig)
        s_all.append(d["scores"]), tp_all.append(matched & ~ig), ig_all.append(ig)
    s, tp, ig = np.concatenate(s_all), np.concatenate(tp_all), np.concatenate(ig_all)
    best = (0, 0.5)
    for thr in np.arange(0.01, 1.0, 0.01):
        keep = (s >= thr) & ~ig
        n_tp = int((tp & keep).sum())
        p, r = n_tp / max(keep.sum(), 1), n_tp / max(npig, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best[0]:
            best = (f1, float(thr))
    return best[1]


def circularity(m):
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.nan
    c = max(contours, key=cv2.contourArea)
    p = cv2.arcLength(c, True)
    return 4 * np.pi * m.sum() / p ** 2 if p > 0 else np.nan


def pair_quality(gt_rle, dt_rle):
    g, p = mask_utils.decode(gt_rle).astype(bool), mask_utils.decode(dt_rle).astype(bool)
    inter = (g & p).sum()
    cg, cp = circularity(g), circularity(p)
    return {"dice": 2 * inter / max(g.sum() + p.sum(), 1),
            "area_error": abs(p.sum() - g.sum()) / max(g.sum(), 1),
            "circularity_error": abs(cp - cg)}


def per_particle(data, gt, matches, thr, kind):
    rows = []
    for image_id, d in data.items():
        dt_gt, keep = matches[image_id], d["scores"] >= thr
        centres = np.c_[d["dt_boxes"][:, 0] + d["dt_boxes"][:, 2] / 2, d["dt_boxes"][:, 1] + d["dt_boxes"][:, 3] / 2][keep]
        for j, gi in enumerate(d["gt_index"]):
            if gt.at[gi, "ignore"]:
                continue
            hit = np.flatnonzero((dt_gt == j) & keep)
            x, y, w, h = gt.at[gi, "bbox"]
            located = bool(((centres[:, 0] >= x - 1) & (centres[:, 0] <= x + w + 1) &
                            (centres[:, 1] >= y - 1) & (centres[:, 1] <= y + h + 1)).any()) if len(centres) else False
            best_iou = float(d["ious"][keep, j].max()) if keep.any() and d["ious"].shape[1] else 0.0
            row = {"gt": gi, "detected": len(hit) > 0, "located": located, "best_iou": best_iou}
            if kind == "mask" and len(hit):
                row.update(pair_quality(gt.at[gi, "rle"], d["dt_masks"][hit[0]]))
            rows.append(row)
    return pd.DataFrame(rows).set_index("gt")


def per_detection(data, gt, matches, thr):
    rows = []
    for image_id, d in data.items():
        dt_gt, keep = matches[image_id], d["scores"] >= thr
        gt_ig = gt.loc[d["gt_index"], "ignore"].to_numpy()
        for k in np.flatnonzero(keep):
            g = dt_gt[k]
            if g >= 0 and gt_ig[g]:
                continue
            rows.append({"tp": g >= 0, "d_px": d["dt_d_px"][k], "d_nm": d["dt_d_nm"][k]})
    return pd.DataFrame(rows)


def bin_labels(edges, unit):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(f"> {lo:g} {unit}" if hi >= 1e8 else (f"< {hi:g} {unit}" if lo == 0 else f"{lo:g}–{hi:g} {unit}"))
    return out


def size_fit(x, y, max_px=FIT_MAX_PX):
    from scipy.optimize import minimize
    sel = x <= max_px
    lx, y = np.log2(x[sel]), y[sel].astype(float)

    def nll(theta):
        a, b, logit_top = theta
        top = 1 / (1 + np.exp(-logit_top))
        p = np.clip(top / (1 + np.exp(-(a + b * lx))), 1e-9, 1 - 1e-9)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).sum()

    a, b, logit_top = minimize(nll, [-6.0, 3.0, 2.0], method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-8}).x
    top = 1 / (1 + np.exp(-logit_top))
    at = lambda q: float(2 ** ((np.log(q / (top - q)) - a) / b)) if q < top else np.nan
    return {"plateau": 100 * top, "D50": at(0.5), "D90_of_plateau": at(0.9 * top)}


def evaluate_one(model, seed, kind, gt, image_ids, nm, val):
    dets_val = load_dets(model, seed, "val")
    gt_v, ids_v, nm_v = val
    data_v = prepare(gt_v, ids_v, dets_val, kind, nm_v)
    thr = choose_threshold(data_v, gt_v, operating_matches(data_v, gt_v))
    data = prepare(gt, image_ids, load_dets(model, seed, "test"), kind, nm)
    matches = operating_matches(data, gt)
    parts = per_particle(data, gt, matches, thr, kind).join(gt[["d_px", "d_nm", "circularity"]])
    dets = per_detection(data, gt, matches, thr)
    rows = []
    for axis, edges, unit in (("px", PX_BINS, "px"), ("nm", NM_BINS, "nm")):
        col = f"d_{axis}"
        labels = ["all"] + bin_labels(edges, unit)
        for label, lo, hi in [("all", 0, 1e12)] + list(zip(labels[1:], edges[:-1], edges[1:])):
            in_gt = ((gt[col] >= lo) & (gt[col] < hi)).to_numpy()
            ap50, ap = coco_ap(data, gt, in_gt, lambda d, lo=lo, hi=hi, col=col: (d[f"dt_{col}"] >= lo) & (d[f"dt_{col}"] < hi))
            p = parts[(parts[col] >= lo) & (parts[col] < hi)]
            dd = dets[(dets[col] >= lo) & (dets[col] < hi)] if len(dets) else dets
            row = {"model": model, "kind": kind, "seed": seed, "threshold": thr, "axis": axis, "bin": label, "lo": lo,
                   "n_particles": len(p), "recall": 100 * p["detected"].mean(), "centre_recall": 100 * p["located"].mean(),
                   "precision": 100 * dd["tp"].mean() if len(dd) else np.nan, "n_detections": len(dd), "AP50": ap50, "AP": ap,
                   "median_best_iou": p["best_iou"].median(), "iou_ge_0.5": 100 * (p["best_iou"] >= 0.5).mean(),
                   "iou_ge_0.75": 100 * (p["best_iou"] >= 0.75).mean()}
            if kind == "mask":
                row.update({k: p[k].median() for k in ("dice", "area_error", "circularity_error")})
                row["n_matched"] = int(p["detected"].sum())
                row["gt_circularity_over_1"] = 100 * (p["circularity"] > 1.0).mean()
            rows.append(row)
    fits = {"model": model, "kind": kind, "seed": seed,
            **size_fit(parts["d_px"].to_numpy(), parts["detected"].to_numpy())}
    return rows, fits, parts.assign(model=model, kind=kind, seed=seed)


def check_against_pycocotools(gt, image_ids):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with open(os.path.join(PATCHES, "test", "annotations.json")) as fh:
        raw = json.load(fh)
    raw["annotations"] = [a for a in raw["annotations"] if polygons(a["segmentation"])]
    for a in raw["annotations"]:
        a["category_id"], a["segmentation"] = 1, polygons(a["segmentation"])
    raw["categories"] = [{"id": 1, "name": "particle"}]
    coco = COCO()
    coco.dataset = raw
    coco.createIndex()
    for model, kind in EVALS:
        dets = json.load(open(prediction_path(model, 0, "test")))
        if kind == "box":
            dets = [{k: v for k, v in d.items() if k != "segmentation"} for d in dets]
        ev = COCOeval(coco, coco.loadRes(dets), "segm" if kind == "mask" else "bbox")
        ev.evaluate(), ev.accumulate()
        ev.summarize()
        ref = ev.stats
        ref_ap = 100 * float(np.mean([ev.eval["precision"][i, :, 0, 0, 2][ev.eval["precision"][i, :, 0, 0, 2] > -1].mean()
                                      for i in range(10)]))
        g0 = gt.assign(ignore=False)
        data = prepare(g0, image_ids, load_dets(model, 0, "test"), kind, {i: 1.0 for i in image_ids})
        ap50, ap = coco_ap(data, g0, np.ones(len(g0), bool), lambda d: np.ones(len(d["scores"]), bool))
        print(f"CHECK {EVAL_NAMES[(model, kind)]} seed 0: pycocotools AP50 {100 * ref[1]:.2f} AP {ref_ap:.2f} | "
              f"this evaluator AP50 {ap50:.2f} AP {ap:.2f}")


def summarize(results, fits):
    metrics = ["n_particles", "recall", "centre_recall", "precision", "AP50", "AP", "median_best_iou", "iou_ge_0.5",
               "iou_ge_0.75", "n_matched", "dice", "area_error", "circularity_error", "gt_circularity_over_1", "threshold"]
    g = results.groupby(["model", "kind", "axis", "bin", "lo"], sort=False)[metrics]
    mean, sd = g.mean(), g.std(ddof=1)
    table = mean.round(2).astype(str) + " ± " + sd.round(2).astype(str)
    table["n_particles"] = mean["n_particles"].round(0).astype(int).astype(str)
    fit_cols = ["plateau", "D50", "D90_of_plateau"]
    fg = fits.groupby(["model", "kind"], sort=False)[fit_cols]
    fit_table = fg.mean().round(1).astype(str) + " ± " + fg.std(ddof=1).round(1).astype(str)
    return mean.reset_index(), sd.reset_index(), table.reset_index(), fit_table.reset_index()


def disc_limits(diameters):
    rng = np.random.default_rng(0)
    rows = []
    for d in diameters:
        circ = []
        for _ in range(200):
            n = int(np.ceil(d)) + 6
            cy, cx = rng.uniform(n / 2 - 0.5, n / 2 + 0.5, 2)
            yy, xx = np.mgrid[:n, :n]
            m = (yy - cy) ** 2 + (xx - cx) ** 2 <= (d / 2) ** 2
            if m.sum():
                circ.append(abs(circularity(m) - 1.0))
        rows.append({"d": d, "iou_1px": d ** 2 / (d + 2) ** 2, "iou_half_px": d ** 2 / (d + 1) ** 2,
                     "disc_circularity_error": float(np.median(circ))})
    return pd.DataFrame(rows)


def bin_x(results, axis):
    b = results[(results["axis"] == axis) & (results["bin"] != "all")].drop_duplicates("bin")
    edges = PX_BINS if axis == "px" else NM_BINS
    centres = {}
    for label, lo, hi in zip(bin_labels(edges, axis), edges[:-1], edges[1:]):
        lo_ = lo if lo > 0 else (edges[1] / 2)
        centres[label] = np.sqrt(lo_ * (hi if hi < 1e8 else lo * 2))
    return b["bin"].tolist(), centres


def plot_metric(ax, mean, sd, axis, metric, evals, centres, labels, style="-o", alpha=1.0, suffix=""):
    for key in evals:
        m = mean[(mean["model"] == key[0]) & (mean["kind"] == key[1]) & (mean["axis"] == axis) & (mean["bin"] != "all")]
        s = sd[(sd["model"] == key[0]) & (sd["kind"] == key[1]) & (sd["axis"] == axis) & (sd["bin"] != "all")]
        x = [centres[b] for b in m["bin"]]
        ax.errorbar(x, m[metric], yerr=s[metric], fmt=style, ms=4, capsize=2, color=COLORS[key], alpha=alpha,
                    label=EVAL_NAMES[key] + suffix)


def style_axis(ax, centres, labels, xlabel):
    ax.set_xscale("log", base=2)
    ax.set_xticks([centres[b] for b in labels])
    ax.set_xticklabels([b.replace(" px", "").replace(" nm", "") for b in labels], fontsize=7, rotation=30)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3)


def plots(mean, sd, fits, limits, parts, gt):
    labels, centres = bin_x(mean, "px")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    plot_metric(ax, mean, sd, "px", "recall", EVALS, centres, labels)
    plot_metric(ax, mean, sd, "px", "centre_recall", [("rtdetr", "box"), ("maskrcnn", "mask")], centres, labels,
                style="--s", alpha=0.6, suffix=", centre match")
    fm = fits.groupby(["model", "kind"])[["D50", "D90_of_plateau"]].mean()
    for key in EVALS:
        for col, ls in (("D50", ":"), ("D90_of_plateau", "-.")):
            ax.axvline(fm.loc[key, col], color=COLORS[key], ls=ls, lw=0.9)
    ax.set_ylabel("Recall at IoU 0.5 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Detection rate (dotted: D50, dash-dot: 90% of plateau)", fontsize=9)
    style_axis(ax, centres, labels, "Particle equivalent diameter (px)")
    ax.legend(fontsize=6.5, loc="lower right")
    ax = axes[1]
    plot_metric(ax, mean, sd, "px", "median_best_iou", EVALS, centres, labels)
    ax.plot(limits["d"], limits["iou_1px"], "k--", lw=1, label="reference: disc with 1 px boundary error")
    ax.plot(limits["d"], limits["iou_half_px"], "k:", lw=1, label="reference: disc with 0.5 px boundary error")
    ax.set_ylabel("Median IoU of best prediction")
    ax.set_ylim(0, 1)
    ax.set_title("Localisation / outline quality (0 = not found)", fontsize=9)
    style_axis(ax, centres, labels, "Particle equivalent diameter (px)")
    ax.legend(fontsize=6.5, loc="lower right")
    ax = axes[2]
    plot_metric(ax, mean, sd, "px", "AP50", EVALS, centres, labels)
    plot_metric(ax, mean, sd, "px", "AP", EVALS, centres, labels, style="--s", alpha=0.6, suffix=", AP50-95")
    ax.set_ylabel("AP (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Average precision per size bin", fontsize=9)
    style_axis(ax, centres, labels, "Particle equivalent diameter (px)")
    ax.legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Size-resolved performance on the 260 test patches (mean ± SD over 3 training seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "size_performance_px.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    seg = [("maskrcnn", "mask")]
    for ax, metric, ylabel in zip(axes, ("dice", "area_error", "circularity_error"),
                                  ("Dice", "Relative area error", "Absolute circularity error (4πA/P², uncapped)")):
        plot_metric(ax, mean, sd, "px", metric, seg, centres, labels)
        if metric == "circularity_error":
            ax.plot(limits["d"], limits["disc_circularity_error"], "k--", lw=1, label="digitised perfect disc vs true value 1")
        ax.set_ylabel(ylabel)
        style_axis(ax, centres, labels, "Particle equivalent diameter (px)")
        ax.legend(fontsize=7)
    fig.suptitle("Mask R-CNN R101: morphology of matched particles (IoU ≥ 0.5), median per size bin", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "size_morphology_px.png"), dpi=200)
    plt.close(fig)

    labels_nm, centres_nm = bin_x(mean, "nm")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    plot_metric(ax, mean, sd, "nm", "recall", EVALS, centres_nm, labels_nm)
    ax.set_ylabel("Recall at IoU 0.5 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Detection rate vs physical size", fontsize=9)
    style_axis(ax, centres_nm, labels_nm, "Particle equivalent diameter (nm)")
    ax.legend(fontsize=6.5, loc="lower right")
    px_edges, nm_edges = [2, 4, 8, 16, 32, 1e9], [0, 250, 500, 1000, 2000, 1e9]
    for ax, key in zip(axes[1:], [("rtdetr", "box"), ("maskrcnn", "mask")]):
        p = parts[(parts["model"] == key[0]) & (parts["kind"] == key[1])]
        p = p.assign(px=pd.cut(p["d_px"], px_edges, right=False), nm=pd.cut(p["d_nm"], nm_edges, right=False))
        rec = p.groupby(["nm", "px"], observed=False)["detected"].mean().unstack() * 100
        cnt = p.groupby(["nm", "px"], observed=False)["detected"].size().unstack() / len(SEEDS)
        rec = rec.where(cnt >= 15)
        im = ax.imshow(rec.to_numpy(), cmap="viridis", vmin=0, vmax=100, aspect="auto", origin="lower")
        for i in range(rec.shape[0]):
            for j in range(rec.shape[1]):
                if not np.isnan(rec.iat[i, j]):
                    ax.text(j, i, f"{rec.iat[i, j]:.0f}\nn={cnt.iat[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if rec.iat[i, j] < 60 else "black")
        ax.set_xticks(range(rec.shape[1]))
        ax.set_xticklabels(bin_labels(px_edges, "")[:-1] + ["≥ 32"], fontsize=7)
        ax.set_yticks(range(rec.shape[0]))
        ax.set_yticklabels(bin_labels(nm_edges, ""), fontsize=7)
        ax.set_xlabel("Equivalent diameter (px)")
        ax.set_ylabel("Equivalent diameter (nm)")
        ax.set_title(f"{EVAL_NAMES[key]}: recall (%) by pixels × nanometres", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Pixels or physical size? Recall at IoU 0.5, test patches, 3 seeds pooled (cells with < 15 particles blank)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "size_performance_nm.png"), dpi=200)
    plt.close(fig)


def write_summary(table, fit_table, gt, out):
    n_all, n_ign = len(gt), int(gt["ignore"].sum())
    lines = ["# Size-resolved detection and segmentation (test patches)", "",
             f"Test patches: 260. Annotated instances: {n_all}; scored: {n_all - n_ign}; ignored: {n_ign} "
             f"({int(gt['truncated'].sum())} cut by the patch border, {int((gt['duplicate'] & ~gt['truncated']).sum())} duplicate copies in overlaps).",
             "Values: mean ± SD over 3 training seeds. Recall, precision and IoU use the confidence threshold with the best F1 on the validation patches.", ""]
    show = {"box": ["n_particles", "recall", "centre_recall", "precision", "AP50", "AP", "median_best_iou"],
            "mask": ["n_particles", "recall", "centre_recall", "precision", "AP50", "AP", "median_best_iou", "n_matched", "dice",
                     "area_error", "circularity_error", "gt_circularity_over_1"]}
    for (model, kind) in EVALS:
        for axis in ("px", "nm"):
            t = table[(table["model"] == model) & (table["kind"] == kind) & (table["axis"] == axis)]
            cols = show[kind]
            lines += [f"## {EVAL_NAMES[(model, kind)]}, by equivalent diameter in {axis}", "",
                      "| Size | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
            lines += ["| " + r["bin"] + " | " + " | ".join(r[c] for c in cols) + " |" for _, r in t.iterrows()]
            lines.append("")
    lines += [f"## Detection threshold in pixels (logistic fit with a plateau, particles ≤ {FIT_MAX_PX} px)", "",
              "| Model | Plateau recall (%) | D50 (px) | Size at 90% of plateau (px) |", "|---|---|---|---|"]
    for _, r in fit_table.iterrows():
        lines.append(f"| {EVAL_NAMES[(r['model'], r['kind'])]} | {r['plateau']} | {r['D50']} | {r['D90_of_plateau']} |")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Size-resolved evaluation of the best detector and segmenter on the test patches.")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--check", action="store_true", help="Compare this evaluator with pycocotools on all particles (seed 0)")
    args = p.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gt, image_ids, nm = load_gt("test")
    if args.check:
        check_against_pycocotools(gt, image_ids)
        return
    val = load_gt("val")
    rows, fits, parts = [], [], []
    for model, kind in EVALS:
        for seed in args.seeds:
            r, f, pp = evaluate_one(model, seed, kind, gt, image_ids, nm, val)
            rows += r
            fits.append(f)
            parts.append(pp)
            print(f"{EVAL_NAMES[(model, kind)]} seed {seed}: threshold {r[0]['threshold']:.2f}, recall {r[0]['recall']:.1f}, "
                  f"AP50 {r[0]['AP50']:.1f}, plateau {f['plateau']:.1f}%, D50 {f['D50']:.1f} px, "
                  f"90% of plateau {f['D90_of_plateau']:.1f} px", flush=True)
    results, fits, parts = pd.DataFrame(rows), pd.DataFrame(fits), pd.concat(parts)
    results.to_csv(os.path.join(RESULTS_DIR, "size_metrics_per_seed.csv"), index=False)
    fits.to_csv(os.path.join(RESULTS_DIR, "size_fits_per_seed.csv"), index=False)
    parts.reset_index().to_csv(os.path.join(RESULTS_DIR, "particles_per_seed.csv"), index=False)
    mean, sd, table, fit_table = summarize(results, fits)
    table.to_csv(os.path.join(RESULTS_DIR, "size_metrics.csv"), index=False)
    limits = disc_limits(np.geomspace(2, 128, 40))
    limits.to_csv(os.path.join(RESULTS_DIR, "pixelisation_limits.csv"), index=False)
    gt.drop(columns=["rle"]).to_csv(os.path.join(RESULTS_DIR, "test_particles.csv"), index=False)
    plots(mean, sd, fits, limits, parts, gt)
    print(write_summary(table, fit_table, gt, os.path.join(RESULTS_DIR, "summary.md")))


if __name__ == "__main__":
    main()
