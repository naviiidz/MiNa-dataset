"""Figures and Table S3 for RT-DETR-L (class-agnostic) + SAM 2.1, one file per panel.

Reads outputs/detector_sam_{box,point}_seed{0,1,2}/<dataset>/ written by run_revision.sh.
Qualitative examples use seed 0; each example is the patch (or image) with the median F1 of its
group, so examples are typical rather than best cases.

    python make_revision_figures.py --out ~/sam_figures
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluation

SEEDS = (0, 1, 2)
PROMPTS = ('box', 'point')
EXTERNAL = ('fragment_1', 'fragment_2', 'dataset2_low_mag', 'dataset2_high_mag')
EXTERNAL_LABELS = {'fragment_1': 'Shi et al. (fragment_1)', 'fragment_2': 'Shi et al. (fragment_2)',
                   'dataset2_low_mag': 'Rettenberger (Low-mag)', 'dataset2_high_mag': 'Rettenberger (High-mag)'}
POLYMERS = ('pp', 'pe', 'ps', 'pet')
COLORS = {'box': '#2a78d6', 'point': '#eb6834'}
GT_COLOR = (0, 200, 0)
PRED_COLOR = (214, 120, 42)
MAX_SIDE = 2000
METRICS = [('precision', 'Precision'), ('recall', 'Recall'), ('f1_score', 'F1')]

plt.rcParams.update({'font.size': 9, 'font.family': 'sans-serif', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'savefig.dpi': 600})


def run_dir(prompt, seed, dataset):
    return evaluation.OUTPUT_ROOT / f'detector_sam_{prompt}_seed{seed}' / dataset


def summaries(prompt, dataset):
    rows = []
    for seed in SEEDS:
        path = run_dir(prompt, seed, dataset) / 'summary.csv'
        if path.exists():
            row = pd.read_csv(path).iloc[0].to_dict()
            rows.append({'seed': seed, **row})
    return pd.DataFrame(rows)


def table_s3(out):
    rows = []
    for dataset, prompts in [('MiNa_dataset', PROMPTS)] + [(d, ('box',)) for d in EXTERNAL]:
        for prompt in prompts:
            df = summaries(prompt, dataset)
            if df.empty:
                continue
            df.assign(prompt=prompt, dataset=dataset).to_csv(out / f'per_seed_{dataset}_{prompt}.csv', index=False)
            row = {'Dataset': EXTERNAL_LABELS.get(dataset, 'MiNa test'), 'Prompt': prompt, 'Images': int(df['images'].iloc[0]),
                   'Seeds': len(df), 'Threshold': '/'.join(f'{c:g}' for c in df['confidence'])}
            for col, name, scale, digits in [('precision', 'Precision (%)', 100, 1), ('recall', 'Recall (%)', 100, 1),
                                             ('f1_score', 'F1 (%)', 100, 1), ('matched_iou', 'Matched IoU', 1, 3),
                                             ('matched_dsc', 'Matched Dice', 1, 3), ('area_mape', 'Area MAPE (%)', 1, 1),
                                             ('mean_seconds_per_image', 'Time (s/image)', 1, 2)]:
                v = df[col] * scale
                row[name] = f'{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}' if len(v) > 1 else f'{v.mean():.{digits}f}'
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'table_S3.csv', index=False)
    table.to_csv(out / 'table_S3.tsv', index=False, sep='\t')
    print(table.to_string(index=False))


def bar_chart(groups, series, values, errors, colors, ylabel, path, ylim=None, legend=True, figsize=(3.5, 2.6)):
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(groups))
    width = 0.8 / len(series)
    for k, s in enumerate(series):
        pos = x + (k - (len(series) - 1) / 2) * width
        ax.bar(pos, values[k], width * 0.92, yerr=errors[k], color=colors[k], label=s,
               error_kw={'elinewidth': 0.8, 'capsize': 2, 'ecolor': '#0b0b0b'})
        top = ylim[1] if ylim else max(map(max, values))
        for p, v, e in zip(pos, values[k], errors[k]):
            ax.text(p, v + (e if np.isfinite(e) else 0) + top * 0.01, f'{v:.1f}' if v >= 1 else f'{v:.2f}',
                    ha='center', va='bottom', fontsize=6.5, color='#52514e')
    ax.set_xticks(x, groups)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color='#e6e5e1', linewidth=0.6)
    ax.set_axisbelow(True)
    if legend and len(series) > 1:
        ax.legend(frameon=False, ncol=len(series), loc='upper center', bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(path.with_suffix(f'.{ext}'), bbox_inches='tight')
    plt.close(fig)


def charts(out):
    mina = {p: summaries(p, 'MiNa_dataset') for p in PROMPTS}
    labels = {'box': 'Box prompt', 'point': 'Point prompt'}
    vals = [[mina[p][m].mean() * 100 for m, _ in METRICS] for p in PROMPTS]
    errs = [[mina[p][m].std(ddof=1) * 100 for m, _ in METRICS] for p in PROMPTS]
    bar_chart([n for _, n in METRICS], [labels[p] for p in PROMPTS], vals, errs, [COLORS[p] for p in PROMPTS],
              'Score (%)', out / 'mina_detection_scores', ylim=(0, 100))

    vals = [[mina[p]['area_mape'].mean()] for p in PROMPTS]
    errs = [[mina[p]['area_mape'].std(ddof=1)] for p in PROMPTS]
    bar_chart(['Area MAPE'], [labels[p] for p in PROMPTS], vals, errs, [COLORS[p] for p in PROMPTS],
              'Area error of matched particles (%)', out / 'mina_area_error', ylim=(0, 20))

    vals = [[mina[p]['matched_iou'].mean()] for p in PROMPTS]
    errs = [[mina[p]['matched_iou'].std(ddof=1)] for p in PROMPTS]
    bar_chart(['Matched IoU'], [labels[p] for p in PROMPTS], vals, errs, [COLORS[p] for p in PROMPTS],
              'IoU of matched particles', out / 'mina_matched_iou', ylim=(0, 1))

    ext = {d: summaries('box', d) for d in EXTERNAL}
    ext = {d: df for d, df in ext.items() if not df.empty}
    names = [EXTERNAL_LABELS[d].replace(' (', '\n(') for d in ext]
    metric_colors = ['#2a78d6', '#eb6834', '#1baf7a']
    vals = [[ext[d][m].mean() * 100 for d in ext] for m, _ in METRICS]
    errs = [[ext[d][m].std(ddof=1) * 100 for d in ext] for m, _ in METRICS]
    fig_path = out / 'external_detection_scores'
    bar_chart(names, [n for _, n in METRICS], vals, errs, metric_colors, 'Score (%)', fig_path, ylim=(0, 105),
              figsize=(6.0, 2.8))


def read_predictions(prompt, dataset, seed=0):
    df = pd.read_csv(run_dir(prompt, seed, dataset) / 'predictions.csv')
    return {name: [json.loads(p) for p in g['polygon_coords']] for name, g in df.groupby('image_name')}


def draw(image, polygons, color, fill):
    gray = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    scale = min(1.0, MAX_SIDE / max(gray.shape[:2]))
    upscale = 4 if max(gray.shape[:2]) <= 512 else 1
    size = (int(round(gray.shape[1] * scale * upscale)), int(round(gray.shape[0] * scale * upscale)))
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_NEAREST if upscale > 1 else cv2.INTER_AREA)
    f = scale * upscale
    pts = [np.round(np.array(p, np.float64) * f).astype(np.int32).reshape(-1, 1, 2) for p in polygons if len(p) >= 3]
    canvas = gray.copy()
    if fill:
        layer = gray.copy()
        for pt in pts:
            cv2.fillPoly(layer, [pt], color)
        canvas = cv2.addWeighted(layer, 0.35, gray, 0.65, 0)
    thickness = max(2, round(max(canvas.shape[:2]) / 400))
    cv2.polylines(canvas, pts, True, color, thickness, lineType=cv2.LINE_AA)
    return canvas


def plain(image):
    return draw(image, [], GT_COLOR, False)


def pick_median(per_image, names):
    df = per_image[per_image['image_name'].isin(names) & (per_image['gt_count'] >= 3)]
    if df.empty:
        df = per_image[per_image['image_name'].isin(names) & (per_image['gt_count'] > 0)]
    df = df.sort_values('f1_score').reset_index(drop=True)
    return df.iloc[len(df) // 2]


def examples(out, dataset, groups, prompts):
    cfg = evaluation.resolve_dataset(dataset)
    per_image = pd.read_csv(run_dir('box', 0, dataset) / 'per_image_metrics.csv')
    preds = {p: read_predictions(p, dataset) for p in prompts}
    rows = []
    for group, names in groups.items():
        if not names:
            continue
        chosen = pick_median(per_image, names)
        name = chosen['image_name']
        image = cv2.imread(str(Path(cfg['image_dir']) / name))
        _, gt = evaluation.load_ground_truth(cfg, name, image.shape)
        stem = f'{dataset}_{group}' if group else dataset
        cv2.imwrite(str(out / f'{stem}_image.png'), plain(image))
        cv2.imwrite(str(out / f'{stem}_ground_truth.png'), draw(image, gt, GT_COLOR, True))
        for p in prompts:
            cv2.imwrite(str(out / f'{stem}_{p}_prompt.png'), draw(image, preds[p].get(name, []), PRED_COLOR, True))
            row = pd.read_csv(run_dir(p, 0, dataset) / 'per_image_metrics.csv').set_index('image_name').loc[name]
            rows.append({'file_prefix': stem, 'source_image': name, 'prompt': p, 'gt': int(row['gt_count']),
                         'predicted': int(row['pred_count']), 'tp': int(row['true_positives']),
                         'precision': row['precision'], 'recall': row['recall'], 'f1': row['f1_score']})
    return rows


def main():
    parser = argparse.ArgumentParser(description='RT-DETR + SAM revision figures, one file per panel.')
    parser.add_argument('--out', default=str(Path.home() / 'sam_figures'))
    args = parser.parse_args()
    out = Path(args.out).expanduser()
    (out / 'qualitative').mkdir(parents=True, exist_ok=True)

    table_s3(out)
    charts(out)

    test_images = [p.name for p in evaluation.list_images(evaluation.resolve_dataset('MiNa_dataset')['image_dir'])]
    rows = examples(out / 'qualitative', 'MiNa_dataset',
                    {pol: [n for n in test_images if n.split('_')[0] == pol] for pol in POLYMERS}, PROMPTS)
    for d in EXTERNAL:
        if (run_dir('box', 0, d) / 'per_image_metrics.csv').exists():
            images = [p.name for p in evaluation.list_images(evaluation.resolve_dataset(d)['image_dir'])]
            rows += examples(out / 'qualitative', d, {'': images}, ('box',))
    pd.DataFrame(rows).to_csv(out / 'qualitative' / 'examples.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
