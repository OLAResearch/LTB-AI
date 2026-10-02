"""Figure 4: what survives the pipeline, in three panels.

(a) Nested funnel from drafted candidate to strictly submittable item.
(b) Co-failure matrix: which translators fail the same items.
(c) Self-review verdict composition, split by how many models were defeated.
"""

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ltb_data import (FAMILY_ORDER, MODEL_SHORT, MODELS, PALETTE, VERDICT_COLORS,
                      VERDICT_LABELS, VERDICT_ORDER, apply_publication_style,
                      finalize_figure, load_samples, provider_of)

FUNNEL_COLORS = [PALETTE['neutral'], PALETTE['green_2'], PALETTE['green_3'],
                 PALETTE['blue_secondary'], PALETTE['blue_main']]

# LTB's own admissibility bar: at most two automatic translations may pass a
# submission's rules. It is a fixed number in the task description, not a
# fraction of the roster, so it does not scale with len(MODELS).
MAX_PASSES_ADMISSIBLE = 2


def funnel_stages(samples):
    """Strictly nested filters, so each stage is a subset of the one above."""
    breaks_one = [s for s in samples if s['n_translators_passed'] < len(MODELS)]
    admissible = [s for s in breaks_one if s['n_translators_passed'] <= MAX_PASSES_ADMISSIBLE]
    kept = [s for s in admissible if s['candidate_verdict'] != 'rejected']
    strict = [s for s in kept if s['candidate_verdict'] == 'strict_pass']
    return [
        ('Drafted candidates', samples),
        ('At least one model fails', breaks_one),
        (f'LTB-admissible (\u2264{MAX_PASSES_ADMISSIBLE} of {len(MODELS)} pass)', admissible),
        ('Kept after curator review', kept),
        ('Strict pass (curator verdict)', strict),
    ]


def panel_funnel(ax, samples):
    stages = funnel_stages(samples)
    total = len(samples)
    y = np.arange(len(stages))[::-1]
    values = [len(group) for _, group in stages]

    ax.barh(y, values, height=0.66, color=FUNNEL_COLORS, edgecolor='black', linewidth=2)
    for yi, value in zip(y, values):
        ax.text(value + 0.015 * total, yi, f'{value}  ({100 * value / total:.0f}%)',
                va='center', ha='left', fontsize=17)

    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _ in stages], fontsize=17)
    ax.set_xlim([0, total * 1.24])
    ax.set_xlabel('Number of items', fontsize=18, labelpad=10)
    ax.set_title('(a) Candidate yield', fontsize=21, pad=18, loc='left')


def panel_agreement(ax, samples):
    """matrix[i, j] = items that BOTH model i and model j fail.

    The diagonal is each model's own failure count, so a row reads as "of the N
    items this model fails, how many does each other model fail too" -- the
    two-model agreement matrix generalised to a roster of any size.
    """
    # Models grouped by provider, so any family-shaped clustering is visible.
    ordered = [m for family in FAMILY_ORDER for m in MODELS if provider_of(m) == family]
    n = len(ordered)
    fails = [{s['id'] for s in samples if not s['model_passed_rules'][m]} for m in ordered]

    wide = n > 6
    if wide:
        # Counts stop being comparable once models fail very different numbers of
        # items, so a wide roster shows the conditional share instead: of what the
        # row model fails, how much does the column model fail too.
        matrix = np.array([[100 * len(fails[i] & fails[j]) / len(fails[i])
                            if fails[i] else 0.0 for j in range(n)] for i in range(n)])
        vmax = 100.0
    else:
        matrix = np.array([[float(len(fails[i] & fails[j])) for j in range(n)]
                           for i in range(n)])
        vmax = matrix.max() or 1.0

    cmap = LinearSegmentedColormap.from_list(
        'ltb_agree', ['#EDF2F8', PALETTE['blue_secondary'], PALETTE['blue_main']])
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    cell_size = 17 if n <= 4 else 13 if n <= 6 else 9 if n <= 10 else 7
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            ax.text(j, i, f'{value:.0f}', ha='center', va='center',
                    fontsize=cell_size, fontweight='bold' if i == j else 'normal',
                    color='white' if value > 0.55 * vmax else 'black')

    # Rules between provider blocks.
    boundary = 0
    for family in FAMILY_ORDER[:-1]:
        boundary += sum(1 for m in ordered if provider_of(m) == family)
        if 0 < boundary < n:
            ax.axhline(boundary - 0.5, color=PALETTE['gray_dark'], linewidth=1.6)
            ax.axvline(boundary - 0.5, color=PALETTE['gray_dark'], linewidth=1.6)

    labels = [MODEL_SHORT[m] for m in ordered]
    ax.set_xticks(np.arange(n))
    tick_size = 14 if n <= 6 else 10 if n <= 10 else 8
    ax.set_xticklabels(labels, fontsize=tick_size, rotation=45, ha='left')
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=tick_size)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=3)
    ax.tick_params(which='minor', length=0)
    subtitle = ('cell: % of the row model\'s failures the column model shares'
                if wide else
                'cell: items the pair both fail; diagonal: all its failures')
    ax.set_title(f'(b) Shared failures\n      {subtitle}', fontsize=17, pad=58, loc='left')
    return im


def panel_verdicts(ax, samples):
    n_models = len(MODELS)
    # One tick per bucket is unreadable past a handful of models: label the ends
    # and every other bucket instead, and say the denominator on the axis.
    step = 1 if n_models <= 6 else 2
    buckets = [((f'{k}' if n_models > 6 else f'{k} of {n_models}')
                if (k % step == 0 or k in (0, n_models)) else '', k)
               for k in range(n_models + 1)]
    x = np.arange(len(buckets), dtype=float)
    bottom = np.zeros(len(buckets))
    for verdict in VERDICT_ORDER:
        heights = np.array([sum(1 for s in samples
                                if s['n_translators_passed'] == k and s['candidate_verdict'] == verdict)
                            for _, k in buckets], dtype=float)
        ax.bar(x, heights, bottom=bottom, width=0.66 if len(buckets) <= 6 else 0.86,
               color=VERDICT_COLORS[verdict],
               edgecolor='black', linewidth=2, label=VERDICT_LABELS[verdict])
        for xi, (h, b) in enumerate(zip(heights, bottom)):
            if h <= 0:
                continue
            if len(buckets) > 8 and h < 6:    # too many bars to label every sliver
                continue
            if h >= 12:                       # thick enough to hold the number
                ax.text(xi, b + h / 2, f'{int(h)}', ha='center', va='center',
                        fontsize=16 if len(buckets) <= 6 else 12,
                        color='white' if verdict != 'rejected' else 'black')
            else:                             # thin slice: label just outside the bar
                ax.text(xi + 0.38, b + h / 2, f'{int(h)}', ha='left', va='center',
                        fontsize=14 if len(buckets) <= 6 else 11, color='black')
        bottom = bottom + heights

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in buckets],
                       fontsize=16 if len(buckets) <= 6 else 13)
    ax.set_xlabel(f'Models passing every verification rule (of {len(MODELS)})',
                  fontsize=18, labelpad=10)
    ax.set_ylabel('Number of items', fontsize=18, labelpad=10)
    ax.set_ylim([0, max(bottom) * 1.22])
    ax.set_xlim([-0.6, len(buckets) - 0.25])
    ax.legend(fontsize=16, loc='upper left', frameon=False)
    ax.set_title('(c) Verdicts by translators passed', fontsize=21, pad=18, loc='left')


def plot_yield_funnel(samples, fig_name: str):
    apply_publication_style(font_size=17, axes_linewidth=2)

    fig = plt.figure(figsize=(22, 6.5 if len(MODELS) <= 6 else 8.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.5, 1.0, 1.15])
    panel_funnel(fig.add_subplot(gs[0, 0]), samples)
    panel_agreement(fig.add_subplot(gs[0, 1]), samples)
    panel_verdicts(fig.add_subplot(gs[0, 2]), samples)

    return finalize_figure(fig, fig_name)


if __name__ == '__main__':
    samples = load_samples()
    plot_yield_funnel(samples, 'yield_funnel')
