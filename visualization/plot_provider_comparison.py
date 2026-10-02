"""Figure: the whole translator roster, across providers.

Grouped bars stop being readable past about five models, so the wide roster in
`../data/ltb_all_models.jsonl` gets its own figure:

(a) one horizontal bar per model, item pass rate, grouped by provider family;
(b) model x mechanism-family heatmap of the same rate, so a family that is hard
    for one provider and easy for another is visible at a glance.

Note the two families are judged by different judges (Opus 5 for Claude, Gemini
3.7 Flash for the rest), which the subtitle states: cross-provider gaps are
indicative, not measured.
"""

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ltb_data import (FAMILY_LABELS, FAMILY_ORDER, MODEL_COLORS, MODEL_LABELS, MODELS,
                      PALETTE, apply_publication_style, family_groups, finalize_figure,
                      group_of, load_samples, provider_of, wrap_label)

MIN_COUNT = 5
OTHER_LABEL = 'Other'
# Panel (b) is one column per family: past this many the labels collide.
MAX_GROUPS = 12
KEEP_HARDEST = 4

CMAP = LinearSegmentedColormap.from_list(
    'ltb_blue', ['#EDF2F8', PALETTE['blue_secondary'], PALETTE['blue_main']])


def pass_rate(samples, model):
    if not samples:
        return np.nan
    return 100 * float(np.mean([s['model_passed_rules'][model] for s in samples]))


def judges_used(samples):
    """{judge: [models]} as recorded in the file, for the subtitle."""
    judges = {}
    for sample in samples:
        for model, judge in (sample.get('judges') or {}).items():
            if model in MODELS:
                judges.setdefault(judge or 'unrecorded', set()).add(model)
    return judges


def panel_models(ax, samples):
    """One horizontal bar per model, blocked by provider and sorted within it."""
    ordered, ticks = [], []
    for family in FAMILY_ORDER:
        members = [m for m in MODELS if provider_of(m) == family]
        if not members:
            continue
        members.sort(key=lambda m: pass_rate(samples, m))
        ordered.extend(members)
        ticks.append((family, len(ordered)))

    y = np.arange(len(ordered))[::-1]
    rates = [pass_rate(samples, m) for m in ordered]
    ax.barh(y, rates, height=0.72, color=[MODEL_COLORS[m] for m in ordered],
            edgecolor='black', linewidth=1.8)
    for yi, (rate, model) in zip(y, zip(rates, ordered)):
        ax.text(rate + 1.2, yi, f'{rate:.0f}', va='center', ha='left', fontsize=15)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[m] for m in ordered], fontsize=15)
    # A rule between provider blocks, labelled on the right.
    for family, boundary in ticks[:-1]:
        ax.axhline(len(ordered) - boundary - 0.5, color='black', linewidth=1.2,
                   linestyle=':')
    start = 0
    for family, boundary in ticks:
        middle = len(ordered) - (start + boundary) / 2 - 0.5
        ax.text(103, middle, FAMILY_LABELS.get(family, family), rotation=270,
                va='center', ha='left', fontsize=15, color=PALETTE['gray_dark'])
        start = boundary

    ax.set_xlim([0, 108])
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel('Items passing every verification rule (%)', fontsize=17, labelpad=10)
    ax.set_title('(a) Pass rate per translator', fontsize=20, pad=16, loc='left')


def panel_heatmap(ax, samples):
    """Model x mechanism-family pass rate."""
    groups = family_groups(samples, min_count=MIN_COUNT, other_label=OTHER_LABEL,
                           max_groups=MAX_GROUPS, keep_hardest=KEEP_HARDEST)
    by_group = {g: [s for s in samples if group_of(s, groups, OTHER_LABEL) == g]
                for g in groups}
    named = [g for g in groups if g != OTHER_LABEL]
    named.sort(key=lambda g: np.mean([pass_rate(by_group[g], m) for m in MODELS]))
    order = named + ([OTHER_LABEL] if OTHER_LABEL in by_group else [])

    rows = [m for family in FAMILY_ORDER for m in MODELS if provider_of(m) == family]
    matrix = np.array([[pass_rate(by_group[g], m) for g in order] for m in rows])

    im = ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=100, aspect='auto')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, '' if np.isnan(value) else f'{value:.0f}',
                    ha='center', va='center', fontsize=12,
                    color='white' if value > 55 else 'black')

    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f'{wrap_label(g, 11)}\n($n$={len(by_group[g])})' for g in order],
                       fontsize=11, linespacing=1.15)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in rows], fontsize=13)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)
    ax.set_title('(b) Pass rate by mechanism family (%)', fontsize=20, pad=64, loc='left')
    return im


def plot_provider_comparison(samples, fig_name: str):
    # The whole point of this figure is the comparison BETWEEN providers: the bar
    # blocking, the family colours and the differing-judges caveat all say nothing
    # about a roster drawn from one provider, where it would merely restate
    # plot_mechanism_heatmap. So it declines rather than drawing that.
    providers = {provider_of(m) for m in MODELS}
    if len(providers) < 2:
        print(f'  skipping {fig_name}: every model is from one provider '
              f'({", ".join(sorted(providers))}), so there is nothing to compare '
              f'across providers. Per-model rates are in the mechanism heatmap; '
              f'point LTB_JSONL at data/ltb_all_models.jsonl for this figure.')
        return []
    apply_publication_style(font_size=15, axes_linewidth=2)

    height = max(7.0, 0.42 * len(MODELS) + 4.5)
    fig = plt.figure(figsize=(23, height))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.75])
    panel_models(fig.add_subplot(gs[0, 0]), samples)
    im = panel_heatmap(fig.add_subplot(gs[0, 1]), samples)
    cbar = fig.colorbar(im, ax=fig.axes[1], fraction=0.025, pad=0.02)
    cbar.set_ticks([0, 50, 100])
    cbar.outline.set_visible(False)

    judges = judges_used(samples)
    if len(judges) > 1:
        detail = '; '.join(f'{judge} judged {len(models)} model'
                           f'{"s" if len(models) != 1 else ""}'
                           for judge, models in sorted(judges.items()))
        fig.suptitle(f'{len(MODELS)} translators over {len(samples)} items — '
                     f'different judges per family ({detail}), so cross-provider '
                     f'gaps are indicative, not measured',
                     fontsize=16, y=0.995, color=PALETTE['gray_dark'])
    return finalize_figure(fig, fig_name, pad=2.4)


if __name__ == '__main__':
    samples = load_samples()
    plot_provider_comparison(samples, 'provider_comparison')
