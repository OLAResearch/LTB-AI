"""Figure 1: how often each mechanism family defeats the machine translators.

One stacked bar per mechanism family, split by how much of the roster in
`ltb_data.MODELS` the item defeats. A small roster splits three ways (all fail /
some fail / all pass); from five models up the middle is graded, because "some
fail" would otherwise cover everything from one slip to near-total failure.
Higher blue = the family is a more reliable source of adversarial items.
"""

import numpy as np
from matplotlib import gridspec
from matplotlib import patheffects as path_effects
from matplotlib import pyplot as plt

from ltb_data import (BAND_COLORS, BAND_HATCH, BAND_LABELS, BANDS, FAMILY_SHORT,
                      MODELS, apply_publication_style, band_fractions, family_groups,
                      finalize_figure, group_of, load_samples, wrap_label)

MIN_COUNT = 5          # families smaller than this are pooled into `Other`
OTHER_LABEL = 'Other'
MAX_GROUPS = 12        # and past this many, the smallest named ones are too


def plot_family_outcomes(samples, fig_name: str):
    apply_publication_style(font_size=24, axes_linewidth=3)

    groups = family_groups(samples, min_count=MIN_COUNT, other_label=OTHER_LABEL,
                           max_groups=MAX_GROUPS, keep_hardest=4)
    by_group = {g: [s for s in samples if group_of(s, groups, OTHER_LABEL) == g]
                for g in groups}
    # Sort by adversarial yield (all models fail), keeping `Other` last.
    named = [g for g in groups if g != OTHER_LABEL]
    named.sort(key=lambda g: -band_fractions(by_group[g])[0])
    groups = named + ([OTHER_LABEL] if OTHER_LABEL in by_group else [])

    fractions = np.stack([band_fractions(by_group[g]) for g in groups], axis=0)
    counts = np.array([len(by_group[g]) for g in groups])

    # Both the roster and the number of families vary, so the canvas and the
    # type scale with them instead of assuming the original seven-family layout.
    crowded = len(groups) > 8
    fig = plt.figure(figsize=(max(24, 1.9 * len(groups) + 8), 10))
    gs = gridspec.GridSpec(1, 5, figure=fig)
    ax = fig.add_subplot(gs[0, :4])

    x = np.arange(len(groups))
    bottom = np.zeros(len(groups))
    label_size = (24 if len(BANDS) <= 3 else 19) - (5 if crowded else 0)
    for band_idx, band in enumerate(BANDS):
        height = fractions[:, band_idx]
        ax.bar(x, height, bottom=bottom, width=0.72,
               color=BAND_COLORS[band], hatch=BAND_HATCH[band],
               edgecolor='black', linewidth=2.5, label=BAND_LABELS[band])
        for xi, (h, b) in enumerate(zip(height, bottom)):
            if h < 0.07:                      # too thin to hold a legible number
                continue
            ax.text(xi, b + h / 2, f'{100 * h:.0f}%',
                    ha='center', va='center', fontsize=label_size,
                    color='#FFD700' if band == 'all_fail' else 'black',
                    path_effects=[path_effects.Stroke(linewidth=4, foreground='black'),
                                  path_effects.Normal()]
                    if band == 'all_fail' else None)
        bottom = bottom + height

    ax.set_xticks(x)
    ax.set_xticklabels([f'{FAMILY_SHORT.get(g, wrap_label(g, 12 if crowded else 16))}'
                        f'\n($n$={n})' for g, n in zip(groups, counts)],
                       fontsize=17 if crowded else 24, linespacing=1.1)
    ax.set_ylim([0, 1.0])
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f'{int(100 * v)}%' for v in np.linspace(0, 1, 6)])
    ax.set_ylabel('Share of candidate items', fontsize=30, labelpad=14)
    ax.set_title('Which Chinese-to-English mechanisms break machine translation?',
                 fontsize=34, pad=28)
    ax.tick_params(axis='both', which='major', length=8, width=3)

    # Dedicated legend panel keeps the data panel clean.
    ax_legend = fig.add_subplot(gs[0, 4])
    handles, labels = ax.get_legend_handles_labels()
    ax_legend.legend(handles, labels, fontsize=26, loc='center left', frameon=False,
                     handlelength=1.6, handleheight=1.6, labelspacing=1.0)
    ax_legend.set_axis_off()

    return finalize_figure(fig, fig_name)


if __name__ == '__main__':
    samples = load_samples()
    plot_family_outcomes(samples, 'family_outcomes')
