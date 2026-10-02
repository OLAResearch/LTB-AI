"""Figure 3: mechanism family x failure-mode heatmap.

Rows are mechanism families (families with at least `MIN_COUNT` items, plus a
pooled `Other` row), columns are percentages computed within the row: one
per-model failure column for every translator in `ltb_data.MODELS`, then joint
failure and the contributor's own strict-pass rate.
Implemented with `imshow` so the script has no seaborn dependency.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ltb_data import (FAMILY_LABELS, FAMILY_ORDER, MODEL_SHORT, MODELS, PALETTE,
                      apply_publication_style, family_counts, finalize_figure,
                      load_samples, provider_of, wrap_label)

# Above this many translators, one column each stops fitting and stops being
# readable, so the columns aggregate by provider family instead. The per-model
# view then lives in plot_provider_comparison.py, which is built for it.
MAX_MODEL_COLUMNS = 6

MIN_COUNT = 3
OTHER_LABEL = 'Other'


def _model_column(model):
    """Share of the row's items this model fails (bound early, per model)."""
    return lambda g: 1 - np.mean([s['model_passed_rules'][model] for s in g])


def _family_column(models):
    """Mean failure rate across a provider's models (bound early)."""
    return lambda g: float(np.mean([1 - np.mean([s['model_passed_rules'][m] for s in g])
                                    for m in models]))


def _translator_columns():
    """One column per model, or one per provider family for a wide roster."""
    if len(MODELS) <= MAX_MODEL_COLUMNS:
        return [(f'{MODEL_SHORT[m]}\nfails', _model_column(m)) for m in MODELS]
    columns = []
    for family in FAMILY_ORDER:
        members = [m for m in MODELS if provider_of(m) == family]
        if members:
            label = FAMILY_LABELS.get(family, family).replace(' / ', '/')
            columns.append((f'{label}\nmean fail\n($n$={len(members)})',
                            _family_column(members)))
    return columns


# LTB accepts a submission only if at most two automatic translations pass it.
MAX_PASSES_ADMISSIBLE = 2

# One column per translator, then two summary columns. `strict pass` is not a
# column of its own: a verdict is strict exactly when all models fail, so it
# would duplicate the joint-failure column cell for cell.
COLUMNS = _translator_columns() + [
    (f'All {len(MODELS)} fail', lambda g: np.mean([s['band'] == 'all_fail' for s in g])),
    (f'\u2264{MAX_PASSES_ADMISSIBLE} of {len(MODELS)} pass\n(LTB-admissible)',
     lambda g: np.mean([s['n_translators_passed'] <= MAX_PASSES_ADMISSIBLE for s in g])),
]

# Near-white -> repo blue, so "hard for the models" reads as saturated blue.
# The floor is tinted (not pure white) so that 0% cells still read as cells.
CMAP = LinearSegmentedColormap.from_list(
    'ltb_blue', ['#EDF2F8', PALETTE['blue_secondary'], PALETTE['blue_main']])


def plot_mechanism_heatmap(samples, fig_name: str):
    apply_publication_style(font_size=17, axes_linewidth=2)

    counts = family_counts(samples)
    families = [fam for fam, n in counts.items() if n >= MIN_COUNT]
    pooled = [s for s in samples if s['family'] not in families]
    row_groups = [(fam, [s for s in samples if s['family'] == fam]) for fam in families]
    if pooled:
        row_groups.append((f'{OTHER_LABEL} families', pooled))
    # Hardest families at the top, with the whole pool as a reference row.
    row_groups.sort(key=lambda kv: (-np.mean([s['band'] == 'all_fail' for s in kv[1]]),
                                    -len(kv[1])))
    row_groups.insert(0, ('All items', list(samples)))

    matrix = np.array([[100 * fn(group) for _, fn in COLUMNS]
                       for _, group in row_groups])

    fig = plt.figure(figsize=(max(11, 2.4 * len(COLUMNS) + 4), 11))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(matrix, cmap=CMAP, vmin=0, vmax=100, aspect='auto')

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f'{value:.0f}', ha='center', va='center', fontsize=16,
                    color='white' if value > 55 else 'black')

    ax.set_xticks(np.arange(len(COLUMNS)))
    ax.set_xticklabels([name for name, _ in COLUMNS], fontsize=16)
    ax.set_yticks(np.arange(len(row_groups)))
    ax.set_yticklabels([f'{wrap_label(name, 26)}  ($n$={len(group)})'
                        for name, group in row_groups], fontsize=17)
    # Rule under the reference row, so it does not read as just another family.
    ax.axhline(0.5, color='black', linewidth=2)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='both', which='both', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # White cell separators, matching the repo's heatmap look.
    ax.set_xticks(np.arange(-0.5, len(COLUMNS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_groups), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label('Share of items in the row (%)', fontsize=18, labelpad=12)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.outline.set_visible(False)

    ax.set_title('Failure modes per mechanism family', fontsize=22, pad=56)

    return finalize_figure(fig, fig_name)


if __name__ == '__main__':
    samples = load_samples()
    plot_mechanism_heatmap(samples, 'mechanism_heatmap')
