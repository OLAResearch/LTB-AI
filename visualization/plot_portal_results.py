"""Figure: how the submitted items fared at the official LTB portal.

Three panels, all from `../data/ltb_validation.jsonl`:

(a) what became of the 45 submitted items;
(b) per rule, how many of the portal's translators failed it — the shape that
    shows one rule carrying each item while its companion is passed by everyone;
(c) the outcome per mechanism family.

The portal runs its own roster and judges the reference translation too, which
is why items can be invalid rather than merely easy.
"""

import json
import os

import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

from ltb_data import (PALETTE, apply_publication_style, finalize_figure, wrap_label)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
SUBMISSIONS = os.path.join(ROOT, 'data', 'ltb_validation.jsonl')

# LTB accepts a submission when at most two automatic translations pass.
MAX_PASSES = 2

STATUS_ORDER = ['passed', 'failed', 'disputed', 'not_checked']
STATUS_LABELS = {
    'passed': 'Passed\n($\\leq$2 systems passed)',
    'failed': 'Failed\n(3 systems passed)',
    'disputed': 'Invalid\n(our reference failed)',
    'not_checked': 'Not checked\nthis round',
}
STATUS_COLORS = {
    'passed': PALETTE['blue_main'],
    'failed': PALETTE['red_strong'],
    'disputed': PALETTE['highlight'],
    'not_checked': PALETTE['neutral'],
}


def load(path=SUBMISSIONS):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def panel_status(ax, rows):
    counts = [sum(1 for r in rows if (r['portal'] or {}).get('status') == s)
              for s in STATUS_ORDER]
    y = np.arange(len(STATUS_ORDER))[::-1]
    ax.barh(y, counts, height=0.66, color=[STATUS_COLORS[s] for s in STATUS_ORDER],
            edgecolor='black', linewidth=2)
    for yi, value in zip(y, counts):
        if value:
            ax.text(value + 0.6, yi, f'{value}', va='center', ha='left', fontsize=17)
    ax.set_yticks(y)
    ax.set_yticklabels([STATUS_LABELS[s] for s in STATUS_ORDER], fontsize=15)
    ax.set_xlim([0, max(counts) * 1.25])
    ax.set_xlabel('Items', fontsize=17, labelpad=10)
    ax.set_title(f'(a) {len(rows)} items submitted', fontsize=20, pad=16, loc='left')


def panel_rules(ax, rows):
    """How many translators failed each rule, one dot per rule."""
    carrying, companion = [], []
    for row in rows:
        results = (row['portal'] or {}).get('rule_results') or {}
        if not results:
            continue
        total = row['portal']['n_translators_total']
        fails = [sum(1 for v in results.values() if v[i] is False)
                 for i in range(row['n_rules'])]
        hardest = max(fails)
        for value in fails:
            (carrying if value == hardest else companion).append(100 * value / total)

    bins = np.arange(0, 110, 10)
    ax.hist([carrying, companion], bins=bins, stacked=True,
            color=[PALETTE['blue_main'], PALETTE['green_2']],
            edgecolor='black', linewidth=1.6,
            label=[f'Hardest rule of its item (n={len(carrying)})',
                   f'Companion rule (n={len(companion)})'])
    ax.axvspan(0, 20, color=PALETTE['red_1'], alpha=0.35, zorder=0)
    ax.text(10, ax.get_ylim()[1] * 0.55, 'passed by\n$\\geq$80%\nof systems',
            ha='center', va='center', fontsize=12, color=PALETTE['red_strong'])
    ax.set_xlabel('Share of portal translators failing the rule (%)', fontsize=17,
                  labelpad=10)
    ax.set_ylabel('Rules', fontsize=17, labelpad=10)
    ax.legend(fontsize=12, loc='upper center', frameon=False)
    ax.set_title('(b) Which rules do the work', fontsize=20, pad=16, loc='left')


def panel_families(ax, rows):
    """Outcome per mechanism family, so it is visible which ones survived."""
    families = {}
    for row in rows:
        family = row['mechanism_en'].split(':')[0]
        families.setdefault(family, []).append((row['portal'] or {}).get('status'))
    order = sorted(families, key=lambda f: (-sum(1 for s in families[f] if s == 'passed'),
                                            -len(families[f])))

    y = np.arange(len(order))[::-1]
    left = np.zeros(len(order))
    for status in STATUS_ORDER:
        widths = np.array([sum(1 for s in families[f] if s == status) for f in order],
                          dtype=float)
        ax.barh(y, widths, left=left, height=0.68, color=STATUS_COLORS[status],
                edgecolor='black', linewidth=1.6,
                label=STATUS_LABELS[status].split('\n')[0])
        for yi, (width, start) in enumerate(zip(widths, left)):
            if width >= 2:
                ax.text(start + width / 2, y[yi], f'{int(width)}', ha='center',
                        va='center', fontsize=13,
                        color='white' if status == 'passed' else 'black')
        left = left + widths

    ax.set_yticks(y)
    ax.set_yticklabels([f'{wrap_label(f, 22)}' for f in order], fontsize=13)
    ax.set_xlabel('Items submitted', fontsize=17, labelpad=10)
    ax.set_xlim([0, left.max() * 1.08])
    ax.legend(fontsize=12, loc='lower right', frameon=False)
    ax.set_title('(c) Outcome by mechanism', fontsize=20, pad=16, loc='left')


def plot_portal_results(samples, fig_name: str):
    """`samples` is ignored: this figure reads the submission record itself."""
    rows = load()
    apply_publication_style(font_size=15, axes_linewidth=2)
    fig = plt.figure(figsize=(22, 7.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.1, 1.25])
    panel_status(fig.add_subplot(gs[0, 0]), rows)
    panel_rules(fig.add_subplot(gs[0, 1]), rows)
    panel_families(fig.add_subplot(gs[0, 2]), rows)
    return finalize_figure(fig, fig_name, pad=2.2)


if __name__ == '__main__':
    plot_portal_results(None, 'portal_results')
