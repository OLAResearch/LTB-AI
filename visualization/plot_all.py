"""Regenerate every Last Translation Benchmark figure into `../png/` and `../pdf/`.

Every figure here reports how the translators performed - our own roster, and
the official LTB portal's. Descriptions of
the corpus itself (source lengths, how many items each mechanism holds, how the
annotation is composed) are not figures: they are one-line facts, and the item
counts they carried appear on the axis labels of the figures below.
"""

from ltb_data import MODELS, load_samples
from plot_family_outcomes import plot_family_outcomes
from plot_mechanism_heatmap import plot_mechanism_heatmap
from plot_portal_results import plot_portal_results
from plot_provider_comparison import plot_provider_comparison
from plot_yield_funnel import plot_yield_funnel

FIGURES = [
    # what the roster does, model by model
    ('provider_comparison', plot_provider_comparison),
    # which mechanisms defeat how much of the roster
    ('family_outcomes', plot_family_outcomes),
    ('mechanism_heatmap', plot_mechanism_heatmap),
    # what survives, and which models fail together
    ('yield_funnel', plot_yield_funnel),
    # how the submitted items fared at the official portal
    ('portal_results', plot_portal_results),
]


if __name__ == '__main__':
    samples = load_samples()
    print(f'Loaded {len(samples)} candidate items, {len(MODELS)} translators.')
    for name, plot_fn in FIGURES:
        plot_fn(samples, name)
