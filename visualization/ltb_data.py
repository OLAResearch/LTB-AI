"""Shared style, palette and data loading for the Last Translation Benchmark figures.

Style conventions follow `figures4papers/scientific-figure-making`:
minimalist spines, Helvetica-like sans fonts, blue/green/red/neutral palette,
`tight_layout(pad=2)` and 300 dpi PNG + vector PDF export.
"""

import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
from matplotlib import pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
# Which file the figures read, and which models they report on. Both can be
# overridden from the environment, so the same scripts plot the Claude-only pool
# or the collected multi-provider file without editing anything:
#
#   LTB_JSONL=../data/ltb_all_models.jsonl python3 plot_all.py
#   LTB_JSONL=../data/ltb_all_models.jsonl LTB_MODELS=claude-opus-4-5,gpt-5.6-luna \
#       python3 plot_all.py
#
# With LTB_JSONL set and LTB_MODELS unset, the roster is read from the file:
# every model present on every item, so no figure is computed over a model that
# only some items carry.
# The canonical dataset is the multi-provider collection: it carries every item
# field plus every translator that has been run, so the figures read it directly.
DEFAULT_JSONL = os.environ.get('LTB_JSONL') or os.path.join(DATA_DIR, 'ltb_all_models.jsonl')
if not os.path.isabs(DEFAULT_JSONL):
    DEFAULT_JSONL = os.path.abspath(os.path.join(os.getcwd(), DEFAULT_JSONL))

# Figures from a non-default file get a suffix, so plotting the multi-provider
# collection never overwrites the figures built from the Claude pool. Override
# with LTB_FIG_SUFFIX (set it empty to write over the default names).
_DEFAULT_STEM = 'ltb_all_models'
_STEM = os.path.splitext(os.path.basename(DEFAULT_JSONL))[0]
FIG_SUFFIX = os.environ.get(
    'LTB_FIG_SUFFIX',
    '' if _STEM == _DEFAULT_STEM else '_' + _STEM.replace('ltb_', ''))

# One output directory per format: rasters in png/, vectors in pdf/.
FORMAT_DIRS = {
    'png': os.path.join(PROJECT_ROOT, 'png'),
    'pdf': os.path.join(PROJECT_ROOT, 'pdf'),
}


# ---------------------------------------------------------------- style / color

PALETTE = {
    'blue_main': '#0F4D92',
    'blue_secondary': '#3775BA',
    'green_1': '#DDF3DE',
    'green_2': '#AADCA9',
    'green_3': '#8BCF8B',
    'red_1': '#F6CFCB',
    'red_2': '#E9A6A1',
    'red_strong': '#B64342',
    'neutral': '#CFCECE',
    'gray_mid': '#767676',
    'gray_dark': '#4D4D4D',
    'highlight': '#FFD700',
    'teal': '#42949E',
    'violet': '#9A4D8E',
}

# Semantics used consistently across every LTB figure.
#   blue   = item defeats every translator in the roster -> a useful benchmark item
#   green  = item defeats some but not all translators   -> partially useful
#   neutral= item is solved by every translator          -> not adversarial
OUTCOME_COLORS = {
    'all_fail': PALETTE['blue_main'],
    'some_fail': PALETTE['green_3'],
    'all_pass': PALETTE['neutral'],
}
OUTCOME_ORDER = ['all_fail', 'some_fail', 'all_pass']
OUTCOME_HATCH = {'all_fail': '', 'some_fail': '///', 'all_pass': '...'}

# `some_fail` is one bucket, which is fine for a handful of models and useless
# for a wide roster: with 14 translators it holds "one model slipped" and
# "thirteen of fourteen failed" side by side. Above WIDE_ROSTER models the
# figures grade the middle instead, by the fraction of the roster that failed.
WIDE_ROSTER = 5
BAND_ORDER_WIDE = ['all_fail', 'most_fail', 'mixed', 'few_fail', 'all_pass']
BAND_COLORS = {
    'all_fail': PALETTE['blue_main'],
    'most_fail': PALETTE['blue_secondary'],
    'mixed': PALETTE['green_3'],
    'few_fail': PALETTE['green_2'],
    'all_pass': PALETTE['neutral'],
    'some_fail': PALETTE['green_3'],
}
BAND_HATCH = {'all_fail': '', 'most_fail': '\\\\', 'mixed': '///',
              'few_fail': 'xx', 'all_pass': '...', 'some_fail': '///'}


def band_of(n_passed: int, n_models: int) -> str:
    """Which outcome band an item falls in, given how many models passed it."""
    if n_models <= 0:
        return 'all_pass'
    if n_passed == 0:
        return 'all_fail'
    if n_passed == n_models:
        return 'all_pass'
    if n_models < WIDE_ROSTER:
        return 'some_fail'
    failed = 1 - n_passed / n_models
    if failed >= 2 / 3:
        return 'most_fail'
    if failed > 1 / 3:
        return 'mixed'
    return 'few_fail'

VERDICT_COLORS = {
    'strict_pass': PALETTE['blue_main'],
    'marginal_pass': PALETTE['blue_secondary'],
    'rejected': PALETTE['neutral'],
}
VERDICT_LABELS = {
    'strict_pass': 'Strict pass',
    'marginal_pass': 'Marginal pass',
    'rejected': 'Rejected',
}
VERDICT_ORDER = ['strict_pass', 'marginal_pass', 'rejected']

# The translator roster the figures report on, in display order. Every figure
# derives its model count from this list, so probing more translators is a
# matter of extending it (and the two dicts below) -- nothing downstream
# assumes a particular number of models.
DEFAULT_MODELS = ['claude-sonnet-5', 'claude-haiku-4-5-20251001',
                  'claude-sonnet-4-5', 'claude-opus-4-5']

# Colours are assigned per provider family, cycling within it, so Claude columns
# read blue/violet, Google green/teal and OpenAI red wherever they appear.
FAMILY_COLORS = {
    'anthropic': [PALETTE['blue_main'], PALETTE['violet'], PALETTE['blue_secondary'],
                  PALETTE['teal']],
    'google': [PALETTE['green_3'], PALETTE['teal'], PALETTE['green_2'],
               PALETTE['blue_secondary'], PALETTE['green_1']],
    'openai': [PALETTE['red_strong'], PALETTE['red_2'], PALETTE['red_1']],
    'other': [PALETTE['gray_mid'], PALETTE['gray_dark'], PALETTE['neutral']],
}
FAMILY_ORDER = ['anthropic', 'google', 'openai', 'other']
FAMILY_LABELS = {'anthropic': 'Claude', 'google': 'Gemini / Gemma',
                 'openai': 'GPT', 'other': 'Other'}

# Display names for model keys the file does not name itself.
KNOWN_LABELS = {
    'claude-sonnet-5': 'Claude Sonnet 5',
    'claude-haiku-4-5-20251001': 'Claude Haiku 4.5',
    'claude-sonnet-4-5': 'Claude Sonnet 4.5',
    'claude-opus-4-5': 'Claude Opus 4.5',
    'claude-opus-5': 'Claude Opus 5',
    'gemma-4-26b-a4b-it': 'Gemma 4 26B',
    'gemma-4-31b-it': 'Gemma 4 31B',
    'gpt-5.6-luna': 'GPT-5.6 Luna',
    'gpt-5.6-sol': 'GPT-5.6 Sol',
}


def provider_of(model: str) -> str:
    """Which family a model key belongs to, by prefix."""
    for prefix, family in (('claude', 'anthropic'), ('gemini', 'google'),
                           ('gemma', 'google'), ('gpt', 'openai'), ('o1', 'openai')):
        if model.startswith(prefix):
            return family
    return 'other'


def label_for(model: str) -> str:
    """'gemini-3.7-flash' -> 'Gemini 3.7 Flash'; known keys keep their proper name."""
    if model in KNOWN_LABELS:
        return KNOWN_LABELS[model]
    words = []
    for chunk in model.split('-'):
        if chunk.isdigit() and words and words[-1][-1].isdigit():
            words[-1] = f'{words[-1]}.{chunk}'          # 4-5 -> 4.5
        elif chunk.isdigit():
            words.append(chunk)
        else:
            words.append(chunk if chunk.isupper() else chunk.capitalize())
    return ' '.join(words).replace('Gpt', 'GPT')


# A model has to cover this share of the items to join the roster. A strict
# intersection would drop a whole column over one unprobed item; the loader then
# skips the handful of items the chosen roster does not cover.
MIN_COVERAGE = 0.9


def models_in(path: str, min_coverage: float = MIN_COVERAGE) -> List[str]:
    """Model keys covering at least `min_coverage` of the items in `path`."""
    seen: Dict[str, int] = {}
    items = 0
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            items += 1
            for key in (json.loads(line).get('model_passed_rules') or {}):
                seen[key] = seen.get(key, 0) + 1
    if not items:
        return []
    kept = [m for m, n in seen.items() if n >= min_coverage * items]
    return sorted(kept, key=lambda m: (FAMILY_ORDER.index(provider_of(m)), m))


def _resolve_models() -> List[str]:
    requested = os.environ.get('LTB_MODELS')
    if requested:
        return [m.strip() for m in requested.split(',') if m.strip()]
    if os.path.exists(DEFAULT_JSONL):
        found = models_in(DEFAULT_JSONL)
        if found:
            _warn_about_partial_models(found)
            return found
    return list(DEFAULT_MODELS)


def _warn_about_partial_models(kept: Sequence[str]) -> None:
    """Say so when models are dropped for not covering every item."""
    seen = set()
    with open(DEFAULT_JSONL, 'r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                seen |= set(json.loads(line).get('model_passed_rules') or {})
    dropped = sorted(seen - set(kept))
    if dropped:
        print(f'ltb_data: plotting {len(kept)} of {len(seen)} models; {len(dropped)} cover '
              f'less than {100 * MIN_COVERAGE:.0f}% of the items and would otherwise be '
              f'scored as failing the ones they never saw '
              f'({", ".join(dropped[:4])}{", ..." if len(dropped) > 4 else ""}). '
              f'Set LTB_MODELS to override.', file=sys.stderr)


MODELS = _resolve_models()
MODEL_LABELS = {model: label_for(model) for model in MODELS}
MODEL_COLORS = {}
for _family in FAMILY_ORDER:
    _palette = FAMILY_COLORS[_family]
    for _index, _model in enumerate([m for m in MODELS if provider_of(m) == _family]):
        MODEL_COLORS[_model] = _palette[_index % len(_palette)]
# Bare model names for tick labels, where "Claude" on every one is just noise.
MODEL_SHORT = {model: label.replace('Claude ', '')
               for model, label in MODEL_LABELS.items()}

# Written out once the roster size is known, so the wording tracks the roster.
OUTCOME_LABELS = {
    'all_fail': f'All {len(MODELS)} models fail',
    'some_fail': 'Some models fail',
    'all_pass': 'All models pass',
}


def apply_publication_style(font_size: int = 16, axes_linewidth: float = 2.0) -> None:
    """Configure rcParams once, before any figure is created."""
    plt.rcParams['font.family'] = ['Helvetica', 'Arial', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = axes_linewidth
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42


def finalize_figure(fig, name: str, dpi: int = 300, pad: float = 2.0,
                    formats: Sequence[str] = ('png', 'pdf')) -> List[str]:
    """`tight_layout` then save the figure once per format.

    `name` is a bare basename (no directory, no extension); rasters land in
    `<project root>/png/` and vectors in `<project root>/pdf/`.
    """
    fig.tight_layout(pad=pad)
    name = os.path.splitext(os.path.basename(name))[0] + FIG_SUFFIX
    saved = []
    for fmt in formats:
        out_dir = FORMAT_DIRS.get(fmt, PROJECT_ROOT)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{name}.{fmt}')
        fig.savefig(out_path, dpi=dpi)
        saved.append(out_path)
    plt.close(fig)
    print('Saved: ' + ', '.join(saved))
    return saved


# ------------------------------------------------------------------------- data

# Mechanism families, in the display order used by every figure. Fine-grained
# `mechanism_en` labels (74 of them) collapse onto these via `mechanism_family`.
FAMILY_SHORT = {
    'Formal register': 'Formal\nregister',
    'Coined abstract noun': 'Coined\nabstract noun',
    'Politeness bias': 'Politeness\nbias',
    'Pragmatic self-reference': 'Pragmatic\nself-reference',
    'Form is meaning': 'Form is\nmeaning',
    'Pseudo-English loanword': 'Pseudo-English\nloanword',
    'Cultural knowledge': 'Cultural\nknowledge',
    'Source-defect fidelity': 'Source-defect\nfidelity',
    'Obligatory English marking': 'Obligatory English\nmarking',
    'Address term': 'Address\nterm',
    'In-group jargon': 'In-group\njargon',
    'Numeral computation': 'Numeral\ncomputation',
    'Irony': 'Irony',
}


def mechanism_family(mechanism_en: str) -> str:
    """Collapse a fine-grained mechanism label onto its family.

    'Formal register: legal terminology' -> 'Formal register'
    'Cultural knowledge: kinship terms'  -> 'Cultural knowledge'
    """
    return mechanism_en.split(':')[0].strip()


def mechanism_subtype(mechanism_en: str) -> str:
    """The part after the colon, or the label itself when there is none."""
    parts = mechanism_en.split(':', 1)
    return parts[1].strip() if len(parts) == 2 else mechanism_en.strip()


def band_labels(n_models: int) -> Dict[str, str]:
    """Legend text for the bands, worded for the roster actually in use."""
    return {
        'all_fail': f'All {n_models} fail',
        'most_fail': 'Two thirds or more fail',
        'mixed': 'A third to two thirds fail',
        'few_fail': 'Up to a third fail',
        'some_fail': 'Some models fail',
        'all_pass': 'All models pass',
    }


def active_bands(n_models: int = None) -> List[str]:
    """The bands in use for this roster size, hardest first."""
    n_models = len(MODELS) if n_models is None else n_models
    return list(BAND_ORDER_WIDE) if n_models >= WIDE_ROSTER else list(OUTCOME_ORDER)


BANDS = active_bands()
BAND_LABELS = band_labels(len(MODELS))


def band_fractions(samples: Sequence[dict], bands: Sequence[str] = None) -> np.ndarray:
    """Fraction of `samples` in each band; zeros for an empty group."""
    bands = BANDS if bands is None else bands
    if not samples:
        return np.zeros(len(bands))
    counts = np.array([sum(1 for s in samples if s['band'] == b) for b in bands],
                      dtype=float)
    total = counts.sum()
    return counts / total if total else counts


def load_samples(path: str = DEFAULT_JSONL) -> List[dict]:
    """Read the JSONL dump and attach the derived fields the figures plot."""
    samples, skipped = [], 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            passed = item['model_passed_rules']
            if any(m not in passed for m in MODELS):
                skipped += 1            # a model never ran on this item
                continue
            n_pass = sum(1 for m in MODELS if passed[m])
            item['family'] = mechanism_family(item['mechanism_en'])
            item['subtype'] = mechanism_subtype(item['mechanism_en'])
            item['n_translators_passed'] = n_pass
            item['n_translators_total'] = len(MODELS)
            item['outcome'] = ('all_pass' if n_pass == len(MODELS)
                               else 'all_fail' if n_pass == 0 else 'some_fail')
            item['band'] = band_of(n_pass, len(MODELS))
            item['n_rules'] = len(item['verification_rules'])
            item['source_len'] = len(item['source'])
            samples.append(item)
    if skipped:
        print(f'ltb_data: {skipped} item(s) skipped - not every model in the roster '
              f'has been run on them', file=sys.stderr)
    return samples


def family_counts(samples: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in samples:
        counts[s['family']] = counts.get(s['family'], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def family_groups(samples: Sequence[dict], min_count: int = 5,
                  other_label: str = 'Other', max_groups: int = None,
                  keep_hardest: int = 0) -> List[str]:
    """Families with at least `min_count` items, largest first, then `Other`.

    `max_groups` caps how many families are named, pooling the remainder into
    `Other`. Bar charts need it once the taxonomy grows past a handful of
    families; the heatmaps, which add a row rather than a column per family, do
    not and leave it unset.
    """
    counts = family_counts(samples)
    eligible = [fam for fam, n in counts.items() if n >= min_count]
    kept = eligible
    if max_groups is not None and len(eligible) > max_groups:
        # Ranking by size alone pools away small families that defeat every
        # model - exactly the ones the benchmark exists to find - so a few slots
        # are reserved for the hardest families, measured by how much of the
        # roster they defeat.
        by_size = eligible[:max(0, max_groups - keep_hardest)]
        if keep_hardest:
            grouped = {}
            for sample in samples:
                grouped.setdefault(sample['family'], []).append(sample)
            hardest = sorted((f for f in eligible if f not in by_size),
                             key=lambda f: np.mean([s['n_translators_passed']
                                                    for s in grouped[f]]))
            chosen = set(by_size) | set(hardest[:keep_hardest])
        else:
            chosen = set(by_size)
        kept = [fam for fam in eligible if fam in chosen]     # keep size order
    if len(kept) < len(counts):
        kept = list(kept) + [other_label]
    return list(kept)


def group_of(sample: dict, groups: Sequence[str], other_label: str = 'Other') -> str:
    return sample['family'] if sample['family'] in groups else other_label


def outcome_fractions(samples: Sequence[dict]) -> np.ndarray:
    """[all_fail, some_fail, all_pass] fractions; zeros for an empty group."""
    if not samples:
        return np.zeros(len(OUTCOME_ORDER))
    counts = np.array([sum(1 for s in samples if s['outcome'] == o)
                       for o in OUTCOME_ORDER], dtype=float)
    return counts / counts.sum()


def wrap_label(text: str, width: int = 16) -> str:
    """Greedy word wrap so long mechanism names fit under a bar or tick."""
    words, lines, cur = text.split(), [], ''
    for word in words:
        candidate = f'{cur} {word}'.strip()
        if len(candidate) > width and cur:
            lines.append(cur)
            cur = word
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return '\n'.join(lines)
