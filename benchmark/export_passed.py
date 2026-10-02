#!/usr/bin/env python3
"""Export the candidates that passed curation to a readable Markdown file.

One section per item: the Chinese input, the reference translation, and the
verification rules — the three things an LTB submission is made of. Items are
grouped by mechanism family, hardest family first, so the coverage of the
submittable pool is visible at a glance.

    python3 export_passed.py                      # data/passed_samples.md
    python3 export_passed.py --strict-only        # drop the marginal section
    python3 export_passed.py --out ../elsewhere.md

`strict_pass` means no translator in the roster satisfied every rule of the
item; `marginal_pass` means exactly one did. Both clear LTB's bar of at most two
passing translations, so both are exported, in separate sections.

By default the verdict is the curator's, decided over the Claude roster in
`data/ltb_all_models.jsonl`. `--verdict roster` instead reads it off the file's own
counts, which is what you want when exporting from `data/ltb_all_models.jsonl`,
where a row spans every provider probed:

    python3 export_passed.py --data ../data/ltb_all_models.jsonl --verdict roster \
        --strict-only --out ../data/strict_passes_all_models.md

Under `--verdict roster` the strict section is split by coverage: an item no
Claude model solved is not the same claim as an item no model of any provider
solved, and the file should not blur the two.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_DATA = os.path.join(ROOT, 'data', 'ltb_all_models.jsonl')
DEFAULT_COLLECTED = os.path.join(ROOT, 'data', 'ltb_all_models.jsonl')
DEFAULT_OUT = os.path.join(ROOT, 'data', 'passed_samples.md')

LANG_NAMES = {'zh': 'Chinese', 'en': 'English'}
PROVIDER_LABELS = {'anthropic': 'Claude', 'google': 'Gemini/Gemma', 'openai': 'GPT'}


def provider_of(model: str) -> str:
    for prefix, family in (('claude', 'anthropic'), ('gemini', 'google'),
                           ('gemma', 'google'), ('gpt', 'openai')):
        if model.startswith(prefix):
            return family
    return 'other'


def family_of(item: dict) -> str:
    return item['mechanism_en'].split(':')[0].strip()


def anchor(text: str) -> str:
    """GitHub-style heading anchor."""
    keep = [c.lower() if c.isalnum() else ('-' if c in ' -' else '')
            for c in text]
    return ''.join(keep)


def wider_status(item: dict, collected: dict, full_roster: int) -> str:
    """How the item fared against every translator that has run on it.

    A verdict decided over one provider's models is a weaker claim than one
    decided over all of them, and this line is what keeps the two apart.
    """
    # The data file may be the pool (Claude only) or the multi-provider collection,
    # and either may be the better-informed row, so read the status off whichever
    # covers more translators rather than assuming which file is which.
    view = max((item, collected.get(item['id']) or {}),
               key=lambda r: r.get('n_translators_total') or 0)
    total = view.get('n_translators_total') or 0
    if not total:
        return ''
    passed = view.get('n_translators_passed') or 0
    if passed == 0 and total >= full_roster:
        return f'✅ **Defeats all {total} translators** probed, across all providers.'
    if passed == 0:
        providers = sorted({provider_of(m) for m in (view.get('model_passed_rules') or {})})
        named = '/'.join(PROVIDER_LABELS.get(p, p) for p in providers) or 'one provider'
        return (f'⚠ **{total} of {total} translators defeated**, but only the {named} '
                f'roster has been run here — the other providers are untested on it.')
    solved = [m for m, ok in (view.get('model_passed_rules') or {}).items() if ok]
    return (f'⚠ **{passed} of {total} translators solved it** '
            f'({", ".join(sorted(solved))}).')


def render_item(item: dict, lines: list, collected: dict, full_roster: int) -> None:
    source_lang = LANG_NAMES.get(item['source_lang'], item['source_lang'])
    target_lang = LANG_NAMES.get(item['target_lang'], item['target_lang'])
    lines.append(f"#### `{item['id']}` — {item['mechanism_en']}")
    lines.append('')
    lines.append(f"**Input ({source_lang}).** {item['source']}")
    lines.append('')
    lines.append(f"**Reference translation ({target_lang}).** "
                 f"{item['perfect_translation']}")
    lines.append('')
    rules = item.get('verification_rules') or []
    rules = [rules] if isinstance(rules, str) else rules
    lines.append('**Verification rules.**')
    lines.append('')
    for index, rule in enumerate(rules, 1):
        lines.append(f'{index}. {rule}')
    lines.append('')
    status = wider_status(item, collected, full_roster)
    if status:
        lines.append(status)
        lines.append('')


def split_by_coverage(items: list, full: int) -> tuple:
    """(items probed by the full roster, items probed by fewer)."""
    return ([i for i in items if i.get('n_translators_total', 0) >= full],
            [i for i in items if i.get('n_translators_total', 0) < full])


def render_section(title: str, blurb: str, items: list, lines: list,
                   collected: dict, full_roster: int) -> None:
    by_family = collections.defaultdict(list)
    for item in items:
        by_family[family_of(item)].append(item)
    # Hardest families first: fewest translators passing, then largest.
    order = sorted(by_family,
                   key=lambda f: (sum(i['n_translators_passed'] for i in by_family[f])
                                  / len(by_family[f]), -len(by_family[f])))

    lines.append(f'## {title} ({len(items)})')
    lines.append('')
    lines.append(blurb)
    lines.append('')
    for family in order:
        group = by_family[family]
        lines.append(f'### {family} ({len(group)})')
        lines.append('')
        for item in sorted(group, key=lambda i: i['id']):
            render_item(item, lines, collected, full_roster)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=DEFAULT_DATA)
    parser.add_argument('--collected', default=DEFAULT_COLLECTED,
                        help='multi-provider results, for the per-item wider-roster note')
    parser.add_argument('--verdict', choices=('curator', 'roster'), default='curator',
                        help="'curator' uses candidate_verdict (decided over the Claude "
                             "roster); 'roster' derives it from the file's own "
                             'n_translators_passed, for the multi-provider collection')
    parser.add_argument('--all-providers-only', action='store_true',
                        help='export only items that defeat every translator probed, '
                             'not just the Claude roster')
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--strict-only', action='store_true',
                        help='export only items no translator passed')
    args = parser.parse_args(argv)

    items = [json.loads(line) for line in open(args.data, encoding='utf-8')
             if line.strip()]
    collected = {}
    if os.path.exists(args.collected):
        collected = {json.loads(line)['id']: json.loads(line)
                     for line in open(args.collected, encoding='utf-8') if line.strip()}
    if args.all_providers_only and collected:
        items = [i for i in items
                 if collected.get(i['id'], {}).get('n_translators_passed', 1) == 0]
    if args.verdict == 'roster':
        strict = [i for i in items if i.get('n_translators_passed') == 0]
        marginal = [i for i in items if i.get('n_translators_passed') == 1]
    else:
        strict = [i for i in items if i.get('candidate_verdict') == 'strict_pass']
        marginal = [i for i in items if i.get('candidate_verdict') == 'marginal_pass']
    if args.strict_only:
        marginal = []
    if not strict and not marginal:
        print('No passed items found.', file=sys.stderr)
        return 1

    # The widest roster anything in either file was probed against: an item is only
    # 'defeats every translator' if it was measured against all of them.
    full_roster = max([(i.get('n_translators_total') or 0) for i in items]
                      + [(r.get('n_translators_total') or 0) for r in collected.values()]
                      or [0])
    roster = strict[0].get('translators') or []
    judges = strict[0].get('judges') or {}
    judge = strict[0].get('judge') or ' + '.join(sorted(set(judges.values()))) or 'unrecorded'

    def view_of(item):
        return max((item, collected.get(item['id']) or {}),
                   key=lambda r: r.get('n_translators_total') or 0)

    kept = strict + marginal
    fully = [i for i in kept if (view_of(i).get('n_translators_total') or 0) >= full_roster]
    beats_all = [i for i in fully if (view_of(i).get('n_translators_passed') or 0) == 0]
    if args.verdict == 'roster':
        wider_note = (
            f'The verdict here is the file\'s own: an item is strict when **no '
            f'translator on its row** satisfied every rule. {len(beats_all)} of the '
            f'{len(kept)} entries were measured against the full roster of '
            f'{full_roster}; the rest have only been run against one provider so far '
            f'and are kept in a separate section rather than counted as equivalent.')
    else:
        solved_elsewhere = [i for i in fully if i not in beats_all]
        wider_note = (
            f'**The verdict is decided over the Claude roster alone.** Of the '
            f'{len(kept)} entries here, {len(beats_all)} are confirmed to defeat every '
            f'translator probed ({full_roster}, across Anthropic, Google and OpenAI), '
            f'{len(solved_elsewhere)} were solved by at least one non-Claude model, and '
            f'{len(kept) - len(fully)} have not yet been run beyond Claude.'
            if collected else
            'The multi-provider results file was not found, so no wider-roster note is '
            'attached to the entries.')

    lines = [
        '# Passed candidates — Last Translation Benchmark (Chinese → English)',
        '',
        f'{len(strict) + len(marginal)} of {len(items)} candidates cleared the bar: '
        f'**{len(strict)} strict** and **{len(marginal)} marginal**. Each '
        'entry below is what an LTB submission consists of — the input, a reference '
        'translation showing the item is translatable, and the verification rules an '
        'AI judge applies.',
        '',
        f'Translators probed for the verdict: {", ".join(f"`{m}`" for m in roster) or "unrecorded"}. '
        f'Judge: `{judge}`, ruling on each rule separately; an item counts as passed '
        'for a model only when its translation satisfies *every* rule.',
        '',
        f'{wider_note} Each entry carries a line saying how it fared against every '
        'translator that has run on it, because an item that only one provider fails is '
        'weak evidence for LTB reviewers, who test against ChatGPT and Google Translate.',
        '',
        'Generated by `benchmark/export_passed.py` from '
        f'`{os.path.relpath(args.data, ROOT)}` — edit the dataset, not this file.',
        '',
        '---',
        '',
    ]

    if args.verdict == 'roster':
        full = max((i.get('n_translators_total', 0) for i in items), default=0)
        complete, partial = split_by_coverage(strict, full)
        render_section(
            f'Strict passes — every one of {full} translators defeated',
            'No translator of any provider satisfied every rule of these items. This '
            'is the strongest evidence the pool holds.',
            complete, lines, collected, full_roster)
        if partial:
            lines.append('---')
            lines.append('')
            render_section(
                'Strict passes — Claude roster only',
                'No Claude translator satisfied every rule of these items, but the '
                'Gemini and GPT translators have not been run on them yet, so whether '
                'another provider solves them is simply unknown.',
                partial, lines, collected, full_roster)
    else:
        render_section(
            'Strict passes', 'No Claude translator satisfied every rule of these items. '
            'Check each entry\'s wider-roster line before treating it as submittable.',
            strict, lines, collected, full_roster)
    if marginal:
        lines.append('---')
        lines.append('')
        render_section(
            'Marginal passes', 'Exactly one Claude translator satisfied every rule. '
            'Still within LTB\'s bar of at most two passing translations, but weaker '
            'evidence than a strict pass.', marginal, lines, collected, full_roster)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines).rstrip() + '\n')
    print(f'wrote {args.out}  ({len(strict)} strict, {len(marginal)} marginal, '
          f'{len(set(family_of(i) for i in strict + marginal))} mechanism families)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
