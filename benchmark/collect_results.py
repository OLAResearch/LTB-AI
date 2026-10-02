#!/usr/bin/env python3
"""Collect every probe's results into one dataset-shaped file for plotting.

Two probes write per-(item, model) records in the same shape but to different
files, with different judges:

    results/claude_translations.jsonl   Claude translators, judged by Opus 5
    results/translations.jsonl          Gemini/Gemma/GPT translators, judged by
                                        Gemini 3.7 Flash

This folds both into `data/ltb_all_models.jsonl`, one row per item, carrying the
item's own fields (source, rules, reference translation, curator verdict) plus
every model that has been run on it. The row keeps the flat
`model_outputs` / `model_passed_rules` / `model_passed_each_rule` keys the
figures already read, so `visualization/` can point straight at it.

    python3 collect_results.py --report          # coverage only, write nothing
    python3 collect_results.py                   # write data/ltb_all_models.jsonl
    python3 collect_results.py --require-complete # only items every model has run

**Judges differ between the two families**, so a Claude column and a Gemini
column are not strictly comparable: each row records `judges` per model, and
Gemini 3.7 Flash judges its own translations. Treat cross-family gaps as
indicative, not measured.
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
DEFAULT_CLAUDE = os.path.join(ROOT, 'results', 'claude_translations.jsonl')
DEFAULT_OTHERS = os.path.join(ROOT, 'results', 'translations.jsonl')
DEFAULT_OUT = os.path.join(ROOT, 'data', 'ltb_all_models.jsonl')

# Item fields copied through from the pool, in this order.
ITEM_FIELDS = ('id', 'date', 'mechanism_zh', 'mechanism_en', 'source_lang',
               'target_lang', 'source', 'perfect_translation',
               'verification_rules', 'candidate_verdict', 'curator_override',
               'notes')

PROVIDER_PREFIXES = (('claude', 'anthropic'), ('gemini', 'google'),
                     ('gemma', 'google'), ('gpt', 'openai'), ('o1', 'openai'))
FAMILY_ORDER = ('anthropic', 'google', 'openai', 'other')


def provider_of(record: dict) -> str:
    """Which family a model belongs to, from the record or its id."""
    provider = (record.get('provider') or '').strip().lower()
    if provider in ('openai', 'google', 'anthropic'):
        return provider
    key = (record.get('model_key') or '').lower()
    for prefix, family in PROVIDER_PREFIXES:
        if key.startswith(prefix):
            return family
    return 'other'


def load_records(path: str, label: str) -> dict:
    """{item id: {model key: record}}, last write wins, errored records dropped."""
    by_item: dict = collections.defaultdict(dict)
    if not os.path.exists(path):
        print(f'  {label:<8} {path} not found - skipped', file=sys.stderr)
        return by_item
    kept = dropped = 0
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue                                   # torn final line
            if row.get('error') or not row.get('translation'):
                dropped += 1
                continue
            by_item[row['id']][row['model_key']] = row
            kept += 1
    models = {m for models in by_item.values() for m in models}
    print(f'  {label:<8} {kept:>6} records, {len(by_item):>4} items, '
          f'{len(models):>2} models'
          + (f'  ({dropped} errored records skipped)' if dropped else ''))
    return by_item


def rules_of(item) -> list:
    rules = item.get('verification_rules') or []
    return [rules] if isinstance(rules, str) else list(rules)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=DEFAULT_DATA, help='the item pool')
    parser.add_argument('--claude-results', default=DEFAULT_CLAUDE)
    parser.add_argument('--other-results', default=DEFAULT_OTHERS)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--models', help='comma-separated allow-list of model keys')
    parser.add_argument('--require-complete', action='store_true',
                        help='emit only items every collected model has been run on')
    parser.add_argument('--report', action='store_true', help='report coverage, write nothing')
    parser.add_argument('--force', action='store_true',
                        help='write even when the result is smaller than the existing --out')
    args = parser.parse_args(argv)

    print('sources:')
    records = collections.defaultdict(dict)
    for path, label in ((args.claude_results, 'claude'), (args.other_results, 'others')):
        for item_id, by_model in load_records(path, label).items():
            records[item_id].update(by_model)

    items = [json.loads(line) for line in open(args.data, encoding='utf-8') if line.strip()]
    allow = {m.strip() for m in args.models.split(',')} if args.models else None

    # Model order: family first, then alphabetical inside it, so columns are stable.
    all_models = {m for by_model in records.values() for m in by_model}
    if allow:
        all_models &= allow
    family_of = {}
    for by_model in records.values():
        for key, row in by_model.items():
            family_of.setdefault(key, provider_of(row))
    order = sorted(all_models, key=lambda m: (FAMILY_ORDER.index(family_of.get(m, 'other')), m))

    out, coverage, incomplete = [], collections.Counter(), []
    for item in items:
        by_model = {m: r for m, r in records.get(item['id'], {}).items() if m in all_models}
        if not by_model:
            continue
        missing = [m for m in order if m not in by_model]
        if missing:
            incomplete.append((item['id'], len(missing)))
            if args.require_complete:
                continue

        models = [m for m in order if m in by_model]
        n_rules = len(rules_of(item))
        row = {k: item[k] for k in ITEM_FIELDS if k in item}
        row['n_rules'] = n_rules
        row['translators'] = models
        row['providers'] = {m: family_of.get(m, 'other') for m in models}
        row['judges'] = {m: by_model[m].get('judge', '') for m in models}
        row['model_outputs'] = {m: by_model[m]['translation'] for m in models}
        row['model_passed_rules'] = {m: bool(by_model[m]['passed_all']) for m in models}
        row['model_passed_each_rule'] = {
            m: [bool(v['passed']) for v in by_model[m].get('rule_verdicts', [])] for m in models}
        row['judge_reasons'] = {
            m: [v.get('reason', '') for v in by_model[m].get('rule_verdicts', [])] for m in models}
        row['n_translators_passed'] = sum(row['model_passed_rules'].values())
        row['n_translators_total'] = len(models)
        # The pool's verdict is decided over the Claude roster alone, so the
        # Claude-only counts travel alongside the roster-wide ones: the same key
        # meaning different denominators in two files is a trap.
        claude = [m for m in models if family_of.get(m) == 'anthropic']
        row['n_claude_passed'] = sum(row['model_passed_rules'][m] for m in claude)
        row['n_claude_total'] = len(claude)
        by_family = collections.defaultdict(lambda: [0, 0])
        for m in models:
            slot = by_family[family_of.get(m, 'other')]
            slot[0] += int(row['model_passed_rules'][m])
            slot[1] += 1
        row['family_pass'] = {f: {'passed': p, 'total': t} for f, (p, t) in by_family.items()}
        if missing:
            row['missing_models'] = missing
        out.append(row)
        for m in models:
            coverage[m] += 1

    print(f'\nitems in pool: {len(items)}   with results: {len(out)}')
    print(f'models: {len(order)}  ({", ".join(f"{f}:{sum(1 for m in order if family_of.get(m)==f)}" for f in FAMILY_ORDER if any(family_of.get(m)==f for m in order))})')
    print('\nper-model coverage and item pass rate:')
    for model in order:
        rows = [r for r in out if model in r['model_passed_rules']]
        passed = sum(r['model_passed_rules'][model] for r in rows)
        judges = {r['judges'][model] for r in rows}
        print(f'  {family_of.get(model,"other"):<10} {model:<26} {len(rows):>4} items  '
              f'{100*passed/len(rows) if rows else 0:5.1f}% pass   judge: {"/".join(sorted(judges))}')
    if incomplete:
        worst = collections.Counter(n for _, n in incomplete)
        print(f'\n{len(incomplete)} items are missing at least one model '
              f'(by count: {dict(sorted(worst.items()))})'
              + ('  - excluded by --require-complete' if args.require_complete else ''))

    if args.report:
        print('\n(report only: nothing written)')
        return 0
    # --data and --out both default to the collected pool, so an empty or
    # half-written results/ would otherwise overwrite it with a shorter file.
    existing = 0
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8') as handle:
            existing = sum(1 for line in handle if line.strip())
    if existing and len(out) < existing and not args.force:
        print(f'\nrefusing to overwrite {args.out}: it holds {existing} items and this '
              f'run produced {len(out)}. Check results/, or pass --force.', file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        for row in out:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'\nwrote {args.out}  ({len(out)} items)')
    print('plot it with:  LTB_JSONL=%s python3 plot_all.py' % os.path.relpath(args.out, ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
