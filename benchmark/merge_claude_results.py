#!/usr/bin/env python3
"""Fold the Claude probe results back into data/ltb_all_models.jsonl.

For every item the merge writes the four translators' outputs, Opus 5's per-rule
verdicts, and a recomputed `candidate_verdict`:

    0 of 4 translators pass every rule -> strict_pass
    1 of 4                             -> marginal_pass
    2 or more                          -> rejected

which generalises the two-model scheme the file already used (none pass /
exactly one passes / two or more pass) without changing its vocabulary, so the
figures in visualization/ keep working.

Curator overrides win over the arithmetic: items withheld on fairness or
confidence grounds stay rejected however the translators score.

    python3 merge_claude_results.py --dry-run     # report, write nothing
    python3 merge_claude_results.py               # rewrite the dataset in place
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_DATA = os.path.join(ROOT, 'data', 'ltb_all_models.jsonl')
DEFAULT_RESULTS = os.path.join(ROOT, 'results', 'claude_translations.jsonl')

JUDGE = 'claude-opus-5'
TRANSLATOR_ORDER = ('claude-sonnet-5', 'claude-haiku-4-5-20251001',
                    'claude-sonnet-4-5', 'claude-opus-4-5')

# Items the curator withholds regardless of how the translators score, with the
# reason kept alongside the item. All three were already rejected in the
# two-model pass for reasons the pass count does not express.
CURATOR_OVERRIDES = {
    'FORM-MEME-01': 'withheld: the literal reading is defensible without the '
                    'Zhihu context, so the item fails the fairness bar',
    'SLOGAN-02': 'withheld: comparable TCM traps are usually solved, so the '
                 'observed failure is low confidence',
    'POL-COL-01': 'withheld: the observed errors are too slight to build a '
                  'submission on',
}

# Field order for the rewritten file: the original schema first, then the
# provenance the re-judging adds.
FIELD_ORDER = ('id', 'date', 'mechanism_zh', 'mechanism_en', 'source_lang',
               'target_lang', 'source', 'perfect_translation',
               'verification_rules', 'translators', 'judge', 'model_outputs',
               'model_passed_rules', 'model_passed_each_rule', 'judge_reasons',
               'n_translators_passed', 'n_translators_total', 'candidate_verdict',
               'previous_verdict',
               'curator_override', 'notes')


def verdict_for(n_passed: int) -> str:
    if n_passed == 0:
        return 'strict_pass'
    if n_passed == 1:
        return 'marginal_pass'
    return 'rejected'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=DEFAULT_DATA)
    parser.add_argument('--results', default=DEFAULT_RESULTS)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-backup', action='store_true')
    parser.add_argument('--force', action='store_true',
                        help='merge even into a pool that carries non-Claude translators')
    args = parser.parse_args(argv)

    records = collections.defaultdict(dict)
    with open(args.results, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            records[row['id']][row['model_key']] = row       # last write wins

    items = [json.loads(line) for line in open(args.data, encoding='utf-8')
             if line.strip()]

    # This script knows the Claude roster only and rewrites each item to
    # FIELD_ORDER, so run against the collected pool it would silently drop
    # every Gemini and GPT column. Stop rather than lose them.
    foreign = sorted({m for item in items for m in (item.get('translators') or [])
                      if m not in TRANSLATOR_ORDER})
    if foreign and not args.force:
        print(f'{args.data} carries {len(foreign)} translators this script does not '
              f'merge ({", ".join(foreign[:3])}, ...); rewriting it here would drop '
              f'them. Use collect_results.py instead, or pass --force.', file=sys.stderr)
        return 1

    incomplete, changed, out = [], 0, []
    verdicts = collections.Counter()
    per_model = collections.Counter()
    per_model_total = collections.Counter()
    moves = collections.Counter()

    for item in items:
        by_model = records.get(item['id'], {})
        missing = [m for m in TRANSLATOR_ORDER if m not in by_model]
        if missing:
            incomplete.append((item['id'], missing))

        models = [m for m in TRANSLATOR_ORDER if m in by_model]
        outputs, passed, each, reasons = {}, {}, {}, {}
        for model in models:
            row = by_model[model]
            outputs[model] = row['translation']
            passed[model] = bool(row['passed_all'])
            each[model] = [bool(v['passed']) for v in row['rule_verdicts']]
            reasons[model] = [v['reason'] for v in row['rule_verdicts']]
            per_model_total[model] += 1
            per_model[model] += bool(row['passed_all'])

        n_passed = sum(1 for m in models if passed[m])
        previous = item.get('candidate_verdict')
        override = CURATOR_OVERRIDES.get(item['id'])
        if missing:
            # An unjudged translator would read as a failure it never earned, so
            # the item keeps its previous verdict until the roster is complete.
            verdict = previous
        elif override:
            verdict = 'rejected'
        else:
            verdict = verdict_for(n_passed)

        merged = dict(item)
        merged.update({
            'translators': models,
            'judge': JUDGE,
            'model_outputs': outputs or item.get('model_outputs', {}),
            'model_passed_rules': passed or item.get('model_passed_rules', {}),
            'model_passed_each_rule': each,
            'judge_reasons': reasons,
            'n_translators_passed': n_passed,
            'n_translators_total': len(models),
            'candidate_verdict': verdict,
            'previous_verdict': previous,
        })
        if override:
            merged['curator_override'] = override
        if missing:
            merged['incomplete_translators'] = missing
        if verdict != previous:
            changed += 1
            moves[(previous, verdict)] += 1
        verdicts[verdict] += 1
        out.append({k: merged[k] for k in FIELD_ORDER if k in merged})

    print(f'items: {len(items)}   re-judged by {JUDGE}')
    if incomplete:
        print(f'INCOMPLETE: {len(incomplete)} items are missing a translator and '
              f'keep their previous verdict, e.g. {incomplete[:3]}')
    print('\nper-translator pass rate (all rules of an item):')
    for model in TRANSLATOR_ORDER:
        total = per_model_total[model]
        if total:
            print(f'  {model:<28} {per_model[model]:>4}/{total}  '
                  f'{100*per_model[model]/total:5.1f}%')
    print('\nverdicts:')
    for verdict in ('strict_pass', 'marginal_pass', 'rejected'):
        print(f'  {verdict:<14} {verdicts[verdict]:>4}')
    print(f'\nverdict changed for {changed} items:')
    for (old, new), count in moves.most_common():
        print(f'  {old} -> {new}: {count}')

    if args.dry_run:
        print('\n(dry run: nothing written)')
        return 0
    if not args.no_backup:
        shutil.copy2(args.data, args.data + '.bak')   # delete once the merge looks right
    with open(args.data, 'w', encoding='utf-8') as handle:
        for row in out:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'\nwrote {args.data}' + ('' if args.no_backup else ' (backup: .bak)'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
