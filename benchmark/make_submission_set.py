#!/usr/bin/env python3
"""Build the submission set: the items that defeat every translator probed.

One row per candidate, carrying everything needed to submit it and to judge the
submission afterwards — the input, the reference translation, the verification
rules, every translator's output, and every per-rule verdict — plus an empty
`portal` block to record what the official LTB portal says when it comes back.

    python3 make_submission_set.py                  # data/ltb_validation.jsonl
    python3 make_submission_set.py --min-models 10  # a laxer coverage bar
    python3 make_submission_set.py --refresh        # rebuild, keeping portal results

`--refresh` is the one to use once results are coming in: it rewrites the
evidence from the current data while preserving any `portal` block already
recorded, so re-running never loses a reviewer's verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_SOURCE = os.path.join(ROOT, 'data', 'ltb_all_models.jsonl')
DEFAULT_OUT = os.path.join(ROOT, 'data', 'ltb_validation.jsonl')

# Recorded per item once the portal replies. `status` is the only field the
# workflow depends on; the rest is whatever the reviewer said.
EMPTY_PORTAL = {
    'status': 'pending',        # pending | accepted | rejected | revise
    'submitted_on': None,
    'checked_on': None,
    'reviewer_comments': None,
    'failed_rules': None,       # which rules a reviewer disputed, if any
    'notes': None,
}

FIELD_ORDER = (
    'submission_index', 'id', 'date', 'mechanism_zh', 'mechanism_en',
    'source_lang', 'target_lang',
    'source', 'perfect_translation', 'verification_rules', 'n_rules',
    'translators', 'providers', 'judges',
    'model_outputs', 'model_passed_rules', 'model_passed_each_rule',
    'judge_reasons', 'n_translators_passed', 'n_translators_total',
    'curator_notes', 'portal',
)


def load(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default=DEFAULT_SOURCE,
                        help='multi-provider results to select from')
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--min-models', type=int, default=0,
                        help='minimum translators an item must have been probed '
                             'against (default: the widest roster in the file)')
    parser.add_argument('--refresh', action='store_true',
                        help='rebuild the evidence but keep any portal block already '
                             'recorded in --out')
    args = parser.parse_args(argv)

    rows = load(args.source)
    full = args.min_models or max((r.get('n_translators_total') or 0) for r in rows)
    selected = [r for r in rows
                if r.get('n_translators_passed') == 0
                and (r.get('n_translators_total') or 0) >= full]
    selected.sort(key=lambda r: (r['mechanism_en'].split(':')[0], r['id']))

    existing = {}
    if os.path.exists(args.out):
        existing = {r['id']: r.get('portal') or {} for r in load(args.out)}
        if not args.refresh and existing:
            print(f'{args.out} already exists with {len(existing)} rows; pass '
                  f'--refresh to rebuild while keeping recorded portal results.',
                  file=sys.stderr)
            return 2

    out = []
    for index, row in enumerate(selected, 1):
        entry = {
            'submission_index': index,
            **{k: row[k] for k in (
                'id', 'date', 'mechanism_zh', 'mechanism_en', 'source_lang',
                'target_lang', 'source', 'perfect_translation',
                'verification_rules', 'n_rules', 'translators', 'providers',
                'judges', 'model_outputs', 'model_passed_rules',
                'model_passed_each_rule', 'judge_reasons',
                'n_translators_passed', 'n_translators_total') if k in row},
            'curator_notes': row.get('notes'),
            'portal': dict(EMPTY_PORTAL, **existing.get(row['id'], {})),
        }
        out.append({k: entry[k] for k in FIELD_ORDER if k in entry})

    with open(args.out, 'w', encoding='utf-8') as handle:
        for entry in out:
            handle.write(json.dumps(entry, ensure_ascii=False) + '\n')

    kept = sum(1 for e in out if e['portal'].get('status') != 'pending')
    print(f'wrote {args.out}  ({len(out)} items, every one defeating all {full} '
          f'translators)')
    if kept:
        print(f'  preserved {kept} recorded portal result(s)')
    print(f'  rules: {sum(e["n_rules"] for e in out)} | '
          f'translations carried: {sum(len(e["model_outputs"]) for e in out)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
