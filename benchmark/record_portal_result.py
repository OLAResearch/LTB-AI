#!/usr/bin/env python3
"""Record an official LTB portal check against a submission.

The portal reports, per translator, its translation followed by one ✓/✗ per
verification rule. Paste that block on stdin; this parses it, stores it on the
item's `portal` object in data/ltb_validation.jsonl, and reports how the portal's
roster compares with our own probe.

    pbpaste | python3 record_portal_result.py --id DEFECT-02
    python3 record_portal_result.py --id DEFECT-02 --paste result.txt \
        --note 'rule 1 considered unnecessary'

An item passes the portal's own bar when at most two of its translators pass
every rule, so the status is derived from the parsed table rather than asserted:
`passed` (0-2 pass), `failed` (3+ pass). `--status` overrides it when a reviewer
says something the table does not.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_FILE = os.path.join(ROOT, 'data', 'ltb_validation.jsonl')

TICKS = {'✓': True, '✔': True, 'v': True, 'y': True, 'pass': True,
         '✗': False, '✘': False, 'x': False, 'n': False, 'fail': False}
# LTB accepts a submission when at most this many automatic translations pass.
MAX_PASSES = 2


def parse_block(text: str, n_rules: int) -> list:
    """[(translator, translation, [rule verdicts])] from the pasted table.

    The shape is a translator name on its own line, the translation on the next
    (possibly wrapped over several), then exactly `n_rules` tick lines.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.lower().startswith('note:')]
    entries, index = [], 0
    while index < len(lines):
        name = lines[index]
        index += 1
        body, verdicts = [], []
        while index < len(lines) and lines[index].lower() not in TICKS:
            body.append(lines[index])
            index += 1
        while index < len(lines) and lines[index].lower() in TICKS and len(verdicts) < n_rules:
            verdicts.append(TICKS[lines[index].lower()])
            index += 1
        if not body:
            raise SystemExit(f'no translation found for {name!r}')
        if len(verdicts) > n_rules:
            raise SystemExit(f'{name!r}: {len(verdicts)} verdicts for {n_rules} rules')
        # A paste can be cut off mid-entry. Record the gap as unknown rather than
        # inventing a verdict: an unrecorded verdict is not a failure.
        verdicts += [None] * (n_rules - len(verdicts))
        entries.append((name, ' '.join(body), verdicts))
    if not entries:
        raise SystemExit('nothing parsed from the pasted block')
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--id', required=True, help='item id, e.g. DEFECT-02')
    parser.add_argument('--file', default=DEFAULT_FILE)
    parser.add_argument('--paste', help='file holding the pasted block (default: stdin)')
    parser.add_argument('--status', choices=('passed', 'failed', 'accepted',
                                             'rejected', 'revise', 'pending',
                                             'disputed'),
                        help="'disputed' parks a result whose evidence looks wrong, "
                             'so a doubtful verdict is never banked as settled')
    parser.add_argument('--note', help='curator note, e.g. a rule to reconsider')
    parser.add_argument('--comments', help='reviewer comments, verbatim')
    parser.add_argument('--date', default=dt.date.today().isoformat())
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in open(args.file, encoding='utf-8') if line.strip()]
    target = next((r for r in rows if r['id'] == args.id), None)
    if target is None:
        raise SystemExit(f'{args.id} is not in {args.file}')

    text = open(args.paste, encoding='utf-8').read() if args.paste else sys.stdin.read()
    entries = parse_block(text, target['n_rules'])

    passed_all = [name for name, _, verdicts in entries
                  if all(v is True for v in verdicts)]
    # A translator output identical to the reference is not evidence about that
    # translator: it means the reference leaked into the run, or into the paste.
    reference = (target.get('perfect_translation') or '').strip()
    echoes = [name for name, translation, _ in entries
              if reference and translation.strip() == reference]
    incomplete = [name for name, _, verdicts in entries if any(v is None for v in verdicts)]
    # With verdicts missing, the pass count can only rise, so the status is safe
    # to derive only when even the worst case stays within the bar.
    worst_case = len(passed_all) + len(incomplete)
    if args.status:
        status = args.status
    elif worst_case <= MAX_PASSES:
        status = 'passed'
    elif len(passed_all) > MAX_PASSES:
        status = 'failed'
    else:
        status = 'incomplete'
    target['portal'] = dict(target.get('portal') or {}, **{
        'status': status,
        'checked_on': args.date,
        'translators': [name for name, _, _ in entries],
        'model_outputs': {name: translation for name, translation, _ in entries},
        'rule_results': {name: verdicts for name, _, verdicts in entries},
        'n_translators_passed': len(passed_all),
        'n_translators_total': len(entries),
        'passed_translators': passed_all,
        'missing_verdicts': incomplete or None,
        'reference_echoes': echoes or None,
        'reviewer_comments': args.comments or (target.get('portal') or {}).get('reviewer_comments'),
        'notes': args.note or (target.get('portal') or {}).get('notes'),
    })

    with open(args.file, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    print(f'{args.id}: {status} — {len(passed_all)} of {len(entries)} portal '
          f'translators passed every rule'
          + (f' ({", ".join(passed_all)})' if passed_all else ''))
    for index in range(target['n_rules']):
        failed = sum(1 for _, _, v in entries if v[index] is False)
        unknown = sum(1 for _, _, v in entries if v[index] is None)
        print(f'  rule {index + 1}: failed by {failed}/{len(entries)}'
              + (f' ({unknown} not recorded)' if unknown else ''))
    if incomplete:
        print(f'  WARNING: no verdict recorded for {", ".join(incomplete)} — '
              f'paste the missing line(s) and re-run to complete the record')
    if echoes:
        print(f'  WARNING: {", ".join(echoes)} returned our reference translation '
              f'verbatim — that is not evidence about those systems')
    ours = target.get('n_translators_passed')
    if ours is not None:
        print(f'  our probe: {ours} of {target["n_translators_total"]} passed')
    done = sum(1 for r in rows if (r.get('portal') or {}).get('status') not in (None, 'pending'))
    print(f'  recorded: {done}/{len(rows)} submissions')
    return 0


if __name__ == '__main__':
    sys.exit(main())
