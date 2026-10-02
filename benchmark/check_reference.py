#!/usr/bin/env python3
"""Validity check: does each item's own reference translation pass its own rules?

LTB requires the "perfect" translation to satisfy the verification rules — that
is what demonstrates the item is translatable and the rules are satisfiable. A
rule the reference itself fails is not a hard rule, it is a broken one, and the
submission is invalid however many translators it defeats.

    python3 check_reference.py --data ../data/ltb_validation.jsonl
    python3 check_reference.py --ids TABOO-06,TABOO-03

Judged by the same model and prompt the probe uses, with the reference supplied
as the candidate; the judge is not told it is the reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_claude_probe import (JUDGE, JUDGE_SYSTEM, call_model, judge_prompt,  # noqa: E402
                              parse_verdict, rules_of, with_retries)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
DEFAULT_DATA = os.path.join(ROOT, 'data', 'ltb_validation.jsonl')


def check(item, args, lock, counter):
    verdicts = []
    for index, rule in enumerate(rules_of(item)):
        raw = with_retries(call_model, JUDGE, JUDGE_SYSTEM,
                           # the reference IS the candidate here, and no reference
                           # is shown, so the judge rules on it blind
                           judge_prompt(item, item['perfect_translation'], rule,
                                        reference=False),
                           retries=args.retries, timeout=args.timeout)
        verdicts.append({'index': index, 'rule': rule, **parse_verdict(raw)})
    with lock:
        counter[0] += 1
        print(f'  [{counter[0]}] {item["id"]:<14} '
              + ' '.join('PASS' if v['passed'] else 'FAIL' for v in verdicts), flush=True)
    return item['id'], verdicts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=DEFAULT_DATA)
    parser.add_argument('--ids')
    parser.add_argument('--out')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=180.0)
    args = parser.parse_args(argv)

    items = [json.loads(line) for line in open(args.data, encoding='utf-8') if line.strip()]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(',')}
        items = [i for i in items if i['id'] in wanted]

    print(f'checking {len(items)} references against '
          f'{sum(len(rules_of(i)) for i in items)} rules, judged by {JUDGE}')
    lock, counter = threading.Lock(), [0]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = dict(pool.map(lambda i: check(i, args, lock, counter), items))

    broken = {i: v for i, v in results.items() if not all(x['passed'] for x in v)}
    print(f'\n{len(broken)} of {len(items)} references fail at least one of their own rules')
    for item_id, verdicts in sorted(broken.items()):
        for verdict in verdicts:
            if not verdict['passed']:
                print(f'  {item_id:<14} rule {verdict["index"] + 1}: {verdict["reason"][:96]}')
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(results, handle, ensure_ascii=False, indent=1)
        print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
