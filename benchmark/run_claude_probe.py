#!/usr/bin/env python3
"""Probe the Claude roster on the LTB Chinese-to-English pool.

Translators are every Claude model this machine can call except Opus 5, which
is reserved as the curator/judge. Each (item, model) pair gets one translation;
each verification rule of the item is then judged separately by Opus 5, exactly
as in `run_translators.py` (rules are written to be atomic, so judging them
together invites the judge to average over them).

Model calls go through the local `claude -p` CLI rather than an SDK, because
this machine authenticates through Claude Code and has no ANTHROPIC_API_KEY.
`--system-prompt` replaces the agent system prompt entirely, so each call is a
plain single-turn model call with no tools and no session state.

    python3 run_claude_probe.py --dry-run --limit 3     # no calls, no cost
    python3 run_claude_probe.py --limit 5               # small live slice
    python3 run_claude_probe.py --concurrency 12        # full pool

Writes results/claude_translations.jsonl (one line per item x model, resumable)
and results/claude_checkpoints.jsonl (one line per completed call).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_DATA = os.path.join(ROOT, 'data', 'ltb_all_models.jsonl')
DEFAULT_OUT = os.path.join(ROOT, 'results', 'claude_translations.jsonl')
DEFAULT_CHECKPOINT = os.path.join(ROOT, 'results', 'claude_checkpoints.jsonl')

CLAUDE = os.environ.get('CLAUDE_CODE_EXECPATH') or 'claude'

SOURCE_LANG, TARGET_LANG = 'zh', 'en'
SOURCE_LANG_NAME, TARGET_LANG_NAME = 'Chinese', 'English'

# Every Claude model callable here except Opus 5 (the judge). Fable 5 is listed
# but skipped by default: it needs usage credits this account does not have.
TRANSLATORS = (
    ('Claude Sonnet 5',   'claude-sonnet-5'),
    ('Claude Haiku 4.5',  'claude-haiku-4-5-20251001'),
    ('Claude Sonnet 4.5', 'claude-sonnet-4-5'),
    ('Claude Opus 4.5',   'claude-opus-4-5'),
)
JUDGE = 'claude-opus-5'
LABELS = dict((model, label) for label, model in TRANSLATORS)

TRANSLATOR_SYSTEM = (
    f'You are a professional translator. Translate the user text faithfully from '
    f'{SOURCE_LANG_NAME} into {TARGET_LANG_NAME}. Preserve meaning, register, tone '
    f'and any implied social relations. Output only the translation: no commentary, '
    f'no quotation marks around it, no romanisation, no notes.'
)

JUDGE_SYSTEM = (
    'You are a strict evaluator for a machine-translation benchmark. You receive a '
    'source text, a reference translation known to be correct, a candidate '
    'translation, and exactly ONE verification rule. Decide whether the candidate '
    'translation satisfies that rule.\n'
    'Rules:\n'
    '- Judge only the given rule. Ignore other flaws, style preferences and '
    'infelicities that the rule does not mention.\n'
    '- Any reasonable phrasing that satisfies the rule passes; the rule is about '
    'meaning, not wording.\n'
    '- The reference shows one correct way to satisfy the rule, not the only way. '
    'Do not fail a candidate merely for differing from it.\n'
    '- If the rule is not satisfied, or the candidate omits what the rule tests, '
    'the verdict is fail.\n'
    'Answer with JSON only, no prose and no code fence: '
    '{"verdict": "pass" | "fail", "reason": "<25 words>"}'
)

# CLI banners that are not model output.
NOISE = re.compile(r'^(⚠|Warning:|\[DEBUG\])')


def sha(*parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode('utf-8'))
        digest.update(b'\x1f')
    return digest.hexdigest()[:16]


# ----------------------------------------------------------------- model calls

class CallError(RuntimeError):
    pass


class FatalError(RuntimeError):
    """Auth, credit or unknown-model problems: retrying cannot help."""


FATAL_MARKERS = ('requires usage credits', 'invalid api key', 'not authenticated',
                 'unknown model', 'credit balance')


def call_model(model: str, system: str, user: str, *, timeout: float = 180.0,
               dry_run: bool = False) -> str:
    """One single-turn model call through the headless CLI."""
    if dry_run:
        digest = sha(model, user)
        if 'verification rule' in system.lower():
            verdict = 'fail' if int(digest[:2], 16) % 3 == 0 else 'pass'
            return json.dumps({'verdict': verdict, 'reason': f'dry-run {digest[:8]}'})
        return f'[dry-run translation by {model} - {digest[:12]}]'

    proc = subprocess.run(
        [CLAUDE, '-p', '--model', model, '--system-prompt', system,
         '--disable-slash-commands', '--strict-mcp-config', '--no-session-persistence'],
        input=user, capture_output=True, text=True, timeout=timeout)
    out = (proc.stdout or '').strip()
    err = (proc.stderr or '').strip()
    low = (out + ' ' + err).lower()
    if any(marker in low for marker in FATAL_MARKERS):
        raise FatalError(f'{model}: {(out or err)[:200]}')
    if proc.returncode != 0:
        raise CallError(f'{model} exited {proc.returncode}: {(err or out)[:200]}')
    lines = [ln for ln in out.splitlines() if not NOISE.match(ln.strip())]
    text = '\n'.join(lines).strip()
    if not text:
        raise CallError(f'{model} returned nothing')
    return text


def with_retries(fn, *args, retries: int = 3, base_delay: float = 2.0, **kwargs):
    last = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except FatalError:
            raise
        except Exception as exc:
            last = exc
            if attempt == retries:
                break
            time.sleep(base_delay * (2 ** attempt) * (0.7 + 0.6 * random.random()))
    raise CallError(f'failed after {retries + 1} attempts: {last}') from last


def parse_verdict(raw: str) -> dict:
    """Pull {'verdict', 'reason'} out of a judge reply, fenced or bare."""
    text = raw.strip()
    text = re.sub(r'^```(?:json)?|```$', '', text, flags=re.M).strip()
    match = re.search(r'\{.*\}', text, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            verdict = str(data.get('verdict', '')).strip().lower()
            if verdict in ('pass', 'fail'):
                return {'passed': verdict == 'pass',
                        'reason': str(data.get('reason', ''))[:300]}
        except json.JSONDecodeError:
            pass
    low = text.lower()
    if 'fail' in low and 'pass' not in low:
        return {'passed': False, 'reason': text[:300]}
    if 'pass' in low and 'fail' not in low:
        return {'passed': True, 'reason': text[:300]}
    raise CallError(f'unparseable judge reply: {text[:200]}')


# ---------------------------------------------------------------- prompt bodies

def translate_prompt(item: dict) -> str:
    return item['source']


def judge_prompt(item: dict, translation: str, rule: str, reference: bool) -> str:
    parts = [f'SOURCE ({SOURCE_LANG_NAME}):\n{item["source"]}']
    if reference:
        parts.append(f'REFERENCE TRANSLATION (correct):\n{item["perfect_translation"]}')
    parts.append(f'CANDIDATE TRANSLATION ({TARGET_LANG_NAME}):\n{translation}')
    parts.append(f'VERIFICATION RULE:\n{rule}')
    parts.append('Does the candidate translation satisfy the rule? JSON only.')
    return '\n\n'.join(parts)


def rules_of(item) -> list:
    rules = item.get('verification_rules') or []
    return [rules] if isinstance(rules, str) else list(rules)


# ------------------------------------------------------------------ checkpoint

class Checkpoint:
    """Append-only log of completed calls, so a rerun never pays twice."""

    def __init__(self, path, enabled=True):
        self.path = path
        self.enabled = enabled
        self.entries = {}
        self._lock = threading.Lock()
        if enabled and os.path.exists(path):
            with open(path, encoding='utf-8') as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue                      # torn final line
                    self.entries[row['key']] = row['value']

    def get(self, key):
        return self.entries.get(key)

    def put(self, key, value):
        with self._lock:
            self.entries[key] = value
            if not self.enabled:
                return
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps({'key': key, 'value': value},
                                        ensure_ascii=False) + '\n')
                handle.flush()


# ------------------------------------------------------------------- the probe

def run_task(item, model, args, checkpoint, existing_translation, counters):
    """Translate one item with one model, then judge every rule of it."""
    started = time.time()
    record = {'id': item['id'], 'model_key': model, 'label': LABELS.get(model, model),
              'mechanism_en': item.get('mechanism_en', ''), 'judge': JUDGE,
              'api_calls': 0, 'cached': 0, 'reused_translation': False}

    src_hash = sha(item['source'])
    tkey = f'translation|{item["id"]}|{model}|{src_hash}'
    translation = None
    if existing_translation and args.reuse_existing:
        translation, record['reused_translation'] = existing_translation, True
    elif checkpoint.get(tkey) is not None:
        translation, record['cached'] = checkpoint.get(tkey), record['cached'] + 1
    else:
        translation = with_retries(call_model, model, TRANSLATOR_SYSTEM,
                                   translate_prompt(item), retries=args.retries,
                                   timeout=args.timeout, dry_run=args.dry_run)
        checkpoint.put(tkey, translation)
        record['api_calls'] += 1
    record['translation'] = translation

    verdicts = []
    for index, rule in enumerate(rules_of(item)):
        vkey = ('verdict|{}|{}|{}|{}|{}|{}|{}'.format(
            item['id'], model, JUDGE, index, sha(rule), sha(translation),
            'ref' if not args.no_reference else 'noref'))
        cached = checkpoint.get(vkey)
        if cached is not None:
            verdict, record['cached'] = cached, record['cached'] + 1
        else:
            raw = with_retries(call_model, JUDGE, JUDGE_SYSTEM,
                               judge_prompt(item, translation, rule,
                                            not args.no_reference),
                               retries=args.retries, timeout=args.timeout,
                               dry_run=args.dry_run)
            verdict = parse_verdict(raw)
            checkpoint.put(vkey, verdict)
            record['api_calls'] += 1
        verdicts.append({'index': index, 'rule': rule, **verdict})

    record['rule_verdicts'] = verdicts
    record['n_rules'] = len(verdicts)
    record['n_rules_passed'] = sum(1 for v in verdicts if v['passed'])
    record['passed_all'] = bool(verdicts) and all(v['passed'] for v in verdicts)
    record['latency_s'] = round(time.time() - started, 2)
    with counters['lock']:
        counters['calls'] += record['api_calls']
        counters['cached'] += record['cached']
    return record


def load_items(path, args):
    items = [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]
    items = [i for i in items
             if i.get('source_lang') == SOURCE_LANG and i.get('target_lang') == TARGET_LANG]
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(',') if i.strip()}
        items = [i for i in items if i['id'] in wanted]
    if args.mechanism:
        needle = args.mechanism.lower()
        items = [i for i in items if needle in i.get('mechanism_en', '').lower()]
    if args.shuffle:
        random.Random(args.seed).shuffle(items)
    if args.limit:
        items = items[:args.limit]
    return items


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=DEFAULT_DATA)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--model', action='append', default=[],
                        help='translator model id; repeatable (default: whole roster)')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--ids')
    parser.add_argument('--mechanism')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=180.0)
    parser.add_argument('--no-reference', action='store_true',
                        help='hide the perfect translation from the judge')
    parser.add_argument('--no-reuse-existing', dest='reuse_existing',
                        action='store_false',
                        help='re-translate even where the dataset already has an output')
    parser.add_argument('--max-calls', type=int, default=2500)
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--list-models', action='store_true')
    args = parser.parse_args(argv)

    if args.list_models:
        for label, model in TRANSLATORS:
            print(f'  translator  {model:<28} {label}')
        print(f'  judge       {JUDGE:<28} Claude Opus 5')
        return 0

    models = args.model or [m for _, m in TRANSLATORS]
    items = load_items(args.data, args)
    if not items:
        print('No matching items.', file=sys.stderr)
        return 1

    if args.overwrite:
        for path in (args.out, args.checkpoint):
            if os.path.exists(path):
                os.remove(path)

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8') as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((row['id'], row['model_key']))

    checkpoint = Checkpoint(args.checkpoint, enabled=not args.no_checkpoint)

    tasks = [(item, model) for item in items for model in models
             if (item['id'], model) not in done]
    n_translations = sum(1 for item, model in tasks
                         if not (args.reuse_existing
                                 and (item.get('model_outputs') or {}).get(model)))
    n_judgements = sum(len(rules_of(item)) for item, _ in tasks)
    estimate = n_translations + n_judgements
    print(f'{len(items)} items x {len(models)} models = {len(tasks)} tasks '
          f'({len(done)} already done)')
    print(f'estimated calls: {n_translations} translations + {n_judgements} judgements '
          f'= {estimate} (checkpoint may cover some)')
    if estimate > args.max_calls and not args.yes and not args.dry_run:
        print(f'Above --max-calls ({args.max_calls}); pass --yes or raise it.',
              file=sys.stderr)
        return 2
    if not tasks:
        print('Nothing to do.')
        return 0

    counters = {'calls': 0, 'cached': 0, 'lock': threading.Lock()}
    write_lock = threading.Lock()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    completed, failed, started = 0, 0, time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for item, model in tasks:
            existing = (item.get('model_outputs') or {}).get(model)
            futures[pool.submit(run_task, item, model, args, checkpoint,
                                existing, counters)] = (item['id'], model)
        for future in as_completed(futures):
            item_id, model = futures[future]
            try:
                record = future.result()
            except FatalError as exc:
                print(f'FATAL {item_id} {model}: {exc}', file=sys.stderr)
                failed += 1
                continue
            except Exception as exc:
                print(f'ERROR {item_id} {model}: {exc}', file=sys.stderr)
                failed += 1
                continue
            completed += 1
            with write_lock:
                with open(args.out, 'a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                    handle.flush()
            if completed % 25 == 0 or completed == len(tasks):
                rate = completed / max(time.time() - started, 1e-9)
                print(f'  {completed}/{len(tasks)} tasks  '
                      f'{counters["calls"]} calls  {rate*60:.0f}/min', flush=True)

    print(f'\ndone: {completed} ok, {failed} failed, '
          f'{counters["calls"]} API calls, {counters["cached"]} reused from checkpoint, '
          f'{time.time() - started:.0f}s')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
