#!/usr/bin/env python3
"""Probe Gemini, Gemma and GPT translators on the LTB Chinese-to-English pool.

Chinese -> English only: items with any other language pair are skipped.

For every (item, model) pair the script
  1. asks the model to translate the item's Chinese source into English, then
  2. asks a judge model to rule pass/fail on EACH verification rule of the item
     separately (`verification_rules` is a list, and items commonly carry more
     than one), and
  3. records the translation, one verdict per rule, and `passed_all` -- true
     only when every rule of that item passed, the definition the dataset uses.

API keys come from the environment: OPENAI_API_KEY and GOOGLE_API_KEY.

    export OPENAI_API_KEY=sk-...
    export GOOGLE_API_KEY=...

    # nothing is called, no key needed -- checks prompts, resume, aggregation
    python3 run_translators.py --dry-run --limit 5

    # the model ids in TRANSLATORS are best-effort guesses: confirm them first
    python3 run_translators.py --verify-models

    # 20 items against the full roster, judged by Gemini 3.7 Flash
    python3 run_translators.py --limit 20

    # a subset of translators and a different judge
    python3 run_translators.py --model google:gemini-2.5-pro \
        --model google:gemma-4-31b-it --judge google:gemini-2.5-pro --limit 50

Results are appended to results/translations.jsonl (one line per item x model,
resumable), and aggregated into results/probe_results.jsonl in the same shape as
data/ltb_all_models.jsonl so the plotting scripts can read it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backends import Backend, ProviderError, build_backend, call_with_retries  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DEFAULT_DATA = os.path.join(PROJECT_ROOT, 'data', 'ltb_all_models.jsonl')
DEFAULT_OUT = os.path.join(PROJECT_ROOT, 'results', 'translations.jsonl')
DEFAULT_AGGREGATE = os.path.join(PROJECT_ROOT, 'results', 'probe_results.jsonl')
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, 'results', 'checkpoints.jsonl')

# This probe is Chinese -> English only; other language pairs are skipped.
SOURCE_LANG, TARGET_LANG = 'zh', 'en'
SOURCE_LANG_NAME, TARGET_LANG_NAME = 'Chinese', 'English'

# The translator roster, as (display name, provider, API model id).
#
# WARNING: the model id strings are best-effort guesses at each provider's
# naming convention, not verified ids. Run `--verify-models` to check them
# against the providers' live model lists before trusting a run, and override
# any that are wrong with `--model provider:correct-id`.
TRANSLATORS = (
    ('Gemini 2.5 Pro',         'google', 'gemini-2.5-pro'),
    ('Gemini 2.5 Flash',       'google', 'gemini-2.5-flash'),
    ('Gemini 3.5 Flash',       'google', 'gemini-3.5-flash'),
    ('Gemini 3.5 Flash Lite',  'google', 'gemini-3.5-flash-lite'),
    ('Gemini 3.6 Flash',       'google', 'gemini-3.6-flash'),
    ('Gemini 3.7 Flash',       'google', 'gemini-3.7-flash'),
    ('Gemini 3.8 Flash',       'google', 'gemini-3.8-flash'),
    ('Gemma 4 26B',            'google', 'gemma-4-26b-a4b-it'),
    ('Gemma 4 31B',            'google', 'gemma-4-31b-it'),
    ('GPT-5.6 Luna',           'openai', 'gpt-5.6-luna'),
)
DEFAULT_JUDGE = 'google:gemini-3.7-flash'          # Gemini 3.7 Flash

# Deliberately kept out of the roster, and blocked as a judge too. Requesting
# one of these needs --allow-excluded, so it cannot be reached by accident.
EXCLUDED_MODELS = {
    'gpt-5.6-sol': 'GPT-5.6 Sol is too expensive for this benchmark',
}

LABELS = {model: display for display, _, model in TRANSLATORS}

TRANSLATOR_SYSTEM = (
    f'You are a professional translator. Translate the user text faithfully from '
    f'{SOURCE_LANG_NAME} into {TARGET_LANG_NAME}. Preserve meaning, register, tone '
    f'and any implied social relations. Output only the translation: no commentary, '
    f'no quotation marks around it, no romanisation, no notes.'
)

JUDGE_SYSTEM = (
    'You are a strict evaluator for a machine-translation benchmark. You receive a '
    'source text, a candidate translation, and exactly ONE verification rule. Decide '
    'whether the candidate translation satisfies that rule.\n'
    'Rules:\n'
    '- Judge only the given rule. Ignore other flaws, style preferences and '
    'infelicities that the rule does not mention.\n'
    '- Any reasonable phrasing that satisfies the rule passes; the rule is about '
    'meaning, not wording.\n'
    '- If the rule is not satisfied, or the translation omits what the rule tests, '
    'the verdict is fail.\n'
    'Answer with JSON only: {"verdict": "pass" | "fail", "reason": "<25 words>"}'
)


@dataclass(frozen=True)
class ModelSpec:
    """One translator (or judge): a provider, a model id, and an output key."""

    provider: str
    model: str
    key: str = ''

    @property
    def output_key(self) -> str:
        return self.key or self.model

    @property
    def label(self) -> str:
        """Human-readable name, from the roster when the model is in it."""
        return LABELS.get(self.model, self.model)

    def __str__(self) -> str:
        return f'{self.provider}:{self.model}'


@dataclass
class RuleVerdict:
    """One verdict for one rule. `index` is its position in the item's list."""

    rule: str
    passed: bool
    reason: str = ''
    index: int = 0

    def as_dict(self) -> dict:
        return {'index': self.index, 'rule': self.rule, 'passed': self.passed,
                'reason': self.reason}


@dataclass
class Record:
    """One (item, translator) result, written as a single JSONL line."""

    id: str
    model_key: str
    provider: str
    model: str
    label: str = ''
    translation: str = ''
    rule_verdicts: List[RuleVerdict] = field(default_factory=list)
    n_rules: int = 0
    n_rules_passed: int = 0
    passed_all: Optional[bool] = None
    cached_translation: bool = False
    cached_verdicts: int = 0
    api_calls: int = 0
    judge: str = ''
    seconds: float = 0.0
    error: str = ''
    mechanism_en: str = ''
    source_lang: str = ''
    target_lang: str = ''

    def as_dict(self) -> dict:
        return {
            'id': self.id,
            'model_key': self.model_key,
            'label': self.label,
            'provider': self.provider,
            'model': self.model,
            'mechanism_en': self.mechanism_en,
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'translation': self.translation,
            'rule_verdicts': [v.as_dict() for v in self.rule_verdicts],
            'n_rules': self.n_rules,
            'n_rules_passed': self.n_rules_passed,
            'passed_all': self.passed_all,
            'cached_translation': self.cached_translation,
            'cached_verdicts': self.cached_verdicts,
            'api_calls': self.api_calls,
            'judge': self.judge,
            'seconds': round(self.seconds, 2),
            'error': self.error,
            'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        }


# --------------------------------------------------------------------- checkpoint

def _digest(text: str) -> str:
    """Short content hash, so edited sources or rules invalidate their cache."""
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]


class Checkpoint:
    """Append-only log of individual completed API calls.

    Every translation and every rule verdict is flushed to disk the moment it
    comes back, so an interrupted or rate-limited run re-does only the calls it
    never completed -- not the whole (item, model) task. Cache keys include the
    content hash of the source and of the rule, plus the judge model, so editing
    the dataset or switching judge invalidates just the affected entries.
    """

    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._translations: Dict[tuple, str] = {}
        self._verdicts: Dict[tuple, dict] = {}
        self._handle = None
        self._lock = threading.Lock()

    # -- keys
    @staticmethod
    def translation_key(item: dict, spec: 'ModelSpec') -> tuple:
        return (item['id'], spec.output_key, _digest(item['source']))

    @staticmethod
    def verdict_key(item: dict, spec: 'ModelSpec', judge: 'ModelSpec', index: int,
                    rule: str, translation: str, variant: str) -> tuple:
        # A verdict is a judgement about one exact (source, translation, rule)
        # triple by one judge under one prompt variant. Every part is in the key,
        # so editing any of them misses the cache instead of reusing a stale call.
        return (item['id'], spec.output_key, str(judge), index, _digest(rule),
                _digest(item['source']), _digest(translation), variant)

    # -- lifecycle
    def load(self) -> tuple:
        """Read an existing log. Returns (n translations, n verdicts) cached."""
        if not self.enabled or not os.path.exists(self.path):
            return 0, 0
        with open(self.path, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # torn last line: ignore it
                kind = entry.get('kind')
                if kind == 'translation':
                    self._translations[(entry['id'], entry['model_key'],
                                        entry['source_hash'])] = entry['text']
                elif kind == 'verdict':
                    self._verdicts[(entry['id'], entry['model_key'], entry['judge'],
                                    entry['rule_index'], entry['rule_hash'],
                                    entry.get('source_hash', ''),
                                    entry.get('translation_hash', ''),
                                    entry.get('variant', ''))] = entry
        return len(self._translations), len(self._verdicts)

    def open(self) -> None:
        if self.enabled and self._handle is None:
            os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
            self._handle = open(self.path, 'a', encoding='utf-8')

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def reset(self) -> None:
        self._translations.clear()
        self._verdicts.clear()
        if os.path.exists(self.path):
            os.remove(self.path)

    # -- reads and writes
    def get_translation(self, item: dict, spec: 'ModelSpec') -> Optional[str]:
        return self._translations.get(self.translation_key(item, spec))

    def put_translation(self, item: dict, spec: 'ModelSpec', text: str) -> None:
        key = self.translation_key(item, spec)
        self._translations[key] = text
        self._append({'kind': 'translation', 'id': item['id'],
                      'model_key': spec.output_key, 'source_hash': key[2],
                      'text': text})

    def get_verdict(self, item, spec, judge, index, rule, translation,
                    variant) -> Optional[RuleVerdict]:
        key = self.verdict_key(item, spec, judge, index, rule, translation, variant)
        entry = self._verdicts.get(key)
        if entry is None:
            return None
        return RuleVerdict(rule=rule, passed=bool(entry['passed']),
                           reason=entry.get('reason', ''), index=index)

    def put_verdict(self, item, spec, judge, index, rule, translation, variant,
                    verdict: RuleVerdict) -> None:
        key = self.verdict_key(item, spec, judge, index, rule, translation, variant)
        entry = {'kind': 'verdict', 'id': item['id'], 'model_key': spec.output_key,
                 'judge': str(judge), 'rule_index': index, 'rule_hash': key[4],
                 'source_hash': key[5], 'translation_hash': key[6],
                 'variant': variant, 'passed': verdict.passed,
                 'reason': verdict.reason}
        self._verdicts[key] = entry
        self._append(entry)

    def _append(self, entry: dict) -> None:
        """One line per completed call, flushed immediately."""
        if not self.enabled or self._handle is None:
            return
        with self._lock:
            self._handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
            self._handle.flush()


# ------------------------------------------------------------------------ prompts

def normalize_rules(item: dict) -> List[str]:
    """`verification_rules` as a clean list of rule strings.

    The field is a list in the dataset and often holds more than one rule, but a
    hand-edited file can carry a bare string; both are accepted here.
    """
    raw = item.get('verification_rules') or []
    if isinstance(raw, str):
        raw = [raw]
    return [rule.strip() for rule in raw if isinstance(rule, str) and rule.strip()]


def is_zh_en(item: dict) -> bool:
    return (item.get('source_lang') == SOURCE_LANG
            and item.get('target_lang') == TARGET_LANG)


def translation_prompt(item: dict) -> tuple:
    """(system, user) for translating one item."""
    parts = [f'Translate from {SOURCE_LANG_NAME} into {TARGET_LANG_NAME}.']
    context = item.get('context') or item.get('additional_context')
    if context:
        parts.append(f'Context to respect: {context}')
    parts.append('Text:\n' + item['source'])
    return TRANSLATOR_SYSTEM, '\n\n'.join(parts)


def judge_prompt(item: dict, translation: str, rule: str, rule_index: int,
                 n_rules: int, with_reference: bool = True) -> tuple:
    """(system, user) for ruling on ONE of the item's verification rules."""
    parts = [f'Source ({SOURCE_LANG_NAME}):\n{item["source"]}',
             f'Candidate translation ({TARGET_LANG_NAME}):\n{translation}']
    if with_reference and item.get('perfect_translation'):
        parts.append('Reference translation (ONE acceptable rendering -- do not require '
                     f'the candidate to match its wording):\n{item["perfect_translation"]}')
    header = (f'Verification rule {rule_index + 1} of {n_rules} '
              f'(judge only this one):' if n_rules > 1 else 'Verification rule:')
    parts.append(f'{header}\n{rule}')
    return JUDGE_SYSTEM, '\n\n'.join(parts)


def parse_verdict(text: str) -> RuleVerdict:
    """Parse the judge's JSON, tolerating fences and stray prose around it."""
    blob = text.strip()
    match = re.search(r'\{.*\}', blob, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            verdict = str(data.get('verdict', '')).strip().lower()
            reason = str(data.get('reason', ''))[:300]
            if verdict in ('pass', 'passed', 'true', 'yes'):
                return RuleVerdict('', True, reason)
            if verdict in ('fail', 'failed', 'false', 'no'):
                return RuleVerdict('', False, reason)
        except json.JSONDecodeError:
            pass
    # No parseable verdict: treat as a failure of the run, not of the translation.
    raise ValueError(f'unparseable judge response: {blob[:200]!r}')


# --------------------------------------------------------------------------- work

def translate_and_judge(item: dict, spec: ModelSpec, judge_spec: Optional[ModelSpec],
                        backends: Dict[str, Backend], args,
                        checkpoint: Optional['Checkpoint'] = None) -> Record:
    rules = normalize_rules(item)
    record = Record(id=item['id'], model_key=spec.output_key, label=spec.label,
                    provider=spec.provider, model=spec.model,
                    mechanism_en=item.get('mechanism_en', ''),
                    source_lang=item.get('source_lang', ''),
                    target_lang=item.get('target_lang', ''),
                    n_rules=len(rules),
                    judge=str(judge_spec) if judge_spec else '')
    started = time.time()
    try:
        cached = checkpoint.get_translation(item, spec) if checkpoint else None
        if cached:
            record.translation = cached
            record.cached_translation = True
        else:
            system, user = translation_prompt(item)
            record.translation = call_with_retries(
                backends[spec.provider].complete, spec.model, system, user,
                retries=args.retries, temperature=args.temperature)
            record.api_calls += 1
            if not record.translation:
                raise RuntimeError('empty translation')
            if checkpoint:                        # persist before judging starts
                checkpoint.put_translation(item, spec, record.translation)

        if judge_spec is not None:
            if not rules:
                raise RuntimeError('item has no verification rules to judge')
            # One judge call per rule: the rules are written to be atomic, and an
            # item passes only if the translation satisfies every one of them.
            variant = 'noref' if args.no_reference else 'ref'
            for index, rule in enumerate(rules):
                cached_verdict = (
                    checkpoint.get_verdict(item, spec, judge_spec, index, rule,
                                           record.translation, variant)
                    if checkpoint else None)
                if cached_verdict is not None:
                    record.rule_verdicts.append(cached_verdict)
                    record.cached_verdicts += 1
                    continue
                j_system, j_user = judge_prompt(
                    item, record.translation, rule, index, len(rules),
                    with_reference=not args.no_reference)
                raw = call_with_retries(
                    backends[judge_spec.provider].complete, judge_spec.model,
                    j_system, j_user, retries=args.retries, json_mode=True,
                    temperature=0.0 if args.temperature is None else args.temperature)
                record.api_calls += 1
                parsed = parse_verdict(raw)
                verdict = RuleVerdict(rule=rule, passed=parsed.passed,
                                      reason=parsed.reason, index=index)
                record.rule_verdicts.append(verdict)
                if checkpoint:                    # persist each verdict immediately
                    checkpoint.put_verdict(item, spec, judge_spec, index, rule,
                                           record.translation, variant, verdict)
            record.n_rules_passed = sum(1 for v in record.rule_verdicts if v.passed)
            record.passed_all = record.n_rules_passed == len(rules)
    except Exception as exc:
        record.error = f'{type(exc).__name__}: {exc}'
        # Whatever completed before the failure is already on disk in the
        # checkpoint log, so a rerun resumes from here rather than from scratch.
        record.n_rules_passed = sum(1 for v in record.rule_verdicts if v.passed)
        record.passed_all = None
    record.seconds = time.time() - started
    return record


# ------------------------------------------------------------------------ plumbing

def parse_spec(text: str) -> ModelSpec:
    """`provider:model` or `provider:model=output_key`."""
    body, _, key = text.partition('=')
    provider, sep, model = body.partition(':')
    if not sep or not provider.strip() or not model.strip():
        raise argparse.ArgumentTypeError(
            f'expected provider:model (e.g. openai:gpt-5), got {text!r}')
    return ModelSpec(provider.strip(), model.strip(), key.strip())


def check_excluded(specs: Sequence[ModelSpec], judge_spec: Optional[ModelSpec],
                   allow: bool) -> Optional[str]:
    """Refuse models on the cost blocklist unless explicitly allowed."""
    if allow:
        return None
    for spec in list(specs) + ([judge_spec] if judge_spec else []):
        reason = EXCLUDED_MODELS.get(spec.model)
        if reason:
            role = 'judge' if judge_spec is not None and spec is judge_spec \
                else 'translator'
            return (f'{reason}; refusing to use it as {role}. '
                    f'Pass --allow-excluded to override.')
    return None


def load_samples(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_samples(samples: Sequence[dict], args) -> tuple:
    """(selected items, note about what was dropped)."""
    chosen = [s for s in samples if is_zh_en(s)]
    skipped_lang = len(samples) - len(chosen)
    if not args.no_judge:
        with_rules = [s for s in chosen if normalize_rules(s)]
        skipped_rules = len(chosen) - len(with_rules)
        chosen = with_rules
    else:
        skipped_rules = 0
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(',') if i.strip()}
        chosen = [s for s in chosen if s['id'] in wanted]
    if args.mechanism:
        needle = args.mechanism.lower()
        chosen = [s for s in chosen if needle in s.get('mechanism_en', '').lower()]
    if args.shuffle:
        random.Random(args.seed).shuffle(chosen)
    if args.limit:
        chosen = chosen[:args.limit]

    notes = []
    if skipped_lang:
        notes.append(f'{skipped_lang} item(s) skipped: not {SOURCE_LANG}->{TARGET_LANG}')
    if skipped_rules:
        notes.append(f'{skipped_rules} item(s) skipped: no verification rules')
    return chosen, '; '.join(notes)


def load_done(path: str) -> set:
    """(id, model_key) pairs already recorded without error, for resuming."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get('error'):
                done.add((row.get('id'), row.get('model_key')))
    return done


def write_aggregate(records_path: str, samples: Sequence[dict], out_path: str) -> int:
    """Fold the flat records into dataset-shaped rows, one per item.

    Output matches data/ltb_all_models.jsonl: `model_outputs` and `model_passed_rules`
    keyed by model, so the same plotting code can consume these results.
    """
    by_item: Dict[str, dict] = {}
    with open(records_path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get('error'):
                continue
            bucket = by_item.setdefault(row['id'], {'outputs': {}, 'passed': {},
                                                    'verdicts': {}})
            bucket['outputs'][row['model_key']] = row['translation']
            if row.get('passed_all') is not None:
                bucket['passed'][row['model_key']] = bool(row['passed_all'])
            # Verdicts stay ordered by the rule's index in `verification_rules`.
            verdicts = sorted(row.get('rule_verdicts', []),
                              key=lambda v: v.get('index', 0))
            bucket['verdicts'][row['model_key']] = verdicts
            bucket['rules_passed'] = bucket.get('rules_passed', {})
            bucket['rules_passed'][row['model_key']] = [
                bool(v.get('passed')) for v in verdicts]

    index = {s['id']: s for s in samples}
    written = 0
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as handle:
        for item_id, bucket in by_item.items():
            item = index.get(item_id, {'id': item_id})
            row = {key: item[key] for key in
                   ('id', 'date', 'mechanism_zh', 'mechanism_en', 'source_lang',
                    'target_lang', 'source', 'perfect_translation',
                    'verification_rules') if key in item}
            row['verification_rules'] = normalize_rules(item)
            row['n_rules'] = len(row['verification_rules'])
            row['model_outputs'] = bucket['outputs']
            row['model_passed_rules'] = bucket['passed']
            row['model_passed_each_rule'] = bucket.get('rules_passed', {})
            row['rule_verdicts'] = bucket['verdicts']
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            written += 1
    return written


def _display_path(path: str) -> str:
    """Project-relative when it sits inside the repo, absolute otherwise."""
    absolute = os.path.abspath(path)
    if absolute.startswith(PROJECT_ROOT + os.sep):
        return os.path.relpath(absolute, PROJECT_ROOT)
    return absolute


def load_records(path: str) -> List[dict]:
    """Every recorded line, so a resumed run can summarise the whole file."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def print_summary(rows: Sequence[dict], specs: Sequence[ModelSpec],
                  errors: Sequence[Record] = ()) -> None:
    """Item-level and rule-level pass rates over every record in the file."""
    ok = [r for r in rows if not r.get('error')]
    print(f'\nResults file: {len(ok)} completed record(s), '
          f'{len(rows) - len(ok)} errored attempt(s)\n')

    width = max([len(spec.label) for spec in specs] + [12])
    print(f'{"model".ljust(width)}  {"items":>5}  {"passed":>6}  {"item rate":>9}'
          f'  {"rule rate":>9}')
    print('-' * (width + 38))
    for spec in specs:
        judged = [r for r in ok if r.get('model_key') == spec.output_key
                  and r.get('passed_all') is not None]
        if not judged:
            print(f'{spec.label.ljust(width)}  {0:>5}  {"-":>6}  {"-":>9}  {"-":>9}')
            continue
        passed = sum(1 for r in judged if r['passed_all'])
        n_rules = sum(r.get('n_rules', 0) for r in judged)
        rules_passed = sum(r.get('n_rules_passed', 0) for r in judged)
        rule_rate = f'{100 * rules_passed / n_rules:>8.1f}%' if n_rules else f'{"-":>9}'
        print(f'{spec.label.ljust(width)}  {len(judged):>5}  {passed:>6}  '
              f'{100 * passed / len(judged):>8.1f}%  {rule_rate}')

    # Items no model got right are the ones worth submitting.
    judged_by_item: Dict[str, list] = {}
    for row in ok:
        if row.get('passed_all') is not None:
            judged_by_item.setdefault(row['id'], []).append(row['passed_all'])
    complete = {i: v for i, v in judged_by_item.items() if len(v) == len(specs)}
    if complete:
        broke_all = sum(1 for v in complete.values() if not any(v))
        print(f'\nItems judged against all {len(specs)} models: {len(complete)}')
        print(f'Items no model passed: {broke_all} '
              f'({100 * broke_all / len(complete):.0f}%)')

    failed = list(errors)
    for record in failed[:5]:
        print(f'  ! {record.id} / {record.model_key}: {record.error[:160]}')
    if len(failed) > 5:
        print(f'  ! ... and {len(failed) - 5} more errors (see the results file)')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Probe OpenAI and Gemini translators on the LTB candidate pool.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='API keys are read from OPENAI_API_KEY and GOOGLE_API_KEY.')
    parser.add_argument('--data', default=DEFAULT_DATA, help='input JSONL')
    parser.add_argument('--out', default=DEFAULT_OUT, help='per-(item, model) JSONL')
    parser.add_argument('--aggregate', default=DEFAULT_AGGREGATE,
                        help='dataset-shaped output, one row per item')
    parser.add_argument('--model', dest='models', action='append', type=parse_spec,
                        metavar='PROVIDER:MODEL',
                        help='translator to probe; repeatable. Default: the '
                             f'{len(TRANSLATORS)}-model roster in TRANSLATORS '
                             '(see --list-models)')
    parser.add_argument('--list-models', action='store_true',
                        help='print the roster and exit')
    parser.add_argument('--verify-models', action='store_true',
                        help="check every configured model id against the providers' "
                             'live model lists and exit')
    parser.add_argument('--judge', type=parse_spec, metavar='PROVIDER:MODEL',
                        help=f'judge model (default {DEFAULT_JUDGE})')
    parser.add_argument('--no-judge', action='store_true',
                        help='translate only, skip rule judging')
    parser.add_argument('--no-reference', action='store_true',
                        help="do not show the judge the item's perfect translation")
    parser.add_argument('--limit', type=int, default=0, help='use at most N items')
    parser.add_argument('--ids', default='', help='comma-separated item ids')
    parser.add_argument('--mechanism', default='',
                        help='keep items whose mechanism_en contains this substring')
    parser.add_argument('--shuffle', action='store_true', help='sample randomly')
    parser.add_argument('--seed', type=int, default=0, help='shuffle seed')
    parser.add_argument('--concurrency', type=int, default=4,
                        help='parallel (item, model) tasks (default 4)')
    parser.add_argument('--retries', type=int, default=3,
                        help='retries per API call on transient errors')
    parser.add_argument('--timeout', type=float, default=120.0,
                        help='per-request timeout in seconds')
    parser.add_argument('--temperature', type=float, default=None,
                        help='sampling temperature; omitted from the request by default')
    parser.add_argument('--allow-excluded', action='store_true',
                        help='permit models on the cost blocklist ('
                             + ', '.join(sorted(EXCLUDED_MODELS)) + ')')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT,
                        help='append-only log of completed calls, used to resume '
                             'a failed run without repeating paid calls')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='do not read or write the checkpoint log')
    parser.add_argument('--overwrite', action='store_true',
                        help='ignore existing results and the checkpoint log, and '
                             're-run everything')
    parser.add_argument('--max-calls', type=int, default=2000,
                        help='abort if the run would need more API calls than this '
                             '(default 2000); raise it or pass --yes to proceed')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='skip the --max-calls guard')
    parser.add_argument('--dry-run', action='store_true',
                        help='no API calls and no key needed: exercises the pipeline')
    return parser


def print_roster(specs: Sequence[ModelSpec], judge_spec: Optional[ModelSpec]) -> None:
    width = max(len(spec.label) for spec in specs)
    print(f'{len(specs)} translators:')
    for spec in specs:
        print(f'  {spec.label.ljust(width)}  {spec.provider}:{spec.model}')
    print(f'\njudge: {judge_spec.label if judge_spec else "none"}'
          + (f'  ({judge_spec.provider}:{judge_spec.model})' if judge_spec else ''))
    if EXCLUDED_MODELS:
        print('\nexcluded (needs --allow-excluded): '
              + ', '.join(f'{model} -- {why}'
                          for model, why in sorted(EXCLUDED_MODELS.items())))
    print('\nModel ids are best-effort guesses at each provider\'s naming '
          'convention.\nRun --verify-models to check them, and override with '
          '--model provider:id.')


def verify_models(specs: Sequence[ModelSpec], judge_spec: Optional[ModelSpec],
                  timeout: float) -> int:
    """Check every configured id against the providers' model lists."""
    wanted = list(specs) + ([judge_spec] if judge_spec else [])
    available: Dict[str, set] = {}
    for provider in sorted({spec.provider for spec in wanted}):
        try:
            backend = build_backend(provider, timeout=timeout)
            available[provider] = backend.list_models()
        except ProviderError as exc:
            print(f'{provider}: cannot list models -- {exc}', file=sys.stderr)
            available[provider] = set()

    width = max(len(spec.label) for spec in wanted)
    missing = 0
    for spec in wanted:
        pool = available.get(spec.provider, set())
        if not pool:
            mark, note = '?', 'provider list unavailable'
        elif spec.model in pool:
            mark, note = 'OK', ''
        else:
            mark, note = 'MISSING', _suggest(spec.model, pool)
            missing += 1
        print(f'[{mark:>7}] {spec.label.ljust(width)}  {spec.provider}:{spec.model}'
              + (f'   {note}' if note else ''))
    if missing:
        print(f'\n{missing} id(s) not offered to this account. Fix TRANSLATORS or '
              'pass --model provider:correct-id.')
    return 1 if missing else 0


def _suggest(model: str, pool: Sequence[str], limit: int = 3) -> str:
    """Closest available ids, to make a wrong guess quick to correct."""
    import difflib
    close = difflib.get_close_matches(model, list(pool), n=limit, cutoff=0.4)
    if not close:
        stem = model.split('-')[0]
        close = sorted(m for m in pool if m.startswith(stem))[:limit]
    return ('closest: ' + ', '.join(close)) if close else ''



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    specs = args.models or [ModelSpec(provider, model)
                            for _, provider, model in TRANSLATORS]
    judge_spec = None if args.no_judge else (args.judge or parse_spec(DEFAULT_JUDGE))

    blocked = check_excluded(specs, judge_spec, args.allow_excluded)
    if blocked:
        print(f'Refused: {blocked}', file=sys.stderr)
        return 2

    if args.list_models:
        print_roster(specs, judge_spec)
        return 0
    if args.verify_models:
        return verify_models(specs, judge_spec, args.timeout)

    samples, note = select_samples(load_samples(args.data), args)
    if note:
        print(note)
    if not samples:
        print('No items selected -- check --ids / --mechanism / --limit.',
              file=sys.stderr)
        return 1

    checkpoint = Checkpoint(args.checkpoint, enabled=not args.no_checkpoint)
    if args.overwrite:
        if os.path.exists(args.out):
            os.remove(args.out)
        checkpoint.reset()
    n_translations, n_verdicts = checkpoint.load()
    done = set() if args.overwrite else load_done(args.out)
    tasks = [(item, spec) for item in samples for spec in specs
             if (item['id'], spec.output_key) not in done]

    # Estimate over remaining tasks only, so a resumed run reports honestly.
    judge_calls = sum(len(normalize_rules(item)) for item, _ in tasks) \
        if judge_spec else 0
    total_calls = len(tasks) + judge_calls
    rule_counts = sorted({len(normalize_rules(item)) for item in samples})
    print(f'{SOURCE_LANG}->{TARGET_LANG} items: {len(samples)}   '
          f'translators: {len(specs)}   '
          f'judge: {judge_spec.label if judge_spec else "none"}')
    print(f'Rules per item: {rule_counts} '
          f'(total {sum(len(normalize_rules(i)) for i in samples)})')
    print(f'Tasks to run: {len(tasks)} '
          f'({len(samples) * len(specs) - len(tasks)} already done)')
    print(f'API calls: ~{len(tasks)} translations'
          + (f' + ~{judge_calls} judgements = ~{total_calls}' if judge_spec else '')
          + (' (dry run: none)' if args.dry_run else ''))
    if checkpoint.enabled:
        reusable = sum(1 for item, spec in tasks
                       if checkpoint.get_translation(item, spec))
        print(f'Checkpoint {_display_path(checkpoint.path)}: '
              f'{n_translations} translations, {n_verdicts} verdicts cached'
              + (f' ({reusable} of the queued tasks need no translation call)'
                 if reusable else ''))

    if not args.dry_run and not args.yes and total_calls > args.max_calls:
        print(f'\nAborting: ~{total_calls} API calls exceeds --max-calls '
              f'{args.max_calls}.\nNarrow the run (--limit / --model), raise '
              f'--max-calls, or pass --yes.', file=sys.stderr)
        return 4

    providers = {spec.provider for spec in specs}
    if judge_spec is not None:
        providers.add(judge_spec.provider)
    try:
        backends = {name: build_backend(name, dry_run=args.dry_run,
                                        timeout=args.timeout)
                    for name in sorted(providers)}
    except ProviderError as exc:
        print(f'Setup failed: {exc}', file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    lock = threading.Lock()
    records: List[Record] = []
    interrupted = False
    checkpoint.open()
    try:
        with open(args.out, 'a', encoding='utf-8') as sink, \
                ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {pool.submit(translate_and_judge, item, spec, judge_spec,
                                   backends, args, checkpoint): (item, spec)
                       for item, spec in tasks}
            try:
                for done_count, future in enumerate(as_completed(futures), start=1):
                    record = future.result()
                    records.append(record)
                    with lock:
                        sink.write(json.dumps(record.as_dict(),
                                              ensure_ascii=False) + '\n')
                        sink.flush()
                    flag = 'ERR ' if record.error else (
                        'pass' if record.passed_all
                        else 'FAIL' if record.passed_all is False else '   -')
                    reused = ''
                    if record.cached_translation or record.cached_verdicts:
                        reused = (f'  [cached: '
                                  f'{"translation" if record.cached_translation else ""}'
                                  f'{"+" if record.cached_translation and record.cached_verdicts else ""}'
                                  f'{f"{record.cached_verdicts} verdict(s)" if record.cached_verdicts else ""}]')
                    print(f'[{done_count}/{len(futures)}] {flag}  {record.id:<16} '
                          f'{record.model_key}{reused}', flush=True)
            except KeyboardInterrupt:
                interrupted = True
                print('\nInterrupted: cancelling queued tasks. Completed calls are '
                      'checkpointed; re-run the same command to resume.',
                      file=sys.stderr)
                pool.shutdown(wait=False, cancel_futures=True)
    finally:
        checkpoint.close()

    print_summary(load_records(args.out), specs,
                  errors=[r for r in records if r.error])
    written = write_aggregate(args.out,
                              [s for s in load_samples(args.data) if is_zh_en(s)],
                              args.aggregate)
    calls_made = sum(r.api_calls for r in records)
    calls_saved = sum(int(r.cached_translation) + r.cached_verdicts for r in records)
    print(f'\nAPI calls this run: {calls_made}'
          + (f'   reused from checkpoint: {calls_saved}' if calls_saved else ''))
    print(f'Per-call records: {args.out}')
    print(f'Aggregated ({written} items): {args.aggregate}')
    if checkpoint.enabled:
        print(f'Checkpoint: {checkpoint.path}')
    errored = [r for r in records if r.error]
    if errored:
        print(f'\n{len(errored)} task(s) failed. Re-run the same command to retry '
              f'only those; their completed calls will not be paid for twice.')
    if interrupted:
        return 130
    return 0 if any(not r.error for r in records) or not records else 3


if __name__ == '__main__':
    raise SystemExit(main())
