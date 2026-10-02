#!/usr/bin/env python3
"""End-to-end driver for the LTB pipeline.

Four stages, each a thin wrapper over the script that already does the work, so
anything you can do by hand you can do here with one command:

  claude   Claude models translate and Opus 5 judges every verification rule
           (run_claude_probe.py); with --new the results are judged into the
           staging file (merge_claude_results.py) and a freshly drafted
           batch is appended to the pool once it has been judged.
  others   Gemini / Gemma / GPT models translate the same items, judged by
           Gemini 3.7 Flash (run_translators.py).
  collect  Both probes' results are folded into data/ltb_all_models.jsonl
           (collect_results.py).
  plots    Figures are rebuilt from the collected file (visualization/).

Drafting the candidates themselves is not a stage: sources, reference
translations and verification rules are written by hand (with Claude's help) into
a staging JSONL, which you then pass as --new.

    # judge a freshly drafted batch, append it to the pool, and stop there
    python3 run_pipeline.py --stages claude --new ../data/new_batch.jsonl

    # the whole thing over the whole pool
    python3 run_pipeline.py --yes

    # rehearse it: no API calls, no writes to the real result files
    python3 run_pipeline.py --dry-run --limit 3

Every stage is resumable: re-running repeats only what did not finish.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
POOL = os.path.join(ROOT, 'data', 'ltb_all_models.jsonl')
COLLECTED = os.path.join(ROOT, 'data', 'ltb_all_models.jsonl')
VIZ = os.path.join(ROOT, 'visualization')

STAGES = ('claude', 'others', 'collect', 'plots')
OTHERS_JUDGE = 'google:gemini-3.7-flash'

# Field order for rows appended to the pool, matching merge_claude_results.py.
FIELD_ORDER = ('id', 'date', 'mechanism_zh', 'mechanism_en', 'source_lang',
               'target_lang', 'source', 'perfect_translation',
               'verification_rules', 'translators', 'judge', 'model_outputs',
               'model_passed_rules', 'model_passed_each_rule', 'judge_reasons',
               'n_translators_passed', 'n_translators_total', 'candidate_verdict',
               'previous_verdict',
               'curator_override', 'notes')


class StageError(RuntimeError):
    pass


def banner(text: str) -> None:
    print(f'\n{"=" * 72}\n== {text}\n{"=" * 72}', flush=True)


def run(cmd, cwd=HERE, allow_fail=False) -> int:
    printable = ' '.join(str(c) for c in cmd)
    print(f'$ {printable}', flush=True)
    code = subprocess.call([str(c) for c in cmd], cwd=cwd)
    if code and not allow_fail:
        raise StageError(f'exited {code}: {printable}')
    return code


def load_jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selection_flags(args):
    """The item-selection flags both probes understand."""
    flags = []
    for name in ('limit', 'ids', 'mechanism'):
        value = getattr(args, name)
        if value:
            flags += [f'--{name}', str(value)]
    return flags


def append_batch(staged_path: str, pool_path: str) -> int:
    """Append judged staging rows to the pool, refusing to duplicate anything."""
    staged, pool = load_jsonl(staged_path), load_jsonl(pool_path)
    pool_ids = {row['id'] for row in pool}
    pool_sources = {row['source'] for row in pool}

    unjudged = [r['id'] for r in staged if r.get('candidate_verdict') in (None, 'pending')
                or not r.get('model_outputs')]
    if unjudged:
        raise StageError(f'{len(unjudged)} staged items are unjudged '
                         f'(e.g. {unjudged[:3]}); re-run the claude stage')
    clashes = [r['id'] for r in staged if r['id'] in pool_ids]
    if clashes:
        raise StageError(f'id already in the pool: {clashes[:5]}')
    duplicates = [r['id'] for r in staged if r['source'] in pool_sources]
    if duplicates:
        raise StageError(f'source text already in the pool: {duplicates[:5]}')

    with open(pool_path, 'a', encoding='utf-8') as handle:
        for row in staged:
            row.setdefault('previous_verdict', None)
            handle.write(json.dumps({k: row[k] for k in FIELD_ORDER if k in row},
                                    ensure_ascii=False) + '\n')
    return len(staged)


def stage_claude(args):
    """Translate with the Claude roster, judge with Opus 5, fold into the pool."""
    target = args.new or args.data
    run(['python3', 'run_claude_probe.py', '--data', target,
         '--concurrency', args.concurrency, '--max-calls', args.max_calls]
        + selection_flags(args)
        + (['--yes'] if args.yes else [])
        + (['--dry-run', '--no-checkpoint', '--out', os.devnull] if args.dry_run else []))

    if args.dry_run:
        print('(dry run: skipping merge and append)')
        return
    if not args.new:
        # The pool carries Gemini and GPT columns that merge_claude_results.py
        # does not know about, so folding Claude results straight back into it
        # would drop them. The collect stage does that job, from both probes.
        print('Claude results are in results/; run the collect stage to fold them '
              f'into {os.path.relpath(args.collected, ROOT)}')
        return

    run(['python3', 'merge_claude_results.py', '--data', target, '--no-backup'])
    added = append_batch(target, args.data)
    print(f'appended {added} judged items to {os.path.relpath(args.data, ROOT)}')
    if args.keep_staging:
        print(f'staging file kept: {args.new}')
    else:
        os.remove(args.new)
        print(f'removed staging file {os.path.relpath(args.new, ROOT)}')


def stage_others(args):
    """Translate with Gemini / Gemma / GPT, judged by Gemini 3.7 Flash."""
    cmd = ['python3', 'run_translators.py', '--data', args.data,
           '--judge', args.judge, '--concurrency', args.concurrency,
           '--max-calls', args.max_calls] + selection_flags(args)
    if args.yes:
        cmd.append('--yes')
    if args.dry_run:
        cmd += ['--dry-run', '--no-checkpoint', '--out', os.devnull,
                '--aggregate', os.devnull]
    for name, var in (('google', 'GOOGLE_API_KEY'), ('openai', 'OPENAI_API_KEY')):
        if not args.dry_run and not os.environ.get(var):
            print(f'note: {var} is unset - {name} models will fail', file=sys.stderr)
    run(cmd)


def stage_collect(args):
    """Fold both probes' results into one file for the figures."""
    run(['python3', 'collect_results.py', '--data', args.data, '--out', args.collected]
        + (['--report'] if args.dry_run else []))


def stage_plots(args):
    """Rebuild the figures from the collected file."""
    try:
        subprocess.check_output([sys.executable, '-c', 'import matplotlib, numpy'],
                                stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        raise StageError('matplotlib/numpy are not importable by this interpreter. '
                         f'Install them ({os.path.basename(sys.executable)} -m pip install '
                         'matplotlib numpy) or run the plots stage with an interpreter '
                         'that has them.')
    source = args.collected if os.path.exists(args.collected) else args.data
    env_note = os.path.relpath(source, ROOT)
    print(f'plotting from {env_note}'
          + ('' if source == args.collected else '  (collected file not found)'))
    environment = dict(os.environ, LTB_JSONL=source)
    if args.models:
        environment['LTB_MODELS'] = args.models
    printable = f'LTB_JSONL={env_note} python3 plot_all.py'
    print(f'$ {printable}', flush=True)
    code = subprocess.call(['python3', 'plot_all.py'], cwd=VIZ, env=environment)
    if code:
        raise StageError(f'exited {code}: {printable}')


RUNNERS = {'claude': stage_claude, 'others': stage_others,
           'collect': stage_collect, 'plots': stage_plots}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stages', default=','.join(STAGES),
                        help=f'comma-separated subset of: {", ".join(STAGES)}')
    parser.add_argument('--data', default=POOL, help='the item pool')
    parser.add_argument('--new', help='staging JSONL of freshly drafted candidates: '
                                      'judged by the claude stage, then appended to the pool')
    parser.add_argument('--keep-staging', action='store_true',
                        help='do not delete the --new file after appending it')
    parser.add_argument('--collected', default=COLLECTED)
    parser.add_argument('--judge', default=OTHERS_JUDGE, help='judge for the others stage')
    parser.add_argument('--models', help='LTB_MODELS for the plots stage')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--ids')
    parser.add_argument('--mechanism')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--max-calls', type=int, default=2500)
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    unknown = [s for s in stages if s not in RUNNERS]
    if unknown:
        parser.error(f'unknown stage(s): {unknown}; pick from {", ".join(STAGES)}')
    if args.new:
        args.new = os.path.abspath(args.new)
        if not os.path.exists(args.new):
            parser.error(f'--new file not found: {args.new}')
        if 'claude' not in stages:
            parser.error('--new only means anything with the claude stage')

    print(f'pipeline: {" -> ".join(stages)}')
    print(f'pool:     {os.path.relpath(args.data, ROOT)}'
          + (f'   staging: {os.path.relpath(args.new, ROOT)}' if args.new else ''))
    if args.dry_run:
        print('DRY RUN: no API calls, no writes to the real result files')

    started, done = time.time(), []
    for stage in stages:
        banner(f'stage: {stage}')
        stage_started = time.time()
        try:
            RUNNERS[stage](args)
        except StageError as exc:
            print(f'\nSTOPPED in stage {stage}: {exc}', file=sys.stderr)
            print(f'completed: {", ".join(done) or "none"}', file=sys.stderr)
            return 1
        done.append(stage)
        print(f'-- {stage} finished in {time.time() - stage_started:.0f}s', flush=True)

    print(f'\nall stages finished in {time.time() - started:.0f}s: {", ".join(done)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
