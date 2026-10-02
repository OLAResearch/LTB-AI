# Translator probes

Two probes, one protocol. Each takes the LTB candidate pool, has every model in
its roster translate each item, and asks a judge model to rule pass/fail on
**each verification rule separately**. Both write dataset-shaped results and both
resume after an interruption.

| Script | Translators | Judge | Auth |
|---|---|---|---|
| [`run_claude_probe.py`](run_claude_probe.py) | Claude roster (4) | Claude Opus 5, fixed | the local `claude` CLI |
| [`run_translators.py`](run_translators.py) | Gemini / Gemma / GPT roster (10) | Gemini 3.7 Flash, `--judge` to swap | `GOOGLE_API_KEY`, `OPENAI_API_KEY` |
| [`merge_claude_results.py`](merge_claude_results.py) | — | — | judges a Claude-only staging batch into dataset shape |
| [`collect_results.py`](collect_results.py) | — | — | folds **both** probes into one file for plotting |
| [`run_pipeline.py`](run_pipeline.py) | — | — | runs the stages below end to end |
| [`backends.py`](backends.py) | — | — | provider wrappers used by `run_translators.py` |

## The pipeline in one command

[`run_pipeline.py`](run_pipeline.py) chains four stages, each a wrapper over the
script that does the work, so nothing here is hidden from you:

| Stage | What runs | Output |
|---|---|---|
| `claude` | `run_claude_probe.py`, then with `--new` `merge_claude_results.py` over the staging file and an append to the pool | `results/claude_translations.jsonl`, and with `--new` the item pool |
| `others` | `run_translators.py --judge google:gemini-3.7-flash` | `results/translations.jsonl` |
| `collect` | `collect_results.py` | `data/ltb_all_models.jsonl` |
| `plots` | `visualization/plot_all.py`, pointed at the collected file | `png/`, `pdf/` |

Drafting candidates is deliberately **not** a stage: sources, reference
translations and verification rules are written by hand (with Claude's help) into
a staging JSONL, which the `claude` stage then judges and appends.

```bash
# judge a freshly drafted batch and append it to the pool
python3 run_pipeline.py --stages claude --new ../data/new_batch.jsonl

# everything, over the whole pool
python3 run_pipeline.py --yes

# rehearse: no API calls, no writes to the real result files
python3 run_pipeline.py --dry-run --limit 3

# just refresh the collected file and the figures
python3 run_pipeline.py --stages collect,plots
```

Each stage is resumable because the scripts under it are; a failed stage stops
the run and names the stages that did complete. `--new` refuses to append a batch
that is unjudged, whose ids are already in the pool, or whose source texts
duplicate existing items.

## Collecting every probe into one file

The two probes write the same record shape to different files with different
judges. [`collect_results.py`](collect_results.py) folds them into
`data/ltb_all_models.jsonl`, one row per item, carrying the item's own fields
plus every model that has been run on it:

```bash
python3 collect_results.py --report            # coverage table, writes nothing
python3 collect_results.py                     # write data/ltb_all_models.jsonl
python3 collect_results.py --require-complete   # only items every model has run
```

`--data` and `--out` both default to the collected pool, so a run that finds an
empty or half-written `results/` would shorten the very file it reads. It refuses
to write fewer items than the existing output holds; `--force` overrides.

Each row keeps the flat `model_outputs` / `model_passed_rules` /
`model_passed_each_rule` keys the figures already read, and adds `providers`,
`judges` (per model), `n_translators_passed` / `n_translators_total`,
`n_claude_passed` / `n_claude_total` and `family_pass` (passed/total per provider
family).

> **Every row states its own denominator.** The pool's verdict is decided over the
> Claude roster while this file spans up to fourteen translators, so a bare "how
> many passed" count would mean different things in the two files. Both now carry
> `n_translators_total` beside the count, and the collected file adds the
> Claude-only pair, so the two views can be compared row by row. Records carrying an `error` are dropped rather than counted as
failures.

> **The two families are judged by different judges** — Opus 5 for Claude, Gemini
> 3.7 Flash for the rest — and Gemini 3.7 Flash judges its own translations. The
> per-model `judges` field records this, and the provider figure states it in its
> subtitle. Treat cross-provider gaps as indicative, not measured.

Both probes are **Chinese-to-English only**: items whose `source_lang`/`target_lang`
are anything else are skipped, not translated.

## The protocol both scripts follow

1. **One translation per (item, model).** The translator prompt is identical in
   both scripts, so their numbers are comparable: translate faithfully, preserve
   register and implied social relations, output the translation and nothing else.
2. **One judge call per rule.** `verification_rules` is a list and items commonly
   carry more than one. Rules are written to be atomic, and judging them together
   invites the judge to average over them, so each gets its own call. The judge
   sees the source, the item's `perfect_translation` as a reference (suppress it
   with `--no-reference`), the candidate, and exactly one rule; it answers
   `{"verdict": "pass"|"fail", "reason": "…"}`.
3. **A model passes an item only when it passes every rule of that item.** That
   all-or-nothing definition is what `passed_all` records and what the figures
   and verdicts use throughout.
4. **Every call is checkpointed as it returns**, so a rerun of the same command
   pays only for what never completed.

A bare string in `verification_rules` is accepted and treated as one rule; an
item with no rules is skipped (unless the run is translate-only).

---

# Claude probe (`run_claude_probe.py`)

## Roster

Translators are every Claude model the machine can call except Opus 5, which is
reserved as judge and curator. The roster lives in the `TRANSLATORS` tuple and
the judge in the `JUDGE` constant, both at the top of the file.

| Role | Model | API id |
|---|---|---|
| translator | Claude Sonnet 5 | `claude-sonnet-5` |
| translator | Claude Haiku 4.5 | `claude-haiku-4-5-20251001` |
| translator | Claude Sonnet 4.5 | `claude-sonnet-4-5` |
| translator | Claude Opus 4.5 | `claude-opus-4-5` |
| judge | Claude Opus 5 | `claude-opus-5` |

There is no `--judge` flag: swapping the judge means editing `JUDGE`. Two models
are deliberately absent — Claude Fable 5 needs usage credits this account does
not have, and `claude-opus-4-1` is silently remapped to Opus 5 by the CLI, so
including it would double-count Opus 5.

## How it calls models

The machine authenticates through Claude Code and has no `ANTHROPIC_API_KEY`, so
the probe shells out to the headless CLI rather than an SDK:

```
claude -p --model <id> --system-prompt <prompt>
        --disable-slash-commands --strict-mcp-config --no-session-persistence
```

`--system-prompt` *replaces* the agent system prompt, so each call is a plain
single-turn model call: no tools, no CLAUDE.md, no session state, nothing carried
between items. The binary is taken from `CLAUDE_CODE_EXECPATH`, falling back to
`claude` on `PATH`.

Failures are sorted into two kinds. Anything matching a fatal marker — missing
credits, bad auth, unknown model — raises immediately, because retrying cannot
help. Everything else (timeouts, transient CLI errors, rate and session limits)
is retried `--retries` times with exponential backoff and jitter, and a task that
still fails is reported and skipped, leaving the rest of the run to finish.

## Quick start

Work down this list and stop where you have what you need. Run from `benchmark/`.

```bash
# 0. free: roster, and the whole pipeline on deterministic stubs
python3 run_claude_probe.py --list-models
python3 run_claude_probe.py --dry-run --limit 3 --no-checkpoint --out /dev/null

# 1. smallest useful live run: one model, two items
python3 run_claude_probe.py --model claude-haiku-4-5-20251001 --limit 2

# 2. one mechanism family, whole roster -- more informative than a random slice
python3 run_claude_probe.py --mechanism "politeness" --limit 20 --concurrency 8

# 3. specific items, e.g. after editing their rules
python3 run_claude_probe.py --ids IRONY-03,DEFECT-05

# 4. the full pool against the full roster
python3 run_claude_probe.py --concurrency 12 --yes

# 5. fold the results into data/ltb_all_models.jsonl
python3 collect_results.py --report           # coverage table, writes nothing
python3 collect_results.py                    # rewrite the collected pool
```

Read a few verdicts before scaling up — the judge's reasoning is recorded:

```bash
python3 -c "import json; [print(r['model_key'], r['passed_all'], '|',
    r['translation'][:70], '|', [(v['index'], v['passed'], v['reason'][:50])
    for v in r['rule_verdicts']]) for r in
    map(json.loads, open('../results/claude_translations.jsonl'))][:8]"
```

Unattended, with a log to read afterwards:

```bash
nohup python3 run_claude_probe.py --concurrency 12 --yes \
    > ../results/claude_run.log 2>&1 &
tail -f ../results/claude_run.log
```

## Flags

| Flag | Effect |
|---|---|
| `--list-models` | print the roster and judge, then exit |
| `--data PATH` | input JSONL (default `data/ltb_all_models.jsonl`) |
| `--model ID` | one translator, repeatable; default is the whole roster |
| `--limit N`, `--ids A,B`, `--mechanism STR`, `--shuffle --seed S` | choose which items to run |
| `--concurrency N` | parallel (item, model) tasks, default 8 |
| `--retries N`, `--timeout S` | retries with backoff; per-call timeout in seconds (default 3, 180) |
| `--no-reference` | hide the `perfect_translation` from the judge |
| `--no-reuse-existing` | re-translate even where the dataset already carries that model's output |
| `--max-calls N`, `--yes` | refuse a run whose estimate exceeds N (default 2500); `--yes` skips the check |
| `--out PATH`, `--checkpoint PATH`, `--no-checkpoint` | where results and the call log go, or no log |
| `--overwrite` | delete the results file *and* the checkpoint, then start fresh |
| `--dry-run` | deterministic stubs: no calls, no cost |

## Sizing a run

For *N* items carrying *R* rules between them, against *M* models:

```
translations = N × M          judgements = R × M          calls = (N + R) × M
```

Judging is the larger half: every item carries at least one rule and many carry two, so
*R* ≥ *N* always. The current pool is **452 items and 615 rules**, so one translator costs
1,067 calls and the full 14-model roster came to 6,327 translations plus 8,608
judgements.
Every run prints its own estimate before making a call and refuses to exceed
`--max-calls` without `--yes`, so the cheap way to size anything is to ask:

```bash
python3 run_claude_probe.py --dry-run --no-checkpoint --out /dev/null | head -2
```

Two things make the real number smaller than the formula. Translations already
present in the dataset's `model_outputs` are reused unless you pass
`--no-reuse-existing` — re-judging a pool the probe has already translated costs
judge calls only — and anything in the checkpoint is free. Note that the default
`--max-calls` will refuse a full-roster run over the whole pool: that is intended
friction, pass `--yes` when you mean it.

## Resume, and the checkpoint

Resume happens at two levels:

1. **Task level.** A completed (item, model) line in the results file means the
   task is skipped entirely on the next run.
2. **Call level.** For a task that did *not* complete, the checkpoint supplies
   whatever calls already succeeded — a failure while judging rule 2 of 2 keeps
   the translation and the rule-1 verdict, and the rerun makes exactly one call.

Checkpoint keys carry content hashes, so nothing stale is reused: a translation
is keyed by (item, model, **source hash**), a verdict by (item, model, judge,
rule index, **rule hash**, **translation hash**, reference-or-not). Edit a source
or a rule and the affected entries miss the cache and are recomputed; everything
else is still reused. A torn final line, from a process killed mid-write, is
skipped when the log is loaded rather than aborting the run.

So the recovery procedure for a rate limit, a session limit, a crash or a Ctrl-C
is: **re-run the identical command.** Nothing else — no flag, no cleanup.

> **Gotcha: a dry run pollutes both output files.** `--dry-run` skips the network
> but not the bookkeeping: its stub translations and pseudo-verdicts are written
> to the results file *and* to the checkpoint, where a later real run will reuse
> them as though they were genuine. Always isolate a dry run —
> `--dry-run --no-checkpoint --out /dev/null` — or give it its own paths. If one
> has already leaked in, the stubs contain the literal string `dry-run`, so they
> are easy to grep out. The same applies to `run_translators.py`.

## What it writes

- `results/claude_translations.jsonl` — one line per (item, model): the
  translation, one verdict per rule with the judge's reason, `n_rules`,
  `n_rules_passed`, `passed_all`, latency, and how many calls the task paid for
  versus reused.
- `results/claude_checkpoints.jsonl` — the call-level log described above.

## Merging into the dataset (`merge_claude_results.py`)

This script predates the Gemini and GPT probes: it knows the four Claude
translators only, and rewrites each item to the Claude-shaped field order. Run
against `data/ltb_all_models.jsonl` it would drop every non-Claude column, so it
refuses (`--force` overrides). Its job now is a **staging batch** — a freshly
drafted file with no other providers in it — which the `claude` stage judges and
appends to the pool; the pool itself is rebuilt by `collect_results.py`.

The merge reads the results file, groups it by item, and rewrites its `--data`
file in place, adding `translators`, `judge`, `model_outputs`, `model_passed_rules`,
`model_passed_each_rule`, `judge_reasons`, `n_translators_passed`,
`n_translators_total` and `previous_verdict`, then recomputing
`candidate_verdict`:

| Translators passing every rule | Verdict |
|---|---|
| 0 of 4 | `strict_pass` |
| 1 of 4 | `marginal_pass` |
| 2 or more | `rejected` |

Two rules override that arithmetic. Items listed in `CURATOR_OVERRIDES` stay
`rejected` however they score — they fail LTB's fairness or confidence bar rather
than the count, and the reason is written to the item as `curator_override`. And
an item **missing any translator keeps its previous verdict** rather than being
scored as though the absent model had failed; it is flagged with
`incomplete_translators` and listed in the run summary.

The merge prints per-translator pass rates and every verdict transition before
writing, so `--dry-run` first is worth the second. A `.bak` of the dataset is
kept unless you pass `--no-backup`.

## Probing a new batch of candidates

Both scripts take `--data`, so a batch of freshly drafted items can be judged
without touching the pool, then appended once it looks right:

```bash
# new_batch.jsonl: id, date, mechanism_*, source_lang/target_lang, source,
# perfect_translation, verification_rules -- the probe needs nothing else
python3 run_claude_probe.py --data ../data/new_batch.jsonl --yes
python3 merge_claude_results.py --data ../data/new_batch.jsonl --no-backup
cat ../data/new_batch.jsonl >> ../data/ltb_all_models.jsonl && rm ../data/new_batch.jsonl
```

Check for id collisions before appending — nothing downstream enforces unique
ids.

## Caveats

- **The judge is family to the translators.** Opus 5 judges four Claude models,
  and Opus 4.5 is a near-neighbour it may score generously. The honest control is
  a judge from outside the family: run the same items through
  `run_translators.py --judge` and compare.
- **Judge calibration is worth roughly a tenth of the verdicts.** Re-judging
  translations that had not changed at all moved about 10% of item verdicts,
  mostly in the stricter direction. Treat any single-judge ranking of
  near-neighbours as provisional.
- **Chat models are not translation products.** These are general models prompted
  to translate, which is not what ChatGPT or Google Translate do for the LTB
  reviewers a submission is ultimately judged against.

---

# Gemini/Gemma/GPT probe (`run_translators.py`)

Same protocol, more providers, real API keys. Model responses come through
`backends.py`, which imports each SDK lazily, retries transient failures, and
drops parameters a model rejects (`temperature`, `response_mime_type`, …) before
retrying rather than failing the call.

## Setup

```bash
python3 -m pip install --user google-genai   # Gemini + Gemma (NOT google-generativeai)
python3 -m pip install --user openai         # only if you run an OpenAI model

export GOOGLE_API_KEY='...'
export OPENAI_API_KEY='sk-...'
```

A key is needed only for the providers you actually request, and none at all for
`--dry-run`. To keep them out of your user site-packages:

```bash
python3 -m venv ~/.venvs/ltb && ~/.venvs/ltb/bin/pip install google-genai openai
alias ltb=~/.venvs/ltb/bin/python        # then: ltb run_translators.py ...
```

## Roster

| Model | Provider | API id |
|---|---|---|
| Gemini 2.5 Pro | google | `gemini-2.5-pro` |
| Gemini 2.5 Flash | google | `gemini-2.5-flash` |
| Gemini 3.5 Flash | google | `gemini-3.5-flash` |
| Gemini 3.5 Flash Lite | google | `gemini-3.5-flash-lite` |
| Gemini 3.6 Flash | google | `gemini-3.6-flash` |
| Gemini 3.7 Flash | google | `gemini-3.7-flash` |
| Gemini 3.8 Flash | google | `gemini-3.8-flash` |
| Gemma 4 26B | google | `gemma-4-26b-a4b-it` |
| Gemma 4 31B | google | `gemma-4-31b-it` |
| GPT-5.6 Luna | openai | `gpt-5.6-luna` |

Judge: **Gemini 3.7 Flash**, overridable with `--judge provider:model`.

Models in `EXCLUDED_MODELS` (currently GPT-5.6 Sol, on cost) are blocked as both
translator and judge; requesting one needs `--allow-excluded`, and without it the
run refuses and exits 2, so it cannot be reached by a typo.

Gemma is served through the Gemini API but has no system-instruction slot, so for
`gemma*` models the system prompt is folded into the user turn automatically.

## Quick start

```bash
# 0. free
python3 run_translators.py --list-models
python3 run_translators.py --dry-run --limit 5 --no-checkpoint --out /dev/null

# 1. do the configured ids exist? one models.list call per provider
python3 run_translators.py --verify-models

# 2. smoke test: one translator, two items
python3 run_translators.py --model google:gemma-4-26b-a4b-it --limit 2

# 3. one family, one translator
python3 run_translators.py --model google:gemma-4-26b-a4b-it \
    --mechanism "politeness" --limit 20 --concurrency 4

# 4. small multi-model comparison, with a judge from another family
python3 run_translators.py --model google:gemini-2.5-pro --model google:gemma-4-31b-it \
    --judge openai:gpt-5.6-luna --limit 50 --concurrency 8

# 5. full sweep -- must be asked for explicitly, see --max-calls
python3 run_translators.py --concurrency 8 --max-calls 8000
```

Prefer running the full sweep **one translator at a time**: rate limits bite
less, the checkpoint makes it free to stop between models, and one bad model id
costs a single model's calls instead of derailing the batch.

```bash
for model in gemini-2.5-pro gemini-2.5-flash gemini-3.7-flash gemini-3.8-flash \
             gemma-4-26b-a4b-it gemma-4-31b-it; do
  echo "=== $model ==="
  python3 run_translators.py --model "google:$model" --concurrency 8 --yes || break
done
```

## Flags beyond the shared set

| Flag | Effect |
|---|---|
| `--verify-models` | check configured ids against each provider's live list |
| `--judge PROVIDER:MODEL` | judge with a different model — the cross-family control |
| `--allow-excluded` | permit a model on the cost blocklist |
| `--no-judge` | translate only, skip judging entirely |
| `--temperature T` | sent only if you pass it; some reasoning models reject it |
| `--aggregate PATH` | where the per-item roll-up is written |

Otherwise the surface matches the Claude probe: `--limit/--ids/--mechanism/
--shuffle/--seed`, `--concurrency` (default 4), `--retries/--timeout` (3, 120s),
`--max-calls` (default 2000) with `--yes`, `--checkpoint/--no-checkpoint`,
`--overwrite`, `--no-reference`, `--dry-run`.

## What it writes

- `results/translations.jsonl` — one line per (item, model), same shape as the
  Claude probe's output, plus provider and any error.
- `results/checkpoints.jsonl` — the call-level log.
- `results/probe_results.jsonl` — the same results folded to one row per item and
  shaped like `data/ltb_all_models.jsonl`, so the figures can read it directly:
  `model_outputs`, `model_passed_rules`, `model_passed_each_rule` and the full
  `rule_verdicts`.

There is no merge script for this probe: it aggregates into
`probe_results.jsonl` and leaves the dataset alone.

## Noise you may see

- `Direct use of automatic function calling (AFC) …` — a `google-genai` log line,
  not an error. `GoogleBackend` declares AFC disabled, so it should not appear;
  on an SDK too old to have that config class the field is skipped and the
  warning is harmless.
- `INFO httpx: HTTP Request: POST …` appears only if your program configures
  logging at INFO/DEBUG; neither probe configures logging itself.

---

# Plotting probe results

`visualization/` reads whatever roster it is told about. `ltb_data.MODELS`,
`MODEL_LABELS` and `MODEL_COLORS` declare the models, and every figure sizes
itself from that list — bar widths, heatmap columns, funnel stages, the
co-failure matrix and the outcome labels (`all_fail` / `some_fail` / `all_pass`).
Pointing the figures at a probe run is a four-line edit in
[`../visualization/ltb_data.py`](../visualization/ltb_data.py):

```python
MODELS = ['gemini-2.5-pro', 'gpt-5.6-luna']            # keys used in the probe output
MODEL_LABELS = {'gemini-2.5-pro': 'Gemini 2.5 Pro', 'gpt-5.6-luna': 'GPT-5.6 Luna'}
MODEL_COLORS = {'gemini-2.5-pro': PALETTE['blue_main'], 'gpt-5.6-luna': PALETTE['teal']}
DEFAULT_JSONL = os.path.join(PROJECT_ROOT, 'results', 'probe_results.jsonl')
```

Then `python3 plot_all.py`. Every model named in `MODELS` must be present in the
file being read.
