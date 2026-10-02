---
pretty_name: "LTB-AI: AI-drafted Chinese→English items for the Last Translation Benchmark"
language:
- zh
- en
task_categories:
- translation
tags:
- machine-translation
- adversarial-evaluation
- llm-as-a-judge
- chinese
- synthetic
- benchmark-construction
annotations_creators:
- machine-generated
language_creators:
- machine-generated
size_categories:
- n<1K
configs:
- config_name: all_models
  default: true
  data_files: ltb_all_models.jsonl
- config_name: validation
  data_files: ltb_validation.jsonl
---

# LTB-AI: AI-drafted Chinese→English items for the Last Translation Benchmark

452 short Chinese source texts, each with a reference ("perfect") English translation and
1–2 atomic pass/fail **verification rules**, drafted by an LLM pipeline in the format of the
[Last Translation Benchmark](https://last-translation-benchmark.vilda.net) (LTB) — a
benchmark that collects inputs major translation systems still get wrong.

What makes this dataset more than a list of hard sentences: every item carries **the output
of 14 translation systems** from three providers, plus **a separate LLM judgement of every
rule for every output** (6,327 translations, 8,608 rule judgements) and one line of judge
reasoning per rule. So it is usable both as a small adversarial MT test set and as a record
of *how often an automated difficulty-hunting pipeline actually produces difficult items* —
it mostly does not: **45 of 452 items (10%) defeat all 14 systems, while 204 (45%) are
solved by all 14**.

- Code / pipeline: [github.com/OLAResearch/LTB-AI](https://github.com/OLAResearch/LTB-AI)
- Write-up of the experiment: [olaresearch.org/jis/blog/LTB-AI](https://olaresearch.org/jis/blog/LTB-AI)
- The benchmark this follows: [last-translation-benchmark.vilda.net](https://last-translation-benchmark.vilda.net)
  (its human-curated dataset is a separate artifact: [`zouhar/last-translation-benchmark`](https://hf.co/datasets/zouhar/last-translation-benchmark))

> **These items were never submitted to the official LTB.** The portal's checking tools were
> used only to *measure* this pipeline's output against a roster we do not run ourselves.
> Nothing here is part of the official benchmark.

## Files and configs

| File | Rows | Contents |
|---|---:|---|
| `ltb_all_models.jsonl` | 452 | The full candidate pool: item, reference, rules, 14 translators' outputs, per-rule verdicts, judge reasoning, curator verdict |
| `ltb_validation.jsonl` | 45 | The subset that defeated all 14 translators, plus results from the official portal's 11-system checking tool |
| `strict_passes_all_models.md` | — | Human-readable rendering of the submittable items (not a data file) |

Both configs have a single `train` split.

```python
from datasets import load_dataset

pool = load_dataset("OLAResearchX/LTB-AI", "all_models", split="train")
checked = load_dataset("OLAResearchX/LTB-AI", "validation", split="train")
```

A few keys are absent on most rows (`curator_override` on 3 rows, `missing_models` on 1, and
the portal's per-system fields on the 4 unchecked submissions). If a loader that infers the
schema from the first rows gives you trouble, pass explicit `features` or read the lines
directly:

```python
import pandas as pd
pool = pd.read_json("ltb_all_models.jsonl", lines=True)
```

## Data fields

### `all_models`

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable item id, mechanism-prefixed (`DEFECT-15`, `ADDR-04`, …) |
| `date` | string | Draft date (`2026-09-02` or `2026-09-03`) |
| `source` | string | Chinese source text. Median 19 characters, range 5–61 |
| `source_lang` / `target_lang` | string | Always `zh` / `en` |
| `perfect_translation` | string | The reference translation the rules were written against |
| `verification_rules` | list[string] | 1–2 atomic checks ("Check whether …"). 615 rules total: 289 items with 1, 163 with 2 |
| `n_rules` | int | `len(verification_rules)` |
| `mechanism_en` / `mechanism_zh` | string | The linguistic mechanism targeted, e.g. `Formal register: legal terminology`. 251 distinct labels; splitting at the colon gives 38 **families** |
| `translators` | list[string] | The 14 model ids probed for this item |
| `providers` | dict[str, str] | translator id → `anthropic` / `google` / `openai` |
| `judges` | dict[str, str] | translator id → the judge that scored it (`claude-opus-5` or `google:gemini-3.7-flash`) |
| `model_outputs` | dict[str, str] | translator id → its English translation |
| `model_passed_each_rule` | dict[str, list[bool]] | translator id → one verdict per rule, aligned with `verification_rules` |
| `model_passed_rules` | dict[str, bool] | translator id → `all()` of the above. **A model passes an item only if it satisfies every rule** |
| `judge_reasons` | dict[str, list[string]] | translator id → one sentence of judge reasoning per rule |
| `n_translators_passed` / `n_translators_total` | int | Pass count over the full roster (`total` is 14 for 451 items, 13 for `DEFECT-15`) |
| `n_claude_passed` / `n_claude_total` | int | The same over the 4 Claude translators only |
| `family_pass` | dict[str, {passed,total}] | Pass counts grouped by provider |
| `candidate_verdict` | string | Curator's call — see below |
| `notes` | string | Free-text curator note (present on 362 items, empty elsewhere) |
| `curator_override` | string | On 3 items only: why the item is held back regardless of the scores |
| `missing_models` | list[string] | On `DEFECT-15` only: `["gemini-2.5-pro"]`, the one translator×item cell never collected |

`candidate_verdict` is computed over the **Claude roster only** (4 models), which is a
historical artifact of the pipeline's order and the field most likely to mislead:

| Verdict | Items | Meaning |
|---|---:|---|
| `strict_pass` | 75 | No Claude translator passed |
| `marginal_pass` | 37 | Exactly one did |
| `rejected` | 340 | Two or more did — or the item carries a `curator_override` |

For any difficulty judgement, prefer `n_translators_passed` over `n_translators_total`. The
two views disagree by construction: 30 of the 75 `strict_pass` items are solved by a Gemini
or GPT model.

### `validation`

Same item fields (minus `candidate_verdict` / `notes` / `family_pass`), plus
`submission_index`, `curator_notes`, and a `portal` object holding the official checking
tool's results — a roster of 11 systems (Lara, Google Translate, Gemini 3.1 Pro, Gemma 4,
Llama 4 Maverick, GPT-6 Astra, GPT-5.6 Sol, Deepseek V4 Pro, Claude Sonnet 4.5, Gemini 3.8
Flash, Claude Haiku 4.5), most of which were never run locally:

`portal.status`, `portal.checked_on`, `portal.translators`, `portal.model_outputs`,
`portal.rule_results` (per-rule bools per system), `portal.n_translators_passed` /
`n_translators_total`, `portal.passed_translators`, `portal.notes`, `portal.summary`, and
`portal.reference_echoes` / `missing_verdicts` / `failed_rules` / `reviewer_comments`
(sparse).

## How the data was built

1. **Draft** — an LLM writes the source, the reference translation and the verification
   rules, aiming at a named mechanism. All Chinese sources are model-authored, not collected
   from users or the web.
2. **Probe (Claude)** — 4 Claude translators (Sonnet 5, Opus 4.5, Sonnet 4.5, Haiku 4.5),
   each rule judged in its own call by **Claude Opus 5**.
3. **Probe (rest)** — 10 Gemini / Gemma / GPT translators, each rule judged in its own call
   by **Gemini 3.7 Flash**.
4. **Select** — the 45 items that no translator passed went to the portal's checking tool.

Rules are judged one call each, because judging several together invites the judge to average
over them.

## Results worth knowing before you use it

Pass rate over all 452 items, per translator:

| Translator | Pass rate | | Translator | Pass rate |
|---|---:|---|---|---:|
| gemini-3.8-flash | 80% | | gemma-4-26b-a4b-it | 75% |
| gemini-3.7-flash | 79% | | gemma-4-31b-it | 75% |
| gemini-2.5-flash | 79% | | claude-opus-4-5 | 73% |
| gemini-3.5-flash | 79% | | gemini-3.5-flash-lite | 73% |
| gemini-3.6-flash | 78% | | claude-sonnet-4-5 | 71% |
| gpt-5.6-luna | 76% | | claude-sonnet-5 | 70% |
| gemini-2.5-pro | 76% | | claude-haiku-4-5-20251001 | 62% |

Model size and generation buy little here, and failures overlap heavily across the roster —
every pair of the 14 systems shares 71–110 of its failures, so items are hard for the field
rather than for one model.

Difficulty is very unevenly distributed across mechanism families. Share of items defeating
all 14 translators:

| Family | All-fail | | Family | All-fail |
|---|---:|---|---|---:|
| Source-defect fidelity | 13/23 | | Regional dialect | 3/13 |
| Taboo homophony | 8/16 | | Cultural knowledge | 2/11 |
| Politeness bias | 5/42 | | Coined abstract noun | 2/49 |
| Pragmatic self-reference | 3/23 | | **Formal register** | **3/123** |

`Formal register` is 27% of everything drafted and almost never works: "make it very formal"
is not a difficulty mechanism for current models. What survives is fidelity to *who is
speaking* — misspellings, dialect, an unpunctuated message home, a lay petitioner's
officialese — which every system tidies into fluent standard English.

Of the 45 items taken to the portal's 11-system checking tool (41 checked, credits ran out):

| `portal.status` | Items | Meaning |
|---|---:|---|
| `passed` | 34 | Cleared LTB's bar — at most 2 of 11 systems passed |
| `disputed` | 6 | Invalid: our own reference translation failed the rules written for it |
| `failed` | 1 | Over the bar — 3 of 11 systems passed |
| `not_checked` | 4 | Never run |

The `disputed` items are the most instructive failure in the pipeline: LTB judges the
reference alongside the systems, and a rule the reference itself cannot meet is a broken
rule, not a hard item.

## Limitations and bias

- **Two judges, one dataset.** The Claude column was judged by Claude Opus 5 and the rest by
  Gemini 3.7 Flash. Rankings *within* a provider group are measured; the gap *between* groups
  is indicative only. Opus 5 is also a family member of the Claude translators it scores.
- **Judge calibration is worth about 10%.** Re-judging unchanged translations moved 58 of 550
  verdicts (50 stricter, 8 more lenient). Treat single-rule verdicts as noisy labels.
- **Our roster overstates difficulty.** `ADDR-REG-01` was 0/14 locally and 3/11 on the portal.
  Expect the same direction of error for any item here.
- **Rules can be wrong.** 6 of 41 portal-checked items had rules their own reference failed,
  and the portal flagged rules that most systems pass as poor benchmark rules.
- **Coverage is uneven.** `Formal register` holds 123 items while 16 of 38 families hold four
  or fewer. Per-family rates on small families are pointers, not measurements.
- **Everything is model-generated,** including the sources, references, rules and the
  analysis in the repository README — with only light human verification. Expect residual
  errors in individual items.
- **Short, single-utterance Chinese only.** Median 19 characters; no document-level context,
  no other language pair.
- **Contamination.** This is a public dataset of known-hard inputs. Once published, high
  scores on it may reflect exposure rather than translation ability.
- **Offensive content.** Some items deliberately target taboo homophony, slurs and regionally
  offensive address terms; the `Taboo homophony` family (16 items) contains language that is
  crude or insulting by design.

## Citation

```bibtex
@misc{ltb-ai-2026,
  title  = {LTB-AI: Agentic Pipeline-drafted Chinese--English Items for the Last Translation Benchmark},
  author = {Shaoxiong Ji},
  year   = {2026},
  month  = {September},
  url    = {https://hf.co/datasets/OLAResearchX/LTB-AI},
  note   = {Dataset has not been submitted to the benchmark. Code: https://github.com/OLAResearch/LTB-AI}
}
```
