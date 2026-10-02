# LTB visualization

Publication-style figures for the Chinese-to-English *Last Translation Benchmark*
candidate pool in [`../data/ltb_all_models.jsonl`](../data/ltb_all_models.jsonl)
(452 items, 14 translators across three providers), and the portal round recorded in
[`../data/ltb_validation.jsonl`](../data/ltb_validation.jsonl).

Scripts live here; outputs are written one directory up, split by format:
300 dpi rasters into [`../png/`](../png) and vector versions into [`../pdf/`](../pdf).

Style follows one set of house conventions throughout:
Helvetica-like sans fonts, top/right spines off, frameless legends (or a dedicated
legend panel), blue/green/red/neutral palette, `tight_layout(pad=2)`, and every
figure exported as 300 dpi PNG plus vector PDF.

Every figure here reports how the translator roster performed. Descriptions of
the corpus itself — source lengths, items per mechanism, how the annotation is
composed — are one-line facts rather than figures, and the item counts they used
to carry now appear on the axis labels of the figures below.

Colour semantics are the same in every outcome figure: blue is an item that
defeats the whole roster, neutral is one every model solves, green is in between.

**How the middle is split depends on how many models there are.** With a handful of
translators, "some fail" is a useful category. With fourteen it is not: it would hold
"one model slipped" and "thirteen of fourteen failed" in the same bar. So
from **five models up** (`WIDE_ROSTER` in `ltb_data.py`) the middle is graded by
the fraction of the roster that failed:

| Band | Colour | Roster < 5 | Roster ≥ 5 |
|---|---|---|---|
| `all_fail` | Blue `#0F4D92` | every model fails | every model fails |
| `most_fail` | Blue `#3775BA` | — | two thirds or more fail |
| `mixed` | Green `#8BCF8B` | — | a third to two thirds fail |
| `few_fail` | Green `#AADCA9` | — | up to a third fail |
| `some_fail` | Green `#8BCF8B` | any partial failure | — |
| `all_pass` | Neutral `#CFCECE` | every model passes | every model passes |

`load_samples()` puts the band on each item as `band`, and `band_fractions()`
returns the per-band shares; `outcome` is still there, with its original three
values, for anything that wants the coarse split.

## The roster

`ltb_data.MODELS` holds the translators the figures report on. It is **not** a hand-kept
list: at import it is read from the dataset — every model covering at least
`MIN_COVERAGE` (90%) of the items — and ordered by provider family. On the current file
that resolves to **14 translators**:

| Provider | Models | Colours |
|---|---|---|
| Anthropic | Claude Sonnet 5, Haiku 4.5, Sonnet 4.5, Opus 4.5 | blue / violet family |
| Google | Gemini 2.5 Pro, 2.5 / 3.5 / 3.5 Lite / 3.6 / 3.7 / 3.8 Flash, Gemma 4 26B / 31B | green / teal family |
| OpenAI | GPT-5.6 Luna | red family |

Labels are generated from the key (`gemini-3.8-flash` → "Gemini 3.8 Flash") with
exceptions in `KNOWN_LABELS`, and colours are assigned per provider so a Claude column
always reads blue and a Gemini one green. Every figure derives its dimensions from the
list — bar widths, heatmap columns, funnel stages, the co-failure matrix, the outcome
band labels — so probing more translators means re-running the probe and the plots,
with no edit to the figure code. Set `LTB_MODELS` to restrict the roster by hand.

## Run

```bash
python3 plot_all.py                  # all five figures, from data/ltb_all_models.jsonl
python3 plot_provider_comparison.py  # or any single figure
```

Requires only `numpy` and `matplotlib`.

### Plotting a different file, or a different roster

Two environment variables steer every script, so nothing needs editing:

| Variable | Effect |
|---|---|
| `LTB_JSONL` | the file to read (default `../data/ltb_all_models.jsonl`) |
| `LTB_MODELS` | comma-separated roster; default is read from the file |
| `LTB_FIG_SUFFIX` | suffix for output names; default is derived from the filename |

```bash
# three translators, one per provider
LTB_MODELS=claude-opus-4-5,gemini-3.8-flash,gpt-5.6-luna python3 plot_provider_comparison.py

# a different dataset entirely
LTB_JSONL=../results/probe_results.jsonl python3 plot_all.py
```

With `LTB_MODELS` unset, the roster is every model covering at least 90% of the
items (`MIN_COVERAGE`), so one unprobed item cannot delete a whole column; models
below that are named on stderr, and the few items the chosen roster does not cover
are skipped by the loader, with the count reported.

Figures from a non-default file get a suffix (`provider_comparison_all_models.png`),
so plotting the collection never overwrites the Claude-pool figures.

### How the figures adapt to roster size

Every outcome figure works at 2 models and at 14; several change form rather
than shrink type, because some designs simply stop being readable:

| Figure | Small roster | Wide roster |
|---|---|---|
| `plot_provider_comparison` | **declines** a single-provider roster: nothing to compare | horizontal bars blocked by provider, plus a model × family heatmap |
| `plot_mechanism_heatmap` | one failure column per model | above 6 models, columns aggregate to **mean failure per provider family**; canvas width follows the column count |
| `plot_yield_funnel` (b) | cell = items the pair both fail | above 6 models, cell = **% of the row model's failures the column shares**, rows grouped by provider with rules between families |
| `plot_yield_funnel` (c) | one labelled bucket per model count | labels thin out, bars widen, the denominator moves to the axis label |
| `plot_family_outcomes` | three bands, large type | five graded bands; canvas and type scale with the family count |

Bar charts also cap how many **families** they name, because every family adds a
cluster: `plot_family_outcomes` names 12, pooling the rest into `Other`. Ranking by size alone would pool away small families that
defeat every model — the ones the benchmark exists to find — so
`family_groups(..., keep_hardest=N)` reserves slots for the hardest families,
measured by how much of the roster they defeat. The heatmaps are uncapped: a
family costs them a row, not a cluster.

The thresholds are constants at the top of each script (`MAX_MODELS`,
`MAX_MODEL_COLUMNS`, `WIDE_ROSTER`), so they are easy to move.

## Files

| Script | Output | What it shows |
|---|---|---|
| `ltb_data.py` | — | Palette, rcParams, JSONL loader, mechanism-family derivation, `png/`+`pdf/` export routing |
| `plot_provider_comparison.py` | `../png/provider_comparison.png`, `../pdf/provider_comparison.pdf` | **The cross-provider figure**: (a) one horizontal bar per translator, blocked by provider, (b) model x mechanism-family heatmap. Draws nothing unless the roster spans two or more providers |
| `plot_family_outcomes.py` | `../png/family_outcomes.png`, `../pdf/family_outcomes.pdf` | Stacked composition per mechanism family: how much of the roster each family defeats |
| `plot_mechanism_heatmap.py` | `../png/mechanism_heatmap.png`, `../pdf/mechanism_heatmap.pdf` | Family x failure-mode heatmap: one column per translator (per provider above 6 models), then joint failure and the LTB-admissible share. An `All items` row carries the overall per-model rate |
| `plot_yield_funnel.py` | `../png/yield_funnel.png`, `../pdf/yield_funnel.pdf` | (a) candidate yield funnel, (b) pairwise co-failure matrix, (c) verdict composition by number of passing models |
| `plot_portal_results.py` | `../png/portal_results.png`, `../pdf/portal_results.pdf` | The official portal round: (a) outcome of the 45 checked items, (b) how many of the 11 portal systems each rule defeats, (c) outcome by mechanism. Reads `ltb_validation.jsonl`, not the pool |

Removed as not about translator results: `plot_input_statistics.py`,
`plot_samples_per_mechanism.py` and `plot_mechanism_taxonomy.py` described the
corpus rather than the models; `plot_formal_register.py` drilled into a single
family; `plot_model_comparison.py`'s grouped bars are what
`plot_provider_comparison.py` does at any roster size.

## Derived fields

`ltb_data.load_samples()` adds to each record:

- `family` / `subtype` — `mechanism_en` split at the colon, e.g.
  `Formal register: legal terminology` -> family `Formal register`, subtype `legal terminology`.
- `n_translators_passed` / `n_translators_total` — models in `MODELS` whose
  translation satisfied *all* verification rules, and how many models that was out
  of. Both are derived from `model_passed_rules` at load time, so they always
  describe the roster being plotted rather than whatever the file was written with.
- `outcome` — `all_fail`, `some_fail`, or `all_pass`, relative to `MODELS`.
- `n_rules`, `source_len`.

An item counts as passed for a model only when that model satisfies every
verification rule written for the item.
