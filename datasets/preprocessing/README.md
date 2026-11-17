# Dataset acquisition and preparation

`configs/datasets.yaml` pins the four corpus repositories to immutable commits.
The downloader verifies individual Git blobs, records content hashes, retains
upstream license files, and checks archive paths before extraction. CodeXGLUE
uses the pinned CONCODE text-to-code subset with its original `nl` and `code` fields.
CONCODE's structural separators delimit the natural-language request from class
context. Intent rules and protected request spans use that request; normalization
preserves the complete source text and its stable identifiers.

```bash
python -m datasets.acquire --catalog configs/datasets.yaml --output data/public
```

Preparation counts the exact rendered input using each model's local tokenizer.
The full workflow below prepares **3 systems × 5 settings × 4 domains = 60 domain
conditions**, using the native tokenizer environment of each system. Every domain
condition contains 3,000 training and 1,000 test prompts. Its profiling training
file contains 3,200 prompts, including the same main-training records, for the
sample-count experiment. Additional outputs cover the 1K/2K/3K/4K context sweep
alongside the five-setting matrix. The auxiliary
set contains 1,000 distinct source records in total and uses source groups excluded from both
attack-training and test pools. `project_data.json` supplies the `datasets`,
`contexts`, `profiling`, `transfer_test`, and `auxiliary` entries for the live experiment project.
Merge these entries into the project used by the Chapter 5 native runner.
`scripts/build_native_project.py` combines this data file with the three executable
backend configs and Python environment paths. The transfer-contribution test file
selects 250 held-out prompts per domain, giving 1,000 total.

Short and long prompts target a mean of approximately 1,000 and 3,000 tokens as
specified in the paper. Cross-template prompts use a 1,000-token mean target as
an artifact choice; the paper specifies template holdout without a separate fixed
length for that condition. The default length target tolerance is ±10%, also an
explicit curation choice rather than a paper-reported value.
Selection prioritizes prompts in that interval. Semantic coverage can retain a
shorter genuine source record when needed for a rare class; the final selected
training, profiling and test sets must each meet the per-domain tokenizer mean
target. `--curate-lengths` selects verbatim spans around the original
request and its semantic evidence when a source is too long. For shorter inputs,
it adds authentic supplementary material from distinct sources in the same fixed
partition and domain. Semantic supplements additionally share all three target
labels. Core request wording remains intact, and no text is repeated to fill a
budget. Every retained span records its source, original character offsets, and
hash in prompt order; `support_source_ids` participates in train/test isolation.
All domains use the same composition rule and shared template text.

`rendering.render_prompt(text, template_id, record)` is shared by curation, final
preparation, and the standalone render command. Unstructured prompts use six
natural connective wordings selected deterministically by source record ID,
without named prompt fields or inserted domain/semantic labels. These are the
artifact's free-wording construction, not the paper's original templates.
CONCODE class-context separators become natural connective text while the original
request and retained source code remain intact. Generated connectives are separate
from `source_spans`, whose offsets and hashes refer to corpus text. Both the
connectives and source material are included in the final tokenizer count.

This explicit source-curation procedure uses the supplied corpus and tokenizer.
If eligible source material cannot satisfy a quota, preparation reports the domain
and available count before writing output. To use independently curated variants,
provide `--prompt-variants variants.json`:

```json
{"normalized-record-id": {"templated_long": "source-derived material", "unstructured_long": "free-form source-derived request", "context_4000": "source-derived material"}}
```

Templated variant text is inserted into the shared shell; unstructured variants
use the same free-wording renderer as curated source material. Cross-template
training uses `shared-v1` and testing uses `shared-v2`, with the same background
instruction across domains. Source IDs are retained across all variants and tokenizers.
Explicit variants take priority over automatic curation. Their ID, setting, text
type, and rendered token counts are checked. Semantic variants must retain the
exact original evidence for every target slot; removing that evidence rejects
the variant before prepared files are written.
The complete original request is also retained and bound to the prompt. Request-level
exclusions prevent a comparison in the surrounding material from assigning an
incompatible target to that request.

The two training tasks use separate targets:

- Domain inference (MAA/SAA) uses only `domain`. It can use the full prepared
  matrix while ignoring semantic targets, or a separate domain-only preparation.
- Semantic inference (MSA/SSA) uses `domain`, `intent`, `attribute`, and one
  `keyword` label. The single-environment preparation command accepts
  `--annotations labels.json --semantic`, with
  `--label-vocabulary label_vocabulary.json` for explicit label lists. The full matrix script below
  always uses semantic targets.

The annotation map joins normalized IDs to the three semantic slots. Its schema
is `annotations.schema.json`; a keyword containing several words remains one
string label. `configs/semantic_labels.yaml` contains exactly **3 intent, 60
sensitive-attribute and 60 keyword classes for each domain**. Every category has
a concrete description, and every attribute is tied to a grounded keyword rule.
Medical attributes describe health properties, financial attributes describe
financial exposures or measures, legal attributes describe contractual rights and
obligations, and code attributes describe computational operations or resources.

[Label evidence](label_evidence.json) gives one actual normalized source record
for each of the 240 keyword/property pairs, with corpus revision, record/source
IDs, source-text digest and exact evidence spans. Attribute targets are explicit
semantic rules associated with the grounded keyword; they are not quotations of
attribute class names from the source. Source hashes can be checked after running
the pinned downloader and normalizer.

```bash
python -m datasets.preprocessing.verify_evidence --input data/public/medquad/normalized.jsonl data/public/finqa/normalized.jsonl data/public/cuad/normalized.jsonl data/public/codexglue/normalized.jsonl
```

`label_vocabulary.json` has the shape `{domain: {intent: [...], attribute: [...],
keyword: [...]}}`, using all four domain names and exactly 3/60/60 distinct labels.
Its declared classes stay fixed through source splitting, tokenizer length
selection, acquisition, classifier training and evaluation. All 3/60/60 classes
must occur in each selected main-training and profiling condition; testing labels
must belong to that trained vocabulary, without requiring every test condition
to contain all 60 attribute or keyword classes. Coverage failure stops preparation before output is
written. No labels or source records are fabricated to fill the counts.

Source splitting first protects a deterministic set of existing source groups
that covers the grounded training vocabulary. Auxiliary selection excludes those
groups, and the remaining source groups follow the 75/25 target before exact
training/test quotas are selected. This preserves rare classes without moving
records from a held-out source into training.

`datasets.preprocessing.grounding` verifies original character spans, class-specific
text rules and the keyword-to-attribute relation. Prepared rows carry an annotation
receipt bound to the source, the D/I/A/K targets, the fixed taxonomy, and the actual
rendered prompt. Acquisition retains the receipt as reference metadata; MSA/SSA
check it again before training. These reference fields are excluded from attack
features. Custom annotation labels must agree with the repository's fixed taxonomy;
inconsistent label names, taxonomy digests or text evidence fail before output.

The vocabulary is carried in `project_data.json` through the native project and
attack training configuration. Keep the domain and semantic prepared directories
separate when their selected source pools differ. Domain records without semantic
matches remain available for the domain inference task.

## Full preparation matrix

The included rule labeler matches source text and records exact character-span
evidence for every assigned slot. Prepare all three systems using their separate
native Python environments:

```bash
python -m datasets.preprocessing.annotate \
  --input data/public/medquad/normalized.jsonl data/public/finqa/normalized.jsonl data/public/cuad/normalized.jsonl data/public/codexglue/normalized.jsonl \
  --output data/semantic_labels
python scripts/prepare_matrix.py \
  --input data/public/medquad/normalized.jsonl data/public/finqa/normalized.jsonl data/public/cuad/normalized.jsonl data/public/codexglue/normalized.jsonl \
  --annotations data/semantic_labels/annotations.json \
  --label-vocabulary data/semantic_labels/label_vocabulary.json \
  --python quest=/venvs/quest/bin/python \
  --python infinigen=/venvs/infinigen/bin/python \
  --python retrievalattention=/venvs/retrievalattention/bin/python \
  --tokenizer quest=/models/Llama-2-7b-hf \
  --tokenizer infinigen=/models/Mistral-7B-v0.1 \
  --tokenizer retrievalattention=/models/Qwen2.5-7B-Instruct \
  --output prepared/matrix
```

The matrix script always prepares semantic targets and enables source-length
curation by default. It accepts `--prompt-variants`, `--seed` (default 42), and
`--length-tolerance` (default 0.1); `--no-curate-lengths` selects already suitable
source inputs. Each system runs in its own process. The final merge requires all
15 system/setting pairs, all four domains, matching source-partition summaries,
exact sample quotas, per-domain token means, and complete training vocabularies.
Tokenizer preparation loads local tokenizer/configuration assets without model
weights or GPU inference.

Use the following assets and the Transformers version locked by each native
backend's `requirements.txt`. `datasets.preprocessing.matrix` verifies the
configuration and tokenizer file hashes against these immutable model revisions.
These revision and software-version choices pin this artifact's implementation.

| System | Official tokenizer/model assets | Asset revision | Transformers |
|---|---|---|---|
| Quest | `meta-llama/Llama-2-7b-hf` | `01c7f73d771dfac7d292323805ebc428287df4f9` | `4.45.2` |
| InfiniGen | `mistralai/Mistral-7B-v0.1` | `27d67f1b5f57dc0953326b2601d68371d40ea8da` | `4.57.1` |
| RetrievalAttention | `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | `4.47.1` |

The merged `prepared/matrix/project_data.json` uses absolute paths to the child
outputs. Alongside the dataset paths and fixed label vocabulary, it contains:

- `tokenizers`: model identity, asset hashes, dependency versions, special-token
  settings, and a tokenizer behavior fingerprint for each system.
- `conditions`: the 15 system/setting entries, each with four-domain train, test,
  and profiling counts, class sets, and measured token statistics.
- `source_partition`: the common source-group split, including the seed and
  per-domain partition counts and digests, before tokenizer-dependent selection.

These fields are data-generation metadata saved with each prepared dataset and
used to bind later collection and training to the same inputs. Source IDs remain
shared across tokenizers, and primary plus supplementary sources are disjoint
between training/profiling and testing. Generated connective text is included in
prompt hashes and token counts, while source spans retain their original offsets.

For a single environment, `python -m datasets.preprocessing.prepare` accepts
`--semantic --annotations ... --label-vocabulary ... --tokenizer system=...` with
`--curate-lengths`. Its `--audit-vocabulary` checks every declared class against
the eligible training records for every selected model and length condition.
`--fit-vocabulary` is a compatibility alias for this same audit and never reduces
a class list. Test records cannot define or remove classes. The annotation command writes the fixed
vocabulary alongside separate `observed_class_counts`; these distinguish declared
class definitions from actual grounded annotations. Optional `annotate --fit-input`
checks that a supplied training-only pool covers all declared classes.

## Individual local operations

The preprocessing package reads local copies of MedQuAD XML, FinQA JSON, CUAD
SQuAD-style JSON, and CodeXGLUE JSONL. The individual commands below operate on
local copies. Run commands from the repository root.

```bash
python -m datasets.preprocessing normalize --dataset medquad --source /data/MedQuAD --output prepared/medical.jsonl
python -m datasets.preprocessing normalize --dataset finqa --source /data/FinQA/train.json --output prepared/financial.jsonl
python -m datasets.preprocessing normalize --dataset cuad --source /data/CUAD_v1.json --output prepared/legal.jsonl
python -m datasets.preprocessing normalize --dataset codexglue --source /data/code.jsonl --output prepared/code.jsonl
python -m datasets.preprocessing split --input prepared/medical.jsonl prepared/financial.jsonl prepared/legal.jsonl prepared/code.jsonl --output-dir prepared/split --train-per-domain 3000 --test-per-domain 1000 --seed 42
python -m datasets.preprocessing render --input prepared/split/train.jsonl --output prepared/train.jsonl --template-id shared-v1
python -m datasets.preprocessing render --input prepared/split/test.jsonl --output prepared/test.jsonl --template-id shared-v2
```

The last two commands illustrate cross-template holdout. Use the same template ID
for ordinary templated evaluation and `unstructured` for the free-wording renderer
that retains the original request.
The three included templates use the same background and field layout across all
four domains; they do not insert domain labels. Freeze the selected template text,
source selection, seed, annotation vocabulary and prompt-length rules for each run.

Normalized records contain `id`, `source_id`, `domain`, and `text`. Medical
questions from the same source document and legal questions from the same contract
are grouped. FinQA uses its stable source-record ID. CodeXGLUE uses URL/source ID,
falling back to a digest of source code. The fallback does not detect near-duplicate
code or cross-document paraphrases; deduplicate those before normalization when
applicable. Source paths are not embedded in output records.

An optional `--annotations labels.json` joins explicitly provided labels by the
normalized record ID. Normalize once to establish these IDs, then provide a map
such as `{"finqa:...": {"intent": "...", "attribute": "...", "keyword": "..."}}`.
The keyword is a single label. The semantic vocabulary file names the actual
intent, attribute, and keyword classes used for training and evaluation.

Splitting retains underlying sources in only one partition. Explicit quotas produce
exactly 3,000 training and 1,000 test records per domain or fail with a clear error;
surplus records are omitted and counted in the report. Without quotas all records
are kept, and the 75/25 ratio is approximate because source groups are indivisible.
The default seed 42 is an implementation choice, not a reported paper parameter.
Existing output files and split directories are never overwritten.

`render --local-tokenizer /models/tokenizer --min-tokens 900 --max-tokens 1100`
counts actual tokenizer tokens, including that tokenizer's special tokens, and
rejects records outside the bounds. This requires the optional `transformers`
package and loads local files only. No character-count approximation, truncation,
or padding is used. The publication's approximate 1K/3K lengths are targets for
curation; these bounds do not manufacture longer text. All three native backends
encode the complete rendered prompt with special tokens and do not apply an
additional chat template. Before collection, the native Chapter 5 runner calls
`describe` for every prompt, checks its prepared token count, and validates the
actual mean separately for every train/test split and domain, including
cross-template inputs. This prevents one domain's excess length from masking
another's shortfall. CodeXGLUE support covers local JSONL exports with `code`/`func` and/or
`docstring`/`text`/`nl`; other task distributions must be converted explicitly.

Python APIs are exported by `datasets.preprocessing`. Call
`assert_template_disjoint(train, test)` to validate both template and source
holdout after rendering a cross-template evaluation set.
