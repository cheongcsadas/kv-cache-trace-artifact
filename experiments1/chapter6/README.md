# Chapter 6: KV-loading defenses

The two experiments use the three system/model pairs and short templated domain
inputs. Table 8 evaluates **MAA on the metadata channel**. Each system contributes
an undefended baseline, fixed-budget aware/oblivious evaluations, and timing
perturbation aware/oblivious evaluations: 15 conditions in total.

The same prepared prompts are used in every defense condition. Aware training uses
the same defense as testing; oblivious training uses the undefended training file
and the defended held-out test file. Both keep 3,000 training and 1,000 test prompts
per domain with disjoint core and supplementary sources. Failed test attempts
remain in attack-success denominators.

```bash
python -m experiments.e13_defense_accuracy --native-manifest native-project.json --output runs/e13
python -m experiments.e14_defense_latency --manifest runs/e13-native/manifest.json --output runs/e14 --plot
```

The native project has the same backend and `datasets` fields used by Chapter 5.
The runner constructs nine actual system/defense configurations and starts the
corresponding native runtime. The `experiment.defense` setting reaches the real
KV manager. Fixed-budget loading uses 65 pages of 16 tokens with actual shuffled
padding transfers; padding is excluded from attention and uses the public event
schema. Dynamic timing samples an independent `U(0,5 ms)` delay before eligible
loads. Quest/InfiniGen apply these paths during decode; RetrievalAttention applies
them during prefill. The runtime exports its actual transfer and allocation data.
Quest shares one budget across each layer/decode step. InfiniGen shares one per
KV head/layer/decode step across prefetch and all grouped-query demand batches.
RetrievalAttention loads the persistent context once per layer after prefill
attention. Internal streaming chunks do not restart the budget; timing draws
one delay before the first actual copy in each operation. Over-budget operations
retain all required transfers.

Completed measured observations must include native execution receipts. The
evaluator checks request identity, actual token and capacity values, selected-token
and generation settings, defense parameters, batch units, transfer counts and
delay counts. Fixed-budget operations transfer at least 65 pages and timing
operations record exactly one delay per eligible loading batch.

Figure 7 uses the server's native execution interval, excluding model startup,
HTTP transport and observer setup. It pairs the same original sample, source
set, rendered prompt digest and actual tokenizer length across the three loading
conditions. The model revision, cache, generation, probe and classifier settings
must also match. The summary records failed and unpaired attempts; latency means
and overhead use the common set of completed triplets. If no triplet completes,
the mean and overhead are `null`.

`e14 --native-manifest` can collect its own held-out test prompts. Reusing e13's
capture manifest avoids repeating inference. Figure 7 uses three system groups,
the three defense series, and y-axis ticks 4, 5, 6, 7 seconds. Values outside that
axis require `--auto-y`, so measurements are never silently clipped. All plotted
values and Table 8 cells come from computed results; published accuracy/latency
values are not inserted into the implementation.

Computed summaries are saved before optional plotting. An existing summary can
be rendered again, including with an expanded axis, without repeating inference:

```bash
python -m experiments.e14_defense_latency.plot --input runs/e14/summary.json --output runs/e14/expanded-figure --auto-y
```

Each capture in an imported manifest declares `system`, `defense`,
`setting: templated_short`, `channel: meta`, `config`, `collection`, and paths to
`train`/`test` JSONL files (`test` alone is accepted for latency). Every row retains
`original_sample_id`, `source_id`, `prompt_sha256`, `token_count`, `domain`,
`defense`, acquisition status and observation/native-receipt paths. The native
collector creates this manifest automatically. `--pilot --allow-synthetic` is
reserved for explicitly labelled small fixtures; native collection rejects it.
