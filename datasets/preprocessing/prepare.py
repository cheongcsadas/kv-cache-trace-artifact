"""Prepare tokenizer-measured corpus inputs for the live Chapter 5 runners."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from .core import DOMAINS, SLOTS, TEMPLATES, read_jsonl, split_records, validate_records, write_jsonl
from .labels import validate_label_vocabulary
from .curation import SourceCurator
from .grounding import validate_grounded_record
from .rendering import render_prompt
from .sources import source_ids

SETTINGS = {"templated_short": (1000, "shared-v1"), "templated_long": (3000, "shared-v1"),
            "unstructured_short": (1000, "unstructured"), "unstructured_long": (3000, "unstructured"),
            "cross_template": (1000, "shared-v1")}
CLASS_COUNTS = {"intent": 3, "attribute": 60, "keyword": 60}


def fit_common_vocabulary(vocabulary, eligible_combinations):
    """Audit every declared class on each eligible training condition.

    The paper vocabulary is immutable. Missing training coverage rejects the
    condition rather than shrinking its class set or adding synthetic records.
    """
    vocabulary=validate_label_vocabulary(vocabulary)
    if not eligible_combinations:
        raise ValueError("Training eligibility conditions are required")
    for index,condition in enumerate(eligible_combinations):
        for domain in DOMAINS.values():
            triples=condition[domain]
            for position,slot in enumerate(SLOTS):
                observed={values[position] for values in triples}
                expected=set(vocabulary[domain][slot])
                if observed != expected:
                    raise ValueError(f"Training condition {index}, {domain}/{slot} must cover all "
                        f"{len(expected)} declared classes; missing={sorted(expected-observed)}, "
                        f"unknown={sorted(observed-expected)}")
    return vocabulary


def join_annotations(rows, annotations, require_complete=False):
    if not isinstance(annotations, dict):
        raise ValueError("Annotations must be an object mapping normalized IDs to semantic slots")
    known = {row["id"] for row in rows}
    if set(annotations) - known:
        raise ValueError("Annotations contain IDs absent from the normalized corpora")
    result = []
    for row in rows:
        label = annotations.get(row["id"], {})
        if not isinstance(label, dict) or set(label) - set(SLOTS):
            raise ValueError("Annotation fields must be intent, attribute and keyword")
        if any(not isinstance(value, str) or not value.strip() for value in label.values()):
            raise ValueError("Semantic slots must be nonempty single-label strings")
        if require_complete and label and set(label) != set(SLOTS):
            raise ValueError("Each semantic annotation must supply intent, attribute and keyword")
        if any(slot in row and (not isinstance(row[slot],str) or not row[slot].strip()) for slot in SLOTS):
            raise ValueError("Normalized semantic slots must be nonempty single-label strings")
        result.append(dict(row, **label))
    return result


def measured_candidates(rows, tokenizer, setting, target, template_id, tolerance,
                        variants=None, semantic=False, curator=None):
    lower, upper = round(target*(1-tolerance)), round(target*(1+tolerance))
    variants = variants or {}
    result = []
    for row in rows:
        if semantic and any(slot not in row for slot in SLOTS):
            continue
        supplied=variants.get(row["id"], {})
        text = supplied.get(setting, row["text"])
        source_metadata={}
        if curator is not None and setting not in supplied:
            curated=curator.curate(row,target,template_id,tolerance)
            if curated is None:
                if not semantic: continue
                # Short original records may be necessary to retain a rare
                # grounded class. Quota selection validates the condition mean.
                text=row["text"]
            else:
                text=curated["text"]
                source_metadata={key:value for key,value in curated.items() if key != "text"}
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Prompt variants must contain nonempty source-derived text")
        prompt = render_prompt(text, template_id, record=row)
        count = len(tokenizer.encode(prompt, add_special_tokens=True))
        if count <= upper and (count >= lower or semantic):
            labels = {"domain": row["domain"], **{slot: row[slot] for slot in SLOTS if slot in row}}
            if semantic:
                source_metadata["annotation"] = validate_grounded_record(row, prompt=prompt)
            result.append(dict(row, **source_metadata, prompt=prompt, token_count=count, context_tokens=count,
                target_tokens=target, template_id=template_id, setting=setting,
                length_band_eligible=count >= lower, length_tolerance=tolerance,
                token_count_scope="rendered_prompt_with_tokenizer_special_tokens",
                labels=labels, sample_id=row["id"], prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
    return result


def select_quota(rows, count, seed, semantic=False, label_vocabulary=None, require_coverage=True,
                 domains=None):
    if isinstance(count,bool) or not isinstance(count,int) or count < 1:
        raise ValueError("Per-domain quotas must be positive integers")
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)
    selected = []
    selected_domains = tuple(DOMAINS.values()) if domains is None else tuple(domains)
    if (not selected_domains or len(set(selected_domains)) != len(selected_domains)
            or set(selected_domains) - set(DOMAINS.values())):
        raise ValueError("Quota domains must be distinct recognized paper domains")
    for domain in selected_domains:
        candidates = sorted(by_domain[domain], key=lambda r: r["id"])
        random.Random(f"{seed}:{domain}").shuffle(candidates)
        # Keep the bulk of the condition at its target length; use shorter real
        # inputs only when class coverage or the quota requires them.
        candidates.sort(key=lambda row: not row.get("length_band_eligible", True))
        if len(candidates) < count:
            raise ValueError(f"{domain}: need {count} eligible prompts, found {len(candidates)}; "
                             "supply length-appropriate corpus records or curated prompt variants")
        required, used = [], set()
        if semantic:
            vocabulary = {slot: {r[slot] for r in candidates} for slot in SLOTS}
            if label_vocabulary is not None:
                if any(vocabulary[slot]-set(label_vocabulary[domain][slot]) for slot in SLOTS):
                    raise ValueError("Semantic candidates contain labels outside the declared vocabulary")
                if require_coverage and any(vocabulary[slot] != set(label_vocabulary[domain][slot]) for slot in SLOTS):
                    raise ValueError(f"{domain}: eligible semantic candidates must cover the declared label vocabulary")
            elif require_coverage and any(len(vocabulary[slot]) != CLASS_COUNTS[slot] for slot in SLOTS):
                raise ValueError(f"{domain}: semantic candidates must contain exactly 3/60/60 classes")
            covered = {slot: set() for slot in SLOTS}
            for index, row in enumerate(candidates if require_coverage else []):
                if any(row[slot] not in covered[slot] for slot in SLOTS):
                    required.append(row)
                    used.add(index)
                    for slot in SLOTS:
                        covered[slot].add(row[slot])
            if len(required) > count:
                raise ValueError("Quota cannot retain semantic vocabulary coverage")
        required += [row for i, row in enumerate(candidates) if i not in used][:count-len(required)]
        if required and all("target_tokens" in row for row in required):
            targets={row["target_tokens"] for row in required}
            tolerances={row["length_tolerance"] for row in required}
            if len(targets) != 1 or len(tolerances) != 1:
                raise ValueError("A selected condition must use one token target and tolerance")
            target,tolerance=next(iter(targets)),next(iter(tolerances))
            mean=sum(row["token_count"] for row in required)/len(required)
            if abs(mean-target) > target*tolerance:
                raise ValueError(f"{domain}: selected tokenizer mean {mean:.1f} does not meet target {target}")
        selected.extend(required)
    return selected


def semantic_training_sources(rows, seed=42):
    """Reserve existing source groups that cover all grounded semantic classes."""
    protected = set()
    for domain in DOMAINS.values():
        groups = defaultdict(set)
        for row in rows:
            if row["domain"] == domain and all(slot in row for slot in SLOTS):
                groups[row["source_id"]].update((slot, row[slot]) for slot in SLOTS)
        uncovered = set().union(*groups.values()) if groups else set()
        rank = lambda source: hashlib.sha256(f"{seed}:{source}".encode()).hexdigest()
        while uncovered:
            source = min(groups, key=lambda key: (-len(groups[key] & uncovered), rank(key)))
            protected.add(source)
            uncovered -= groups.pop(source)
    return protected


def prepare(records, tokenizers, output, *, annotations=None, variants=None, semantic=False,
            tolerance=.1, seed=42, train_per_domain=3000, test_per_domain=1000,
            profiling_per_domain=3200, auxiliary_per_domain=250,
            settings=None, context_targets=(1000, 2000, 3000, 4000), label_vocabulary=None,
            curate_lengths=False, fit_vocabulary=False, require_complete_matrix=False):
    from .matrix import describe_tokenizer, summarize_condition, validate_complete_matrix
    if not 0 <= tolerance < 1:
        raise ValueError("Token-length tolerance must be in [0,1)")
    for count in (train_per_domain,test_per_domain,profiling_per_domain,auxiliary_per_domain):
        if isinstance(count,bool) or not isinstance(count,int) or count < 1:
            raise ValueError("Per-domain quotas must be positive integers")
    if not tokenizers or set(tokenizers) - {"quest","infinigen","retrievalattention"}:
        raise ValueError("Supply at least one named system tokenizer")
    settings = list(SETTINGS) if settings is None else list(settings)
    if len(settings) != len(set(settings)) or set(settings) - set(SETTINGS):
        raise ValueError("Prompt settings must be unique recognized conditions")
    context_targets = list(context_targets)
    if (len(context_targets) != len(set(context_targets)) or
            any(isinstance(value,bool) or value not in (1000,2000,3000,4000) for value in context_targets)):
        raise ValueError("Context conditions must be unique 1K/2K/3K/4K targets")
    if not settings and not context_targets:
        raise ValueError("Select at least one prompt setting or context condition")
    if require_complete_matrix and (not semantic or set(tokenizers) != {"quest", "infinigen", "retrievalattention"}
                                   or set(settings) != set(SETTINGS)):
        raise ValueError("Complete semantic matrix requires three systems and all five settings")
    tokenizer_descriptions = {system: describe_tokenizer(tokenizer, system=system)
                             for system, tokenizer in tokenizers.items()}
    if profiling_per_domain < train_per_domain:
        raise ValueError("Profiling pool must cover the main training quota")
    rows = [dict(row) for row in records]
    validate_records(rows)
    rows = join_annotations(rows, {} if annotations is None else annotations, require_complete=semantic)
    if label_vocabulary is not None:
        label_vocabulary=validate_label_vocabulary(label_vocabulary)
        for row in rows:
            for slot in SLOTS:
                if slot in row and row[slot] not in label_vocabulary[row["domain"]][slot]:
                    raise ValueError("Annotation label is outside the declared semantic vocabulary")
    if semantic and not annotations:
        raise ValueError("Semantic preparation requires an explicit annotation map")
    if semantic:
        for row in rows:
            if all(slot in row for slot in SLOTS):
                row["annotation"] = validate_grounded_record(row)
    if fit_vocabulary and (not semantic or label_vocabulary is None):
        raise ValueError("Coverage auditing requires semantic preparation and explicit label lists")
    if variants and set(variants) - {row["id"] for row in rows}:
        raise ValueError("Prompt variants contain unknown normalized IDs")
    if variants is not None:
        allowed=set(SETTINGS)|{"auxiliary","context_1000","context_2000","context_3000","context_4000"}
        if not isinstance(variants,dict) or any(not isinstance(value,dict) or set(value)-allowed or
                any(not isinstance(text,str) or not text.strip() for text in value.values())
                for value in variants.values()):
            raise ValueError("Prompt variants must map record IDs to recognized conditions and nonempty source-derived text")
    # Reserve whole source groups before splitting the remaining corpus. Auxiliary
    # observation-quality inputs cannot leak into attack training or held-out tests.
    representatives = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        representatives.setdefault(row["source_id"], row)
    protected_sources = semantic_training_sources(rows, seed) if semantic else set()
    auxiliary = select_quota([row for key,row in representatives.items() if key not in protected_sources],
                             auxiliary_per_domain, seed+1)
    auxiliary_sources = {r["source_id"] for r in auxiliary}
    pools = split_records([r for r in rows if r["source_id"] not in auxiliary_sources],
                          seed=seed, train_fraction=.75, required_train_sources=protected_sources)
    if fit_vocabulary:
        # Check fixed source-disjoint training conditions; classes never change.
        eligible=[]
        fit_conditions=[(setting,*SETTINGS[setting]) for setting in settings]
        fit_conditions += [(f"context_{target}",target,"shared-v1") for target in context_targets]
        for tokenizer in tokenizers.values():
            curator=SourceCurator(pools["train"],tokenizer,seed=seed,semantic=True) if curate_lengths else None
            for setting,target,template in fit_conditions:
                candidates=measured_candidates(pools["train"],tokenizer,setting,target,template,
                                                tolerance,variants,True,curator)
                eligible.append({domain:{tuple(row[slot] for slot in SLOTS) for row in candidates
                    if row["domain"] == domain} for domain in DOMAINS.values()})
        label_vocabulary=fit_common_vocabulary(label_vocabulary,eligible)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    bridge = dict(datasets={}, contexts={}, profiling={}, auxiliary={}, transfer_test={},
                  semantic_annotations=bool(semantic), tokenizers=tokenizer_descriptions, conditions=[])
    # This describes the common source partition, before tokenizer-dependent
    # selection, so independently prepared native environments can be compared.
    bridge["source_partition"] = dict(seed=seed, train_fraction=.75, domains={})
    for domain in DOMAINS.values():
        partitions = {**pools, "auxiliary": auxiliary}
        bridge["source_partition"]["domains"][domain] = {}
        for split, values in partitions.items():
            sources = sorted(set().union(*(source_ids(row) for row in values if row["domain"] == domain)))
            bridge["source_partition"]["domains"][domain][split] = dict(source_count=len(sources),
                sha256=hashlib.sha256(json.dumps(sources, separators=(",", ":")).encode()).hexdigest())
    if label_vocabulary is not None:
        bridge["label_vocabulary"]=label_vocabulary
    planned = []
    for system, tokenizer in tokenizers.items():
        curators={split:SourceCurator(pool,tokenizer,seed=seed,semantic=semantic) for split,pool in pools.items()} if curate_lengths else {}
        bridge["datasets"][system], bridge["contexts"][system], bridge["profiling"][system] = {}, {}, {}
        conditions = [(setting, *SETTINGS[setting]) for setting in settings]
        conditions += [(f"context_{target}", target, "shared-v1") for target in context_targets]
        for setting, target, shell in conditions:
            candidates = {}
            for split, pool in pools.items():
                template = "shared-v2" if setting == "cross_template" and split == "test" else shell
                candidates[split] = measured_candidates(pool, tokenizer, setting, target, template,
                                                       tolerance, variants, semantic, curators.get(split))
            train = select_quota(candidates["train"], profiling_per_domain, seed, semantic, label_vocabulary)
            test = select_quota(candidates["test"], test_per_domain, seed+2, semantic, label_vocabulary, require_coverage=False)
            if semantic:
                for domain in DOMAINS.values():
                    for slot in SLOTS:
                        if ({r[slot] for r in test if r["domain"] == domain}
                                - {r[slot] for r in train if r["domain"] == domain}):
                            raise ValueError("Test semantic vocabulary is absent from training")
            main_train = select_quota(train, train_per_domain, seed, semantic, label_vocabulary)
            summary = summarize_condition(main_train, test, train, system=system, setting=setting,
                vocabulary=label_vocabulary if semantic else None, train_per_domain=train_per_domain,
                test_per_domain=test_per_domain, profiling_per_domain=profiling_per_domain,
                tolerance=tolerance)
            summary["tokenizer_fingerprint"] = tokenizer_descriptions[system]["fingerprint"]
            if not setting.startswith("context_"):
                bridge["conditions"].append(summary)
            directory = output/system/setting
            pair = dict(train=str(directory/"train.jsonl"), test=str(directory/"test.jsonl"))
            planned += [(directory/"train.jsonl", main_train), (directory/"test.jsonl", test)]
            if setting.startswith("context_"):
                bridge["contexts"][system][str(target)] = pair
            else:
                bridge["datasets"][system][setting] = pair
                bridge["profiling"][system][setting] = dict(train=str(directory/"profiling_train.jsonl"), test=pair["test"])
                planned.append((directory/"profiling_train.jsonl", train))
                if setting == "templated_short":
                    transfer_path = directory / "transfer_test.jsonl"
                    # Section 5.4 uses 1,000 total, rather than 1,000 per domain.
                    transfer = select_quota(test, min(250, test_per_domain), seed+3)
                    planned.append((transfer_path, transfer))
                    bridge["transfer_test"][system] = str(transfer_path)
        auxiliary_output = []
        for row in auxiliary:
            text = (variants or {}).get(row["id"], {}).get("auxiliary", row["text"])
            prompt = render_prompt(text, "unstructured", record=row)
            count = len(tokenizer.encode(prompt, add_special_tokens=True))
            auxiliary_output.append(dict(row, sample_id=row["id"], prompt=prompt, token_count=count,
                                         context_tokens=count, template_id="unstructured", setting="auxiliary"))
        auxiliary_path = output/system/"auxiliary.jsonl"
        planned.append((auxiliary_path, auxiliary_output))
        bridge["auxiliary"][system] = str(auxiliary_path)
    # Validate every condition before creating any prepared output.
    if require_complete_matrix:
        bridge["matrix"] = validate_complete_matrix(bridge)
    output.mkdir(parents=True)
    for path, values in planned:
        write_jsonl(path, values)
    if label_vocabulary is not None:
        (output/"label_vocabulary.json").write_text(json.dumps(label_vocabulary,indent=2)+"\n",encoding="utf-8")
    (output/"project_data.json").write_text(json.dumps(bridge, indent=2)+"\n", encoding="utf-8")
    return bridge


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--tokenizer", action="append", required=True, help="system=/local/tokenizer (one per backend)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotations")
    parser.add_argument("--label-vocabulary",help="Explicit per-domain intent/attribute/keyword label lists")
    parser.add_argument("--audit-vocabulary", "--fit-vocabulary",dest="fit_vocabulary",action="store_true",
                        help="Audit full 3/60/60 class coverage across every selected training condition")
    parser.add_argument("--prompt-variants")
    parser.add_argument("--curate-lengths",action="store_true",help="Select and combine genuine source spans within each fixed partition")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--require-complete-matrix", action="store_true",
                        help="Require three real tokenizers, all five conditions and complete per-domain semantic quotas")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length-tolerance", type=float, default=.1)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizers = {}
    for item in args.tokenizer:
        system, path = item.split("=", 1)
        if system not in {"quest", "infinigen", "retrievalattention"} or system in tokenizers:
            raise ValueError("Provide each named system tokenizer once")
        if not Path(path).is_dir():
            raise ValueError("Tokenizer must be an existing local directory")
        tokenizers[system] = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8")) if path else None
    result = prepare([row for path in args.input for row in read_jsonl(path)], tokenizers, args.output,
        annotations=load(args.annotations), variants=load(args.prompt_variants), semantic=args.semantic,
        tolerance=args.length_tolerance, seed=args.seed, label_vocabulary=load(args.label_vocabulary),
        curate_lengths=args.curate_lengths,fit_vocabulary=args.fit_vocabulary,
        require_complete_matrix=args.require_complete_matrix)
    print(json.dumps({"project_data": str(Path(args.output).resolve()/"project_data.json"),
                      "systems": list(result["datasets"])}))


if __name__ == "__main__":
    main()
