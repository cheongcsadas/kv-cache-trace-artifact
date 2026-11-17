"""Assemble length-controlled prompts from authentic source spans.

Each curator is bound to one preassigned train/test partition. Supplementary
material comes from distinct source records in that partition; no text is
repeated to fill a budget. Core request and annotated evidence remain verbatim.
"""
import hashlib
from collections import OrderedDict, defaultdict
from threading import Lock
from .rendering import render_prompt


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _merge(spans):
    merged=[]
    for start,end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else: merged.append((start,end))
    return merged


class _TextCache:
    """Exact string-keyed LRU with both entry and retained-text limits."""
    def __init__(self, max_entries=512, max_chars=1048576):
        self.max_entries, self.max_chars = max_entries, max_chars
        self.values = OrderedDict()
        self.chars = 0
        self._lock = Lock()

    def get(self, text, compute):
        with self._lock:
            if text in self.values:
                self.values.move_to_end(text)
                return self.values[text]
        # Tokenization remains outside the lock; concurrent readers can progress.
        value = compute()
        if len(text) <= self.max_chars:
            with self._lock:
                if text in self.values:
                    self.values.move_to_end(text)
                    return self.values[text]
                self.values[text] = value
                self.chars += len(text)
                while len(self.values) > self.max_entries or self.chars > self.max_chars:
                    old, _ = self.values.popitem(last=False)
                    self.chars -= len(old)
        return value


class SourceCurator:
    def __init__(self, rows, tokenizer, *, seed=42, semantic=False):
        self.rows=sorted(rows,key=lambda row:_hash(f"{seed}:{row['id']}"))
        self.tokenizer=tokenizer
        self.semantic=semantic
        self.seed=seed
        self._counts = _TextCache()
        self._digests = _TextCache()
        self.groups=defaultdict(list)
        for row in self.rows:
            self.groups[self._group(row)].append(row)

    def _group(self,row):
        return (row["domain"],)+(tuple(row.get(slot) for slot in ("intent","attribute","keyword"))
                                if self.semantic else ())

    def count(self,text,template_id,record=None):
        rendered=render_prompt(text,template_id,record=record)
        return self._counts.get(rendered, lambda: len(self.tokenizer.encode(rendered,add_special_tokens=True)))

    def _digest(self,text):
        return self._digests.get(text, lambda: _hash(text))

    def _anchors(self,row):
        text=row["text"]
        request=row.get("request")
        if not request:
            paragraphs=[part.strip() for part in text.split("\n\n") if part.strip()]
            request=paragraphs[0] if row["domain"] == "medical" else paragraphs[-1]
        index=text.rfind(request)
        if index < 0: raise ValueError("Core request must be an actual substring of its source record")
        anchors=[(index,index+len(request))]
        if self.semantic:
            evidence=row.get("annotation",{}).get("evidence",{})
            for slot in ("intent","attribute","keyword"):
                item=evidence.get(slot)
                if isinstance(item,dict):
                    start,end=item.get("start"),item.get("end")
                    if (isinstance(start,bool) or isinstance(end,bool) or not isinstance(start,int) or
                            not isinstance(end,int) or not 0 <= start < end <= len(text) or
                            text[start:end] != item.get("text")):
                        raise ValueError("Semantic evidence must identify an exact original source span")
                    anchors.append((start,end))
                else:
                    label=row.get(slot,"")
                    position=text.casefold().find(label.casefold()) if label else -1
                    if position < 0:
                        raise ValueError("Semantic length curation requires grounded source-span evidence")
                    anchors.append((position,position+len(label)))
        return _merge(anchors)

    def _provenance(self,row,spans):
        return [dict(record_id=row["id"],source_id=row["source_id"],start=start,end=end,
                     sha256=self._digest(row["text"][start:end])) for start,end in spans]

    def curate(self,row,target,template_id,tolerance=.1):
        text=row["text"]
        lower,upper=round(target*(1-tolerance)),round(target*(1+tolerance))
        count=self.count(text,template_id,row)
        spans=[(0,len(text))]
        anchors=self._anchors(row) if self.semantic else None
        if count > upper:
            anchors=anchors or self._anchors(row)
            def expanded(margin):
                selected=_merge([(max(0,start-margin),min(len(text),end+margin)) for start,end in anchors])
                return selected,"\n\n".join(text[start:end] for start,end in selected)
            spans,core=expanded(0)
            if self.count(core,template_id,row) > upper: return None
            # Retain as much genuine material around the protected spans as the
            # tokenizer budget permits. A fixed 256-character margin can discard
            # a rare class even when its original document is amply long enough.
            lo,hi=0,len(text)
            while lo < hi:
                mid=(lo+hi+1)//2
                _,candidate=expanded(mid)
                if self.count(candidate,template_id,row) <= target: lo=mid
                else: hi=mid-1
            spans,core=expanded(lo)
        else: core=text
        fragments=[]
        support=[]
        primary_provenance=self._provenance(row,spans)
        provenance=[]
        used_sources={row["source_id"]}
        used_text={self._digest(text),self._digest(core)}
        count=self.count(core,template_id,row)
        candidates=self.groups[self._group(row)]
        start=int(_hash(f"{self.seed}:{row['id']}")[:16],16)%len(candidates)
        for offset in range(len(candidates)):
            if count >= lower: break
            candidate=candidates[(start+offset)%len(candidates)]
            raw=candidate["text"]
            if candidate["source_id"] in used_sources: continue
            raw_digest=self._digest(raw)
            if raw_digest in used_text: continue
            used_sources.add(candidate["source_id"])
            def combined(prefix): return "\n\n".join(fragments+[prefix,core])
            # Binary search an exact source prefix against the real tokenizer.
            lo,hi=0,len(raw)
            while lo < hi:
                mid=(lo+hi+1)//2
                if self.count(combined(raw[:mid]),template_id,row) <= target: lo=mid
                else: hi=mid-1
            prefix=raw[:lo].rstrip()
            if lo < len(raw):
                boundary=max(prefix.rfind(" "),prefix.rfind("\n"))
                if boundary > 0: prefix=prefix[:boundary].rstrip()
            prefix_digest=self._digest(prefix)
            if not prefix or prefix_digest in used_text: continue
            fragments.append(prefix)
            support.append(candidate["source_id"])
            provenance.extend(self._provenance(candidate,[(0,len(prefix))]))
            used_text.add(raw_digest)
            used_text.add(prefix_digest)
            count=self.count("\n\n".join(fragments+[core]),template_id,row)
        if not lower <= count <= upper: return None
        return dict(text="\n\n".join(fragments+[core]),support_source_ids=support,
                    source_spans=provenance+primary_provenance,curation="partition_source_spans")
