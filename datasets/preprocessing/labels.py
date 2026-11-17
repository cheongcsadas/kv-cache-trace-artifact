"""Validate explicit per-domain label vocabularies used by semantic tasks."""
from collections.abc import Mapping

DOMAINS=("medical","financial","legal","code")
CLASS_LIMITS={"intent":3,"attribute":60,"keyword":60}


def validate_label_vocabulary(value, *, class_counts=None):
    expected=CLASS_LIMITS if class_counts is None else class_counts
    if (not isinstance(expected,Mapping) or set(expected) != set(CLASS_LIMITS) or
            any(isinstance(n,bool) or not isinstance(n,int) or not 1 <= n <= CLASS_LIMITS[slot]
                for slot,n in expected.items())):
        raise ValueError("Class counts must declare positive intent/attribute/keyword counts within paper limits")
    if not isinstance(value,Mapping) or set(value) != set(DOMAINS):
        raise ValueError("Label vocabulary must declare all four domains")
    result={}
    for domain in DOMAINS:
        slots=value[domain]
        if not isinstance(slots,Mapping) or set(slots) != set(CLASS_LIMITS):
            raise ValueError("Each domain vocabulary requires intent, attribute and keyword")
        result[domain]={}
        for slot,limit in expected.items():
            labels=slots[slot]
            if (not isinstance(labels,(list,tuple)) or len(labels) != limit or
                    any(not isinstance(label,str) or not label.strip() for label in labels)):
                raise ValueError(f"{domain}/{slot} must list exactly {limit} nonempty string labels")
            if len(labels) != len(set(labels)):
                raise ValueError("Duplicate label in semantic vocabulary")
            result[domain][slot]=list(labels)
    return result
