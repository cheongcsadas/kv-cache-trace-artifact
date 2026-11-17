"""Source provenance including genuine supplementary prompt material."""


def source_ids(row):
    primary=row.get("source_id")
    if not isinstance(primary,str) or not primary.strip():
        raise ValueError("A nonempty source_id is required")
    support=row.get("support_source_ids",[])
    if (not isinstance(support,list) or any(not isinstance(value,str) or not value.strip() for value in support)
            or len(support) != len(set(support)) or primary in support):
        raise ValueError("Supplementary source IDs must be distinct nonempty strings excluding the primary source")
    return {primary,*support}
