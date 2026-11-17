"""Native FAISS implementation of MInference's per-KV-head FlatIndex API."""
import numpy as np


class FlatIndex:
    def __init__(self, head_num, dim):
        import faiss
        self.head_num, self.dim = int(head_num), int(dim)
        self.indices = [faiss.IndexFlatIP(self.dim) for _ in range(self.head_num)]

    @property
    def ntotal(self):
        counts = {index.ntotal for index in self.indices}
        if len(counts) != 1:
            raise RuntimeError("KV-head index lengths disagree")
        return counts.pop()

    def paraadd(self, keys, core):
        import faiss
        faiss.omp_set_num_threads(max(1, int(core)))
        keys = np.asarray(keys, dtype=np.float32)
        if keys.ndim != 3 or keys.shape[0] != self.head_num or keys.shape[-1] != self.dim:
            raise ValueError("Expected [KV heads, tokens, head dimension]")
        for head, index in enumerate(self.indices):
            index.add(np.ascontiguousarray(keys[head]))

    def get_index_type(self):
        return "Flat"

    def search(self, queries, groups, top_k, core):
        import faiss
        faiss.omp_set_num_threads(max(1, int(core)))
        queries = np.asarray(queries, dtype=np.float32)
        if (queries.ndim != 3 or queries.shape[0] != self.head_num * groups
                or queries.shape[-1] != self.dim or not 1 <= top_k <= self.ntotal):
            raise ValueError("Invalid grouped-query retrieval shape or top-k")
        distances, indices = [], []
        for head, query in enumerate(queries):
            distance, index = self.indices[head // groups].search(
                np.ascontiguousarray(query), int(top_k))
            distances.append(distance)
            indices.append(index)
        return np.stack(distances), np.stack(indices)
