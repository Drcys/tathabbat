"""Load and query the project's trained word-embedding model.

The model is trained once with gensim (models/train_embeddings.py) and saved
as plain numpy arrays (models/hadith_vectors.npz). Running the app only needs
numpy, so it installs on any Python version and any platform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class WordVectors:
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.words: list[str] = [str(w) for w in data["words"]]
        vectors = data["vectors"].astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.maximum(norms, 1e-9)   # unit length -> dot = cosine
        self.key_to_index = {w: i for i, w in enumerate(self.words)}

    def __contains__(self, word: str) -> bool:
        return word in self.key_to_index

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.vectors[self.key_to_index[key]]
        return self.vectors[[self.key_to_index[k] for k in key]]

    def most_similar(self, word: str, topn: int = 10) -> list[tuple[str, float]]:
        sims = self.vectors @ self[word]
        order = np.argsort(-sims)
        out = []
        for i in order:
            if self.words[i] != word:
                out.append((self.words[i], float(sims[i])))
                if len(out) == topn:
                    break
        return out

    @staticmethod
    def cosine_similarities(vector, matrix) -> np.ndarray:
        return matrix @ vector
