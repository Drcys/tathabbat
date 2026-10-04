"""Train the project's own Arabic word-embedding model (Word2Vec, skip-gram).

The model learns what words mean from the contexts they appear in across the
six hadith books and the Quran (about 2.4 million words). Words used in the
same contexts end up close together, e.g. «الفجر» ≈ «الصبح», «الفرد» ≈ «الفذ».
The hadith finder uses it to recognise a hadith quoted *by meaning* with
different words.

    pip install gensim                      (only needed for training)
    python models/train_embeddings.py      -> models/hadith_vectors.npz  (~6 MB)

The app itself only needs numpy to use the saved vectors.
"""

import json

import numpy as np
import sys
import time
from pathlib import Path

from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.hadith import get_index  # noqa: E402
from core.normalize import words  # noqa: E402

OUT = ROOT / "models" / "hadith_vectors.npz"


def corpus():
    sents = [d["norm"].split() for d in get_index().docs]
    quran = json.loads((ROOT / "data" / "quran_simple.json").read_text("utf-8"))["quran"]
    sents += [words(v["text"]) for v in quran]
    return sents


def main():
    sents = corpus()
    print(f"{len(sents)} texts, {sum(map(len, sents))} words")
    t = time.time()
    model = Word2Vec(sents, vector_size=100, window=6, min_count=3, sg=1, negative=10,
                     epochs=8, workers=1, seed=1)
    print(f"trained in {time.time() - t:.0f}s, vocabulary {len(model.wv)} words")
    np.savez_compressed(OUT, words=np.array(model.wv.index_to_key),
                        vectors=model.wv.vectors.astype(np.float16))
    for w in ["الفجر", "الطيبه", "الفذ"]:
        print(w, "≈", [x for x, _ in model.wv.most_similar(w, topn=4)])


if __name__ == "__main__":
    main()
