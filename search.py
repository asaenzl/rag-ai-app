import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from corpus import CORPUS
from embeddings import embeddings, modelo
from index import indice

query = "¿Cómo funcionan las redes neuronales?"

query_vector = modelo.encode(
    [query],
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

k = 3  # cuántos resultados quieres

distancias, indices = indice.search(query_vector, k)

print(f"\nResultados para: '{query}'\n")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. [{distancias[0][i]:.4f}] {CORPUS[idx]}")