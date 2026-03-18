import faiss
import numpy as np
from embeddings import embeddings

dimension = embeddings.shape[1]  # 384

indice = faiss.IndexHNSWFlat(dimension, 32)  # 32 = conexiones por nodo
indice.hnsw.efConstruction = 200  # precisión al construir

indice.add(embeddings)
print(f"Vectores en el índice: {indice.ntotal}")