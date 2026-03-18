from sentence_transformers import SentenceTransformer
#from corpus import CORPUS
import numpy as np
from papers_connection import load_paper

modelo = SentenceTransformer("all-MiniLM-L6-v2")  #convierte textos en vectores

CORPUSS = load_paper()
textos = [doc["texto"] for doc in CORPUSS]

embeddings = modelo.encode(
    textos,
    normalize_embeddings=True,  # importante para similitud coseno
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")  # FAISS requiere float32
print(embeddings.shape)


# a este punto convertimos los textos a numeros

