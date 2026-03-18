import os
import io
from sentence_transformers import SentenceTransformer

# Tamaño de cada chunk en caracteres
CHUNK_SIZE = 500

def extraer_texto(file_bytes: bytes, filename: str) -> str:
    """
    Extrae texto de un archivo según su extensión.
    Soporta: PDF, DOCX, TXT
    """
    extension = filename.lower().split(".")[-1]

    if extension == "pdf":
        return _extraer_pdf(file_bytes)
    elif extension == "docx":
        return _extraer_docx(file_bytes)
    elif extension == "txt":
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Formato no soportado: {extension}. Usa PDF, DOCX o TXT.")


def _extraer_pdf(file_bytes: bytes) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texto = ""
        for pagina in doc:
            texto += pagina.get_text()
        return texto.strip()
    except ImportError:
        raise ImportError("Instala PyMuPDF: pip install pymupdf")


def _extraer_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        texto = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return texto.strip()
    except ImportError:
        raise ImportError("Instala python-docx: pip install python-docx")


def dividir_en_chunks(texto: str, url: str = "uploaded") -> list[dict]:
    """
    Divide el texto en chunks de CHUNK_SIZE caracteres.
    Retorna lista de dicts con 'texto' y 'url'.
    """
    chunks = []
    for i in range(0, len(texto), CHUNK_SIZE):
        chunk = texto[i:i + CHUNK_SIZE].strip()
        if chunk:  # ignorar chunks vacíos
            chunks.append({"texto": chunk, "url": url})
    return chunks


def indexar_chunks(chunks: list[dict], modelo, indice, corpus: list) -> int:
    """
    Genera embeddings de los nuevos chunks y los agrega al índice FAISS.
    También los agrega al corpus en memoria.

    Retorna la cantidad de chunks indexados.
    """
    import numpy as np

    textos = [c["texto"] for c in chunks]

    nuevos_embeddings = modelo.encode(
        textos,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    indice.add(nuevos_embeddings)
    corpus.extend(chunks)

    return len(chunks)
