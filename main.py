from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
from index import indice
from embeddings import embeddings, modelo, CORPUSS
from rag import generar_respuesta
from document_processor import extraer_texto, dividir_en_chunks, indexar_chunks
import numpy as np
import json
import os
from datetime import datetime

app = Flask(__name__)

FEEDBACK_FILE = "feedback.json"


def cargar_feedback() -> list:
    if not os.path.exists(FEEDBACK_FILE):
        return []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def guardar_feedback(data: list):
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def recuperar_chunks(query: str, k: int = 3) -> list[dict]:
    query_vector = modelo.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    distancias, indices = indice.search(query_vector, k)

    chunks = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        chunk = dict(CORPUSS[int(idx)])
        chunk["distancia"] = round(float(distancias[0][i]), 4)
        chunks.append(chunk)

    return chunks


# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
@app.route("/buscar", methods=["POST"])
def buscar():
    query = request.json.get("query", "")
    if not query:
        return jsonify({"error": "Debes enviar una 'query'"}), 400
    chunks = recuperar_chunks(query)
    return jsonify(chunks)


# ──────────────────────────────────────────────
# RAG completo
# ──────────────────────────────────────────────
@app.route("/preguntar", methods=["POST"])
def preguntar():
    data = request.json or {}
    pregunta = data.get("pregunta", "").strip()

    if not pregunta:
        return jsonify({"error": "Debes enviar una 'pregunta'"}), 400

    k = data.get("k", 5)
    chunks = recuperar_chunks(pregunta, k=k)
    resultado = generar_respuesta(pregunta, chunks)
    resultado["chunks_usados"] = chunks

    return jsonify(resultado)


# ──────────────────────────────────────────────
# Upload de documentos
# ──────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    archivo = request.files["file"]
    filename = archivo.filename

    if not filename:
        return jsonify({"error": "Archivo sin nombre"}), 400

    extension = filename.lower().split(".")[-1]
    if extension not in ["pdf", "docx", "txt"]:
        return jsonify({"error": "Formato no soportado. Usa PDF, DOCX o TXT"}), 400

    try:
        file_bytes = archivo.read()
        texto = extraer_texto(file_bytes, filename)

        if not texto.strip():
            return jsonify({"error": "No se pudo extraer texto del archivo"}), 400

        chunks = dividir_en_chunks(texto, url=filename)
        total = indexar_chunks(chunks, modelo, indice, CORPUSS)

        return jsonify({
            "mensaje": "Documento indexado correctamente",
            "archivo": filename,
            "chunks_generados": total,
            "total_en_indice": indice.ntotal
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/documentos", methods=["GET"])
def documentos():
    fuentes = list(set([doc.get("url", "desconocido") for doc in CORPUSS]))
    return jsonify({
        "total_chunks": len(CORPUSS),
        "documentos": fuentes
    })


# ──────────────────────────────────────────────
# Feedback
# ──────────────────────────────────────────────
@app.route("/feedback", methods=["POST"])
def guardar_feedback_endpoint():
    data = request.json or {}

    pregunta = data.get("pregunta", "").strip()
    respuesta = data.get("respuesta", "").strip()
    voto = data.get("voto", "").strip()
    comentario = data.get("comentario", "").strip()

    if not pregunta or not respuesta:
        return jsonify({"error": "Se requieren 'pregunta' y 'respuesta'"}), 400

    if voto not in ["like", "dislike"]:
        return jsonify({"error": "'voto' debe ser 'like' o 'dislike'"}), 400

    registro = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "fecha": datetime.now().isoformat(),
        "pregunta": pregunta,
        "respuesta": respuesta,
        "voto": voto,
        "comentario": comentario
    }

    feedbacks = cargar_feedback()
    feedbacks.append(registro)
    guardar_feedback(feedbacks)

    return jsonify({
        "mensaje": "Feedback guardado correctamente",
        "id": registro["id"]
    })


@app.route("/feedback", methods=["GET"])
def ver_feedback():
    feedbacks = cargar_feedback()
    total = len(feedbacks)
    likes = sum(1 for f in feedbacks if f["voto"] == "like")
    dislikes = total - likes

    return jsonify({
        "total": total,
        "likes": likes,
        "dislikes": dislikes,
        "tasa_likes": round(likes / total * 100, 1) if total > 0 else 0,
        "registros": feedbacks
    })


# ──────────────────────────────────────────────
# Frontend
# ──────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)