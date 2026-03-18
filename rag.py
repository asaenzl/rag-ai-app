import os
from openai import AzureOpenAI

# ──────────────────────────────────────────────
# Cliente Azure OpenAI
# .env debe tener:
#   AZURE_OPENAI_KEY      → tu API key
# ──────────────────────────────────────────────
client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version="2025-01-01-preview"
)

MODELO = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o")

SYSTEM_PROMPT = """Eres un asistente experto que responde preguntas basándose ÚNICAMENTE 
en el contexto proporcionado. 

Reglas:
- Si la respuesta está en el contexto, respóndela de forma clara y concisa.
- Si el contexto no contiene suficiente información, dilo explícitamente: 
  "No encontré información suficiente en los documentos para responder esto."
- No inventes información que no esté en el contexto.
- Cita brevemente de qué parte del contexto proviene tu respuesta cuando sea relevante.
- Responde siempre en el mismo idioma en que se hizo la pregunta."""


def construir_prompt(pregunta: str, chunks: list[dict]) -> str:
    if not chunks:
        contexto = "No se encontraron fragmentos relevantes en los documentos."
    else:
        partes = []
        for i, chunk in enumerate(chunks, 1):
            similitud = round((1 - chunk["distancia"]) * 100, 1)
            partes.append(f"[Fragmento {i} — similitud: {similitud}%]\n{chunk['texto']}")
        contexto = "\n\n".join(partes)

    return f"""Contexto recuperado de los documentos:
---
{contexto}
---

Pregunta del usuario: {pregunta}

Responde basándote en el contexto anterior."""


def generar_respuesta(pregunta: str, chunks: list[dict]) -> dict:
    prompt = construir_prompt(pregunta, chunks)

    respuesta = client.chat.completions.create(
        model=MODELO,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
        temperature=0.3
    )

    respuesta_texto = respuesta.choices[0].message.content

    fuentes = []
    for chunk in chunks:
        fuente = chunk.get("url", chunk["texto"][:80] + "...")
        if fuente not in fuentes:
            fuentes.append(fuente)

    return {
        "respuesta": respuesta_texto,
        "fuentes": fuentes,
        "modelo": MODELO,
        "tokens_usados": respuesta.usage.total_tokens
    }
