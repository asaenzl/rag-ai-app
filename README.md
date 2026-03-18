# VECTOR//SEARCH — Sistema RAG Avanzado

Sistema de consulta inteligente basado en **Retrieval Augmented Generation (RAG)** con interfaz web, carga dinámica de documentos y sistema de retroalimentación de usuario.

---

## Descripción

Las organizaciones manejan grandes volúmenes de documentos que son difíciles de consultar con búsquedas tradicionales. Este sistema permite hacer preguntas en lenguaje natural sobre documentos cargados, recuperando información relevante y generando respuestas contextualizadas usando un modelo de lenguaje.

---

## Arquitectura

```
┌─────────────────────────────────────────────┐
│              Frontend (HTML/JS)              │
│  RAG Chat · Vector Search · Upload · Stats  │
└────────────────────┬────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────┐
│             Backend (Flask)                  │
│  /preguntar · /buscar · /upload · /feedback  │
└──────┬─────────────┬───────────────┬────────┘
       │             │               │
┌──────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
│  Embeddings │ │  FAISS   │ │  Azure GPT  │
│ MiniLM-L6  │ │   HNSW   │ │    4o       │
└─────────────┘ └──────────┘ └─────────────┘
```

---

## Tecnologías

| Capa | Tecnología |
|------|-----------|
| Backend | Flask (Python) |
| Embeddings | SentenceTransformers — `all-MiniLM-L6-v2` |
| Índice vectorial | FAISS — IndexHNSWFlat |
| Modelo LLM | GPT-4o via Azure AI Foundry |
| Extracción PDF | PyMuPDF (fitz) |
| Extracción DOCX | python-docx |
| Frontend | HTML · CSS · Vanilla JS |

---

## Estructura del proyecto

```
├── main.py                 # Servidor Flask + endpoints
├── rag.py                  # Pipeline RAG (prompt + generación)
├── embeddings.py           # Generación de embeddings
├── index.py                # Construcción del índice FAISS
├── document_processor.py   # Extracción de texto y chunking
├── papers_connection.py    # Carga de papers desde PubMed
├── corpus.py               # Corpus base (opcional)
├── visualization.py        # Visualización 3D del espacio vectorial
├── feedback.json           # Almacenamiento de feedback (generado automáticamente)
├── .env                    # Variables de entorno (no subir a git)
├── .gitignore
└── templates/
    └── index.html          # Interfaz web
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repo>
cd <nombre-del-proyecto>
```

### 2. Crear entorno virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install flask sentence-transformers faiss-cpu numpy \
            openai python-dotenv pymupdf python-docx \
            requests beautifulsoup4 scikit-learn plotly pandas
```

### 4. Configurar variables de entorno

Crear archivo `.env` en la raíz del proyecto:

```
AZURE_OPENAI_KEY=tu-api-key-aqui
AZURE_OPENAI_ENDPOINT=https://tu-recurso.cognitiveservices.azure.com/
AZURE_DEPLOYMENT_NAME=gpt-4o
```

### 5. Ejecutar

```bash
python main.py
```

Abrir el navegador en: **http://127.0.0.1:5000**

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/preguntar` | Pipeline RAG completo — retrieval + generación |
| `POST` | `/buscar` | Solo retrieval vectorial (sin LLM) |
| `POST` | `/upload` | Subir y indexar un documento |
| `GET` | `/documentos` | Listar documentos indexados |
| `POST` | `/feedback` | Guardar like/dislike/comentario |
| `GET` | `/feedback` | Ver estadísticas de feedback |

### Ejemplo `/preguntar`

```json
POST /preguntar
{
  "pregunta": "What is sleep apnea?",
  "k": 5
}
```

```json
{
  "respuesta": "Sleep apnea is a disorder characterized by...",
  "fuentes": ["https://pmc.ncbi.nlm.nih.gov/articles/PMC8826344/"],
  "modelo": "gpt-4o",
  "tokens_usados": 523,
  "chunks_usados": [...]
}
```

### Ejemplo `/feedback`

```json
POST /feedback
{
  "pregunta": "What is sleep apnea?",
  "respuesta": "Sleep apnea is...",
  "voto": "like",
  "comentario": "Muy buena respuesta"
}
```

---

## Pipeline RAG

### Flujo de carga
```
Documento → Extracción de texto → Chunking (500 chars) → Embeddings → FAISS
```

### Flujo de consulta
```
Pregunta → Embedding → FAISS Search (top-k) → Prompt con contexto → GPT-4o → Respuesta
```

---

## Interfaz

La interfaz tiene 4 secciones:

- **RAG CHAT** — Chat con el sistema RAG y botones de feedback por respuesta
- **VECTOR SEARCH** — Búsqueda semántica pura (sin generación)
- **UPLOAD DOCS** — Subir archivos PDF, DOCX o TXT
- **FEEDBACK STATS** — Métricas de likes/dislikes en tiempo real

---

## Formatos soportados

- `.pdf` — extracción con PyMuPDF
- `.docx` — extracción con python-docx
- `.txt` — lectura directa UTF-8

---

## Notas

- El archivo `feedback.json` se genera automáticamente al recibir el primer feedback
- El índice FAISS es en memoria — se reconstruye al reiniciar el servidor
- El modelo de embeddings `all-MiniLM-L6-v2` está optimizado para inglés; para consultas en español usar `paraphrase-multilingual-MiniLM-L12-v2`
- No subir `.env` al repositorio — contiene credenciales

---

## .gitignore recomendado

```
.env
.venv/
__pycache__/
*.pyc
feedback.json
```
