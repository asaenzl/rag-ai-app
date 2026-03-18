import requests
from bs4 import BeautifulSoup

PAPERS = [
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8826344/",  # sleep apnea
    # agrega más links aquí
]


def cargar_paper(url):
    # extraer el ID del link, ej: PMC8826344
    pmc_id = url.strip("/").split("/")[-1].replace("PMC", "")

    api_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_id}&rettype=xml"
    res = requests.get(api_url)
    soup = BeautifulSoup(res.text, "xml")

    # extraer solo los párrafos del artículo
    parrafos = soup.find_all("p")
    texto = " ".join([p.get_text(strip=True) for p in parrafos])

    return texto if texto else None

def load_paper():
    corpus = []
    for url in PAPERS:
        print(f"Cargando: {url}")
        texto = cargar_paper(url)
        if texto:
            # dividir en chunks de ~500 caracteres
            chunks = [texto[i:i + 500] for i in range(0, len(texto), 500)]
            for chunk in chunks:
                corpus.append({"url": url, "texto": chunk})

    print(f"Total chunks: {len(corpus)}")
    return corpus

corpus = load_paper()
print(corpus[0]["texto"])