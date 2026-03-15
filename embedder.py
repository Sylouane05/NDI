# embedder.py
#
# Rôle : transformer chaque projet du JSON en vecteur d'embedding,
#        puis les sauvegarder sur disque pour ne pas les recalculer à chaque démarrage.
#
# Pourquoi des embeddings plutôt que rapidfuzz ?
#   rapidfuzz compare des caractères. "malade" et "santé" ont 0% de ressemblance.
#   Un embedding capture le SENS : ces deux mots seront proches dans l'espace vectoriel.
#
# Fichiers produits :
#   embeddings.npy  — matrice numpy (N projets × 1536 dimensions)
#   embed_ids.json  — liste des IDs dans le même ordre que les lignes de la matrice
#
# Usage :
#   python embedder.py          → génère les fichiers
#   from embedder import search_projects  → recherche depuis app.py

import json
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
ENV_FILE        = "cle.env"
DB_FILE         = "info_projet.json"
EMBED_MODEL     = "text-embedding-3-small"   # 1536 dims, rapide, pas cher
EMBEDDINGS_FILE = "embeddings.npy"
IDS_FILE        = "embed_ids.json"
TOP_K           = 8


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def project_to_text(p: dict) -> str:
    """
    Concatène les champs d'un projet en une seule chaîne de texte.
    C'est CE texte qui sera transformé en vecteur.
    Plus tu mets d'information ici, plus la recherche sera précise.
    """
    parts = [
        p.get("nom_projet", ""),
        p.get("ville", ""),
        p.get("departement", ""),
        p.get("region", ""),
        p.get("type_projet", ""),
        p.get("etat", ""),
        p.get("description", ""),
    ]
    return " | ".join(str(x) for x in parts if x)


def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Calcule la similarité cosinus entre un vecteur requête et toutes les lignes de la matrice.

    Formule : sim(A, B) = (A · B) / (||A|| × ||B||)

    Résultat entre -1 et 1.  1 = identiques,  0 = sans rapport,  -1 = opposés.
    En pratique pour des embeddings texte on reste entre 0 et 1.

    Pourquoi cosinus et pas distance euclidienne ?
    Les embeddings OpenAI sont normalisés (||v|| ≈ 1), donc cosinus = produit scalaire.
    Mais on garde la formule complète pour être rigoureux.
    """
    # Normalise le vecteur requête
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)

    # Normalise chaque ligne de la matrice
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix_norm = matrix / norms

    # Produit scalaire → un score par projet
    return matrix_norm @ query_norm


# ─────────────────────────────────────────
# CLIENT OPENAI
# ─────────────────────────────────────────
def get_client() -> OpenAI:
    load_dotenv(ENV_FILE)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY introuvable dans cle.env")
    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """
    Envoie une liste de textes à l'API OpenAI et récupère leurs embeddings.
    L'API accepte jusqu'à 2048 textes par appel — on envoie tout d'un coup.
    Retourne une matrice numpy de shape (len(texts), 1536).
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    # response.data est une liste d'objets, chacun a un attribut .embedding (liste de floats)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


# ─────────────────────────────────────────
# INDEXATION (à lancer une fois)
# ─────────────────────────────────────────
def build_index():
    """
    Charge info_projet.json, génère un embedding par projet,
    et sauvegarde la matrice + les IDs sur disque.

    À relancer uniquement si tu modifies la base de données.
    """
    print("Chargement des projets...")
    with open(DB_FILE, "r", encoding="utf-8") as f:
        projects = json.load(f)

    texts = [project_to_text(p) for p in projects]
    ids   = [str(p.get("id", i)) for i, p in enumerate(projects)]

    print(f"{len(texts)} projets à embedder...")
    client = get_client()
    matrix = embed_texts(client, texts)

    np.save(EMBEDDINGS_FILE, matrix)
    with open(IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(f"Index sauvegardé : {matrix.shape} → {EMBEDDINGS_FILE}")
    print(f"IDs sauvegardés  : {IDS_FILE}")


# ─────────────────────────────────────────
# RECHERCHE (utilisée par app.py)
# ─────────────────────────────────────────
# Seuil de similarité cosinus pour le filtre thème.
# En dessous, le projet n'est pas considéré comme lié au thème.
# Valeur empirique : 0.30 élimine les projets sans rapport tout en restant
# souple sur les formulations ("santé" matche "soins palliatifs", "CHU"…).
SIMILARITY_THRESHOLD = 0.30

def search_projects(
    query: str,
    all_projects: list[dict],
    client: OpenAI,
    top_k: int = TOP_K,
    min_score: float = 0.0,   # 0.0 = pas de seuil (comportement par défaut pour la recherche libre)
) -> list[dict]:
    """
    Recherche vectorielle par similarité cosinus.

    Paramètres :
      query        — texte de la requête
      all_projects — liste des projets dans laquelle chercher
      client       — instance OpenAI
      top_k        — nombre max de résultats
      min_score    — seuil de similarité minimum (0.0 à 1.0).
                     Mettre SIMILARITY_THRESHOLD pour le filtre thème,
                     laisser à 0.0 pour la recherche libre (on veut toujours
                     des résultats même si rien n'est très proche).
    """
    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(IDS_FILE):
        raise FileNotFoundError(
            "Index introuvable. Lance d'abord : python embedder.py"
        )

    matrix = np.load(EMBEDDINGS_FILE)
    with open(IDS_FILE, "r", encoding="utf-8") as f:
        ids = json.load(f)

    # Restreindre la matrice aux projets de all_projects (filtre amont possible)
    allowed_ids = {str(p.get("id")) for p in all_projects}
    mask = [i for i, pid in enumerate(ids) if pid in allowed_ids]
    if not mask:
        return []
    sub_matrix = matrix[mask]
    sub_ids    = [ids[i] for i in mask]

    query_vec = embed_texts(client, [query])[0]
    scores    = cosine_similarity(query_vec, sub_matrix)

    # Appliquer le seuil — les projets sous min_score sont exclus
    sorted_indices = np.argsort(scores)[::-1]
    id_to_project  = {str(p.get("id")): p for p in all_projects}
    results = []
    for i in sorted_indices:
        if scores[i] < min_score:
            break   # tableau trié décroissant — dès qu'on passe sous le seuil, on arrête
        pid = sub_ids[i]
        if pid in id_to_project:
            results.append(id_to_project[pid])
        if len(results) >= top_k:
            break

    return results


# ─────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────
if __name__ == "__main__":
    build_index()