# 💬 Code ton territoire

> Chatbot conversationnel pour explorer les projets locaux soutenus par le Crédit Agricole en région PACA.

Développé lors d'une nuit du code, puis refactorisé pour intégrer un pipeline RAG complet et une architecture agent GPT-4o.

---

## Le projet

Le Crédit Agricole soutient des dizaines de projets locaux chaque année en PACA — associations, initiatives culturelles, sportives, environnementales, solidaires. Ces projets sont dispersés sur tout le territoire et difficiles à explorer.

**Code ton territoire** permet à n'importe qui de poser une question en langage naturel et d'obtenir une réponse précise sur ces projets :

```
"Quels projets soutient le Crédit Agricole à Toulon ?"
"Y a-t-il des initiatives pour le handicap dans le Var ?"
"Combien de projets sont encore en cours en 2025 ?"
"Je veux soumettre un projet à soutenir"
```

---

## Fonctionnalités

- **Recherche par lieu** — ville ou département, avec tolérance aux fautes de frappe
- **Recherche par thème** — santé, jeunesse, environnement, culture, solidarité... en langage libre
- **Filtres combinés** — "projets de santé en cours dans le 06"
- **Comptages exacts** — "combien de projets sont terminés en 2025 ?"
- **Soumission guidée** — formulaire conversationnel pour proposer un nouveau projet

---

## Stack technique

| Brique | Technologie |
|--------|-------------|
| Interface | Streamlit |
| LLM & routing | GPT-4o (function calling) |
| Embeddings | text-embedding-3-small (OpenAI) |
| Recherche vectorielle | NumPy — similarité cosinus |
| Recherche floue | rapidfuzz |
| Données | JSON |

---

## Installation

```bash
pip install streamlit openai python-dotenv numpy pandas rapidfuzz
```

Créer `cle.env` à la racine :

```
OPENAI_API_KEY=sk-...
```

Générer l'index vectoriel :

```bash
python embedder.py
```

Lancer :

```bash
streamlit run app.py
```

---

## Structure

```
├── app.py              # Application principale — boucle agent Streamlit
├── embedder.py         # Génération et recherche dans l'index d'embeddings
├── info_projet.json    # Base de données des projets soutenus
├── cle.env             # Clé API OpenAI  ← ne pas versionner
├── embeddings.npy      # Généré par embedder.py  ← ne pas versionner
└── embed_ids.json      # Généré par embedder.py  ← ne pas versionner
```

---

## Comment ça marche

Quand l'utilisateur pose une question, GPT-4o lit la question et choisit parmi 4 outils :

- **`filter_structured`** pour les filtres exacts (ville, département, état, année)
- **`search_semantic`** pour les recherches thématiques via similarité cosinus
- **`count_projects`** pour les questions de comptage
- **`get_all_projects`** pour tout lister

Python exécute l'outil et retourne le résultat. GPT formule ensuite la réponse en langage naturel.

Pour une question combinée — "projets de solidarité en cours à Nice" — GPT appelle `filter_structured` en premier pour restreindre le périmètre géographique, puis `search_semantic` sur ce sous-ensemble pour affiner par thème. Les deux outils partagent le même pool de résultats dans le même tour de conversation.

La recherche sémantique repose sur des embeddings vectoriels (1536 dimensions). Chaque projet est transformé en vecteur lors de l'indexation. À la requête, la question est également embeddée et les projets les plus proches sont sélectionnés par similarité cosinus, avec un seuil à 0.30 pour filtrer le bruit.

---

## Données

La base contient 38 projets soutenus par le Crédit Agricole PACA, couvrant les départements 04, 06 et 83. Chaque projet inclut le nom, la ville, le département, la date, l'état d'avancement, le type et une description.

Pour mettre à jour la base, modifier `info_projet.json` puis relancer `python embedder.py` pour régénérer l'index.

## Demo 

https://github.com/user-attachments/assets/a57c196e-34b0-45ae-98ed-c18e13799c52

