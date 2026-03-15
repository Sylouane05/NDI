# app.py — Architecture agent multi-outils
#
# Prérequis :
#   pip install streamlit openai python-dotenv numpy pandas rapidfuzz
#   python embedder.py   (une fois, ou après modif du JSON)
#   streamlit run app.py

import json
import os
import re
import unicodedata
from typing import Any

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz, process as fuzz_process

from embedder import embed_texts, cosine_similarity, EMBEDDINGS_FILE, IDS_FILE, SIMILARITY_THRESHOLD

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
ENV_FILE   = "cle.env"
DB_FILE    = "info_projet.json"
CHAT_MODEL = "gpt-4o"

# ─────────────────────────────────────────
# UTILS TEXTE  (en premier — utilisés par load_projects)
# ─────────────────────────────────────────
def _norm(s: Any) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).lower().strip()
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s)

def _parse_date(d: str) -> str:
    d = (d or "").strip()
    for pat, fmt in [
        (r"^(\d{2})/(\d{2})/(\d{4})$", lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}"),
        (r"^(\d{1,2})/(\d{4})$",        lambda m: f"{m.group(2)}-{m.group(1).zfill(2)}-01"),
        (r"^(\d{4})$",                   lambda m: f"{m.group(1)}-01-01"),
    ]:
        m = re.match(pat, d)
        if m:
            return fmt(m)
    return "0000-00-00"

def _compact(p: dict) -> dict:
    return {k: p.get(k) for k in
            ["id", "nom_projet", "ville", "departement",
             "region", "date", "etat", "type_projet", "description"]}

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.set_page_config(page_title="Code ton territoire", page_icon="💬", layout="centered")
st.markdown("""
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; }
.header {
  padding: 14px 16px;
  border: 1px solid rgba(20,20,20,0.10);
  border-radius: 18px;
  background: rgba(255,255,255,0.88);
}
.small { color: rgba(0,0,0,0.62); font-size: 0.92rem; margin-top: 4px; }
.pill {
  display:inline-block; padding: 2px 10px; border-radius: 999px;
  border: 1px solid rgba(20,20,20,0.10);
  background: rgba(255,255,255,0.75); font-size: .82rem;
  margin-right: 6px; margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
  <div style="font-size: 1.35rem; font-weight: 780;">💬 Code ton territoire</div>
  <div class="small">Projets près de chez toi · Documents · Soumettre un projet ✨</div>
  <div style="margin-top: 10px;">
    <span class="pill">"Projets à Toulon"</span>
    <span class="pill">"Thème santé"</span>
    <span class="pill">"Combien en cours ?"</span>
    <span class="pill">"Je veux soumettre un projet"</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.title("Conversation")

# ─────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────
@st.cache_resource
def load_client() -> OpenAI:
    load_dotenv(ENV_FILE)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY introuvable dans cle.env")
    return OpenAI(api_key=key)

@st.cache_resource
def load_projects() -> list[dict]:
    with open(DB_FILE, encoding="utf-8") as f:
        data = json.load(f)
    for p in data:
        p["_date_sort"] = _parse_date(p.get("date", ""))
    return data

@st.cache_resource
def load_embedding_index() -> tuple[np.ndarray, list[str]]:
    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(IDS_FILE):
        raise FileNotFoundError("Lance d'abord : python embedder.py")
    matrix = np.load(EMBEDDINGS_FILE)
    with open(IDS_FILE, encoding="utf-8") as f:
        ids = json.load(f)
    return matrix, ids

client              = load_client()
PROJECTS            = load_projects()
EMB_MATRIX, EMB_IDS = load_embedding_index()

# ─────────────────────────────────────────
# OPÉRATIONS PYTHON PURES
# ─────────────────────────────────────────
def _op_semantic(query: str, pool: list[dict] | None = None) -> list[dict]:
    pool    = pool or PROJECTS
    allowed = {str(p.get("id")) for p in pool}
    mask        = [i for i, pid in enumerate(EMB_IDS) if pid in allowed]
    sub_matrix  = EMB_MATRIX[mask]
    sub_ids     = [EMB_IDS[i] for i in mask]
    query_vec   = embed_texts(client, [query])[0]
    scores      = cosine_similarity(query_vec, sub_matrix)
    id_to_p     = {str(p.get("id")): p for p in pool}
    results     = []
    for i in np.argsort(scores)[::-1]:
        if scores[i] < SIMILARITY_THRESHOLD:
            break
        pid = sub_ids[i]
        if pid in id_to_p:
            results.append(id_to_p[pid])
    return results


def _op_filter(
    pool: list[dict],
    ville: str | None       = None,
    departement: str | None = None,
    etat: str | None        = None,
    annee: str | None       = None,
) -> list[dict] | None:
    results = pool[:]
    if annee:
        results = [p for p in results if annee in str(p.get("date", ""))]
    if etat:
        results = [p for p in results if _norm(p.get("etat", "")) == _norm(etat)]
    if departement:
        results = [p for p in results if _norm(str(p.get("departement", ""))) == _norm(departement)]
    if ville:
        target = _norm(ville)
        exact  = [p for p in results if _norm(p.get("ville", "")) == target]
        if exact:
            results = exact
        else:
            choices = list({_norm(p.get("ville", "")) for p in results})
            match   = fuzz_process.extractOne(target, choices, scorer=fuzz.ratio)
            if match and match[1] >= 88:
                results = [p for p in results if _norm(p.get("ville", "")) == match[0]]
            else:
                return None
    return sorted(results, key=lambda p: p.get("_date_sort", "0000-00-00"))


# ─────────────────────────────────────────
# OUTILS GPT-4o
# ─────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_semantic",
            "description": (
                "Recherche des projets par sens/thème via embeddings vectoriels. "
                "Utiliser pour : thèmes libres (sante, jeunesse, environnement...), "
                "mots-cles semantiques. "
                "Ne pas utiliser pour des filtres exacts (ville, etat, annee)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Requete semantique libre."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_structured",
            "description": (
                "Filtre les projets sur des champs exacts : ville, departement, etat, annee. "
                "Pour combiner avec un theme : appelle filter_structured d'abord, "
                "puis search_semantic sur le resultat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ville":       {"type": "string"},
                    "departement": {"type": "string", "description": "Numero ex: 83, 06, 04"},
                    "etat":        {"type": "string", "enum": ["finit", "en cours", "pas commencé"]},
                    "annee":       {"type": "string", "description": "Annee 4 chiffres ex: 2025"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_projects",
            "description": "Compte les projets du pool courant. Appeler apres filter_structured ou search_semantic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"}
                },
                "required": ["label"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_projects",
            "description": "Retourne tous les projets sans filtre.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
]

SYSTEM_AGENT = """Tu es un assistant chaleureux qui aide a explorer des projets locaux en PACA.

Pour repondre aux questions sur les projets, utilise TOUJOURS les outils disponibles.
Regles :
- Ville / departement / etat / annee  ->  filter_structured
- Theme / sens / description libre    ->  search_semantic
- Combinaison (ex: sante a Nice)      ->  filter_structured PUIS search_semantic
- Comptage                            ->  count_projects (apres filtrage)
- Lister tout                         ->  get_all_projects

Apres les outils, formule une reponse naturelle et concise en francais.
Si une ville n'est pas dans la base, dis-le clairement.
Si les resultats sont vides, dis-le sans inventer.
Ne mentionne jamais les outils, embeddings ou base de donnees."""


# ─────────────────────────────────────────
# BOUCLE AGENT
# ─────────────────────────────────────────
def _execute_tool(name: str, args: dict, pool: list[dict] | None) -> Any:
    if name == "get_all_projects":
        return PROJECTS[:]
    if name == "filter_structured":
        return _op_filter(
            pool=pool or PROJECTS,
            ville=args.get("ville"),
            departement=args.get("departement"),
            etat=args.get("etat"),
            annee=args.get("annee"),
        )
    if name == "search_semantic":
        return _op_semantic(args["query"], pool=pool or PROJECTS)
    if name == "count_projects":
        n = len(pool) if pool is not None else len(PROJECTS)
        return {"count": n, "label": args.get("label", "")}
    return {"error": f"Outil inconnu : {name}"}


def _serialize(name: str, result: Any, args: dict) -> str:
    if result is None:
        ville = args.get("ville", "cette ville")
        return f"Aucun resultat pour '{ville}'. Ville absente de la base (couverture PACA uniquement)."
    if isinstance(result, list):
        if not result:
            return "Aucun projet ne correspond."
        return json.dumps(
            {"count": len(result), "projets": [_compact(p) for p in result]},
            ensure_ascii=False
        )
    if isinstance(result, dict):
        return result.get("answer") or result.get("error") or json.dumps(result, ensure_ascii=False)
    return str(result)


def run_agent(user_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_AGENT},
        {"role": "user",   "content": user_text},
    ]
    current_pool: list[dict] | None = None

    for _ in range(6):
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.1,
        )
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content or "Je n'ai pas pu repondre."

        for tc in msg.tool_calls:
            name   = tc.function.name
            args   = json.loads(tc.function.arguments)
            result = _execute_tool(name, args, current_pool)
            if isinstance(result, list):
                current_pool = result
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      _serialize(name, result, args),
            })

    return "Je n'ai pas pu traiter cette demande."


# ─────────────────────────────────────────
# SOUMISSION GUIDEE
# ─────────────────────────────────────────
SUBMIT_STEPS = [
    ("titre",          "Super ✨ Quel est le **titre** de ton projet (5-8 mots) ?"),
    ("ville",          "Dans quelle **ville** se deroule le projet ?"),
    ("description",    "Decris-le en **2-4 phrases** (quoi, pourquoi, comment)."),
    ("objectif",       "Quel est l'**objectif** principal ?"),
    ("beneficiaires",  "Qui sont les **beneficiaires** ?"),
    ("etat_avancement","Il en est ou ? (idee / en cours / deja lance / termine)"),
    ("contact",        "Un **contact** a ajouter ? (sinon : non)"),
]

def _submit_next(s: dict) -> str:
    step = int(s.get("_step", 0))
    return SUBMIT_STEPS[step][1] if step < len(SUBMIT_STEPS) else "Merci !"

def _submit_apply(s: dict, text: str) -> dict:
    out  = dict(s)
    step = int(out.get("_step", 0))
    if step >= len(SUBMIT_STEPS):
        return out
    key, _ = SUBMIT_STEPS[step]
    out[key] = None if (key == "contact" and _norm(text) in ("non", "non merci", "no", "nop")) else text.strip()
    out["_step"] = step + 1
    return out

def _submit_complete(s: dict) -> bool:
    return all(k in s for k, _ in SUBMIT_STEPS)

def _submit_summary(s: dict) -> str:
    return "\n".join([
        "Parfait 🙌 Voila ton projet resume :",
        f"- **{s.get('titre','?')}** ({s.get('ville','?')})",
        f"- Objectif : {s.get('objectif','?')}",
        f"- Pour : {s.get('beneficiaires','?')}",
        f"- Avancement : {s.get('etat_avancement','?')}",
        f"- Details : {s.get('description','?')}",
        f"- Contact : {s.get('contact') or 'non renseigne'}",
        "\nJe peux aussi t'aider a le reformuler en version affiche ou reseaux sociaux 😊",
    ])

def _wants_submit(text: str) -> bool:
    return any(k in _norm(text) for k in
               ["soumettre", "proposer", "ajouter", "deposer", "soumission"])


# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "chat" not in st.session_state:
    st.session_state["chat"] = [
        {"role": "assistant", "content":
         "Salut 😊 Dis-moi ta ville, un theme, ou uploade un document pour que je puisse t'aider !"}
    ]
for k, v in [("mode", "qa"), ("submit", {"_step": 0})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────
# RENDU CHAT
# ─────────────────────────────────────────
for m in st.session_state["chat"]:
    with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
        st.markdown(m["content"])

user_input = st.chat_input("Ecris ton message...")
if user_input:
    st.session_state["chat"].append({"role": "user", "content": user_input})
    q = _norm(user_input)

    if st.session_state["mode"] == "submit" and q in ("annuler", "stop", "retour", "quitter"):
        st.session_state.update(mode="qa", submit={"_step": 0})
        st.session_state["chat"].append(
            {"role": "assistant", "content": "Pas de souci 😊 On revient aux questions."})
        st.rerun()

    if st.session_state["mode"] != "submit" and _wants_submit(user_input):
        st.session_state.update(mode="submit", submit={"_step": 0})
        st.session_state["chat"].append({"role": "assistant", "content":
            "Avec plaisir ✨ (Tu peux dire annuler a tout moment.)\n\n" +
            _submit_next(st.session_state["submit"])})
        st.rerun()

    if st.session_state["mode"] == "submit":
        s   = _submit_apply(st.session_state["submit"], user_input)
        st.session_state["submit"] = s
        msg = _submit_summary(s) if _submit_complete(s) else _submit_next(s)
        if _submit_complete(s):
            st.session_state.update(mode="qa", submit={"_step": 0})
        st.session_state["chat"].append({"role": "assistant", "content": msg})
        st.rerun()

    else:
        with st.spinner("Reflexion en cours..."):
            ans = run_agent(user_input)
        st.session_state["chat"].append({"role": "assistant", "content": ans})
        st.rerun()