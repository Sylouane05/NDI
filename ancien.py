# app.py — Version OpenAI API
#
# Prérequis :
#   pip install streamlit openai python-dotenv rapidfuzz pandas
#
# Fichiers :
#   - cle.env  (OPENAI_API_KEY=...)
#   - info_projet.json
#
# Lancer :
#   streamlit run app.py

import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz, process

# =========================
# CONFIG
# =========================
ENV_FILE = "cle.env"
DB_FILE = "info_projet.json"
MODEL = "gpt-5.4"

TOP_K = 10
FUZZY_THRESHOLD_PROJECT = 50
FUZZY_THRESHOLD_CITY = 85

# =========================
# UI STYLE
# =========================
st.set_page_config(page_title="Code ton territoire", page_icon="💬", layout="centered")

st.markdown(
    """
<style>
.block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; }
body {
  background: radial-gradient(1200px 700px at 15% 0%, rgba(0, 180, 255, 0.12), transparent 55%),
              radial-gradient(1000px 700px at 92% 10%, rgba(120, 255, 180, 0.14), transparent 55%),
              #fbfbfd;
}
.header {
  padding: 14px 16px;
  border: 1px solid rgba(20,20,20,0.10);
  border-radius: 18px;
  background: rgba(255,255,255,0.88);
  backdrop-filter: blur(8px);
}
.small { color: rgba(0,0,0,0.62); font-size: 0.92rem; margin-top: 4px; }
.pill { display:inline-block; padding: 2px 10px; border-radius: 999px; border: 1px solid rgba(20,20,20,0.10);
        background: rgba(255,255,255,0.75); font-size: .82rem; margin-right: 6px; margin-top: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="header">
  <div style="font-size: 1.35rem; font-weight: 780;">💬 Code ton territoire</div>
  <div class="small">Je peux te montrer les projets près de chez toi, ou t’aider à proposer le tien ✨</div>
  <div style="margin-top: 10px;">
    <span class="pill">“Projets à Toulon”</span>
    <span class="pill">“Et dans le Var ?”</span>
    <span class="pill">“Thème santé / jeunesse”</span>
    <span class="pill">“Lesquels sont terminés ?”</span>
    <span class="pill">“Je veux soumettre un projet”</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.title("Conversation")

# =========================
# TEXT UTILS
# =========================
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))

def norm(s: Any) -> str:
    s = str(s or "").strip().lower()
    s = strip_accents(s)
    s = re.sub(r"\s+", " ", s)
    return s

def parse_date_to_sortable(date_str: str) -> str:
    d = (date_str or "").strip()
    if not d:
        return "0000-00-00"
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", d)
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}"
    m = re.match(r"^(\d{1,2})/(\d{4})$", d)
    if m:
        mm, yyyy = m.group(1).zfill(2), m.group(2)
        return f"{yyyy}-{mm}-01"
    m = re.match(r"^(\d{4})$", d)
    if m:
        yyyy = m.group(1)
        return f"{yyyy}-01-01"
    return "0000-00-00"

def compact_project(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": p.get("id"),
        "nom_projet": p.get("nom_projet"),
        "ville": p.get("ville"),
        "departement": p.get("departement"),
        "region": p.get("region"),
        "date": p.get("date"),
        "etat": p.get("etat"),
        "type_projet": p.get("type_projet"),
        "description": p.get("description"),
    }

def pretty_list(items: List[Dict[str, Any]], title: str, limit: int = 20) -> str:
    items = items[:limit]
    if not items:
        return "Je n’ai pas cette information."
    lines = [f"😊 {title}\n"]
    for p in items:
        date = p.get("date") or "date à préciser"
        etat = p.get("etat") or "statut à préciser"
        lines.append(
            f"- **{p.get('nom_projet')}** — {p.get('ville')} — {date} — *{etat}*\n"
            f"  {p.get('description')}"
        )
    return "\n".join(lines)

# =========================
# LOAD OPENAI + DATA
# =========================
@st.cache_resource
def load_client() -> OpenAI:
    load_dotenv(ENV_FILE)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY introuvable dans cle.env")
    return OpenAI(api_key=api_key)

@st.cache_resource
def load_projects() -> List[Dict[str, Any]]:
    with open(DB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    for p in data:
        p["_date_sort"] = parse_date_to_sortable(p.get("date", ""))
    return data

client = load_client()
PROJECTS = load_projects()

DF = pd.DataFrame([{**compact_project(p), "_date_sort": p["_date_sort"]} for p in PROJECTS])
VILLES = sorted([v for v in DF["ville"].dropna().unique().tolist()])
DEPS = sorted([d for d in DF["departement"].dropna().unique().tolist()])

# =========================
# PROJECT SEARCH (fuzzy)
# =========================
def score_project(query: str, p: Dict[str, Any]) -> int:
    hay = " ".join([
        str(p.get("nom_projet", "")),
        str(p.get("ville", "")),
        str(p.get("departement", "")),
        str(p.get("region", "")),
        str(p.get("date", "")),
        str(p.get("etat", "")),
        str(p.get("type_projet", "")),
        str(p.get("description", "")),
    ])
    return int(fuzz.partial_ratio(norm(query), norm(hay)))

def search_projects(query: str, limit: int = TOP_K) -> List[Dict[str, Any]]:
    scored = []
    for p in PROJECTS:
        s = score_project(query, p)
        if s >= FUZZY_THRESHOLD_PROJECT:
            scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:limit]]

# =========================
# THEME SEARCH
# =========================
THEME_KEYWORDS = {
    "jeunesse": ["jeunesse", "jeune", "enfant", "enfants", "ado", "scolaire", "école", "ecole", "collège", "college", "lycée", "lycee"],
    "santé": ["santé", "sante", "hôpital", "hopital", "chu", "soins", "médical", "medical", "patients", "palliatif", "palliatifs"],
    "handicap": ["handicap", "inclusion", "accessibilité", "accessibilite", "fauteuil", "autisme", "handfauteuil"],
    "environnement": ["environnement", "écologie", "ecologie", "biodiversité", "biodiversite", "nature", "recycl", "déchet", "dechet", "corail"],
    "sport": ["sport", "football", "rugby", "course", "trail", "athlétisme", "athletisme", "pétanque", "petanque"],
    "culture": ["culture", "musée", "musee", "patrimoine", "exposition", "artist", "observatoire", "planétarium", "planetarium"],
    "solidarité": ["solidarité", "solidarite", "précarité", "precarite", "caritatif", "restos du coeur", "restos"],
}

def detect_theme(text: str) -> Optional[str]:
    t = norm(text)
    for theme, keywords in THEME_KEYWORDS.items():
        if any(norm(kw) in t for kw in keywords):
            return theme
    return None

def projects_by_theme(theme: str) -> List[Dict[str, Any]]:
    keywords = [norm(k) for k in THEME_KEYWORDS.get(theme, [])]
    results = []
    for p in PROJECTS:
        hay = " ".join([
            str(p.get("nom_projet", "")),
            str(p.get("type_projet", "")),
            str(p.get("description", "")),
            str(p.get("ville", "")),
            str(p.get("departement", "")),
        ])
        h = norm(hay)
        if any(kw in h for kw in keywords):
            results.append(p)
    return sorted(results, key=lambda p: p.get("_date_sort", "0000-00-00"))

# =========================
# PLACE DETECTION
# =========================
def best_city_match(text: str) -> Optional[str]:
    t = norm(text)
    for city in VILLES:
        if norm(city) and norm(city) in t:
            return city

    match = process.extractOne(
        query=t,
        choices=[norm(c) for c in VILLES],
        scorer=fuzz.partial_ratio
    )
    if match and match[1] >= FUZZY_THRESHOLD_CITY:
        idx = [norm(c) for c in VILLES].index(match[0])
        return VILLES[idx]

    m = re.search(r"\b(a|à)\s+([a-zA-ZÀ-ÿ' -]+)", text.strip(), flags=re.IGNORECASE)
    if m:
        guess = m.group(2).strip()
        match2 = process.extractOne(
            query=norm(guess),
            choices=[norm(c) for c in VILLES],
            scorer=fuzz.ratio
        )
        if match2 and match2[1] >= FUZZY_THRESHOLD_CITY:
            idx = [norm(c) for c in VILLES].index(match2[0])
            return VILLES[idx]
    return None

def best_dep_match(text: str) -> Optional[str]:
    t = norm(text)
    for dep in DEPS:
        if norm(dep) and norm(dep) in t:
            return dep
    if "var" in t and any(norm(d) == "var" for d in DEPS):
        return "Var"
    return None

def projects_in_city(city: str) -> List[Dict[str, Any]]:
    target = norm(city)
    items = [p for p in PROJECTS if target and target == norm(p.get("ville"))]
    return sorted(items, key=lambda p: p.get("_date_sort", "0000-00-00"))

def projects_in_dep(dep: str) -> List[Dict[str, Any]]:
    target = norm(dep)
    items = [p for p in PROJECTS if target and target == norm(p.get("departement"))]
    return sorted(items, key=lambda p: p.get("_date_sort", "0000-00-00"))

def infer_dep_from_city(city: str) -> Optional[str]:
    items = [p for p in PROJECTS if norm(p.get("ville")) == norm(city)]
    deps = [p.get("departement") for p in items if p.get("departement")]
    if not deps:
        return None
    return pd.Series(deps).mode().iloc[0]

# =========================
# OPENAI RESPONSES API
# =========================
SYSTEM_QA = (
    "Tu es un assistant chaleureux.\n"
    "Tu réponds uniquement avec les informations du CONTEXTE.\n"
    "Ne mentionne jamais de base de données, JSON, extraits, contexte.\n"
    "Si tu ne peux pas répondre, dis: \"Je n’ai pas cette information.\".\n"
    "Sois clair, concis, naturel.\n"
)

def llm_answer(question: str, context_obj: Dict[str, Any]) -> str:
    response = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_QA,
        input=f"QUESTION:\n{question}\n\nCONTEXTE:\n{json.dumps(context_obj, ensure_ascii=False, indent=2)}",
    )
    text = response.output_text.strip()
    return text or "Je n’ai pas cette information."

# =========================
# SCOPE POUR QUESTIONS DE SUIVI
# =========================
def set_scope(items: List[Dict[str, Any]]):
    st.session_state["scope_ids"] = [p.get("id") for p in items if p.get("id") is not None]

def get_scope_items() -> List[Dict[str, Any]]:
    ids = st.session_state.get("scope_ids") or []
    if not ids:
        return []
    idset = set(str(x) for x in ids)
    return [p for p in PROJECTS if str(p.get("id")) in idset]

def filter_by_etat(items: List[Dict[str, Any]], etat: str) -> List[Dict[str, Any]]:
    w = norm(etat)
    return [p for p in items if w == norm(p.get("etat"))]

def scope_villes(items: List[Dict[str, Any]]) -> List[str]:
    return sorted({p.get("ville") for p in items if p.get("ville")})

# =========================
# QA ROUTER
# =========================
def answer_qa(user_text: str) -> str:
    q = norm(user_text)

    theme = detect_theme(user_text)
    if theme:
        items = projects_by_theme(theme)
        if items:
            set_scope(items)
            return pretty_list(items, f"Voici des projets liés au thème **{theme}** :", limit=25)
        return "Je n’ai pas cette information."

    scope = get_scope_items()
    if scope:
        if any(k in q for k in ["termin", "fini", "finis", "termine", "terminé"]):
            items = filter_by_etat(scope, "finit")
            set_scope(items)
            return pretty_list(items, "Voici ceux qui sont terminés :", limit=25)

        if "en cours" in q:
            items = filter_by_etat(scope, "en cours")
            set_scope(items)
            return pretty_list(items, "Voici ceux qui sont en cours :", limit=25)

        if any(k in q for k in ["pas commenc", "non commenc"]):
            items = filter_by_etat(scope, "pas commencé")
            set_scope(items)
            return pretty_list(items, "Voici ceux qui n’ont pas encore commencé :", limit=25)

        if any(k in q for k in ["quelles villes", "villes concerne", "villes concern", "quels lieux", "lieux"]):
            villes = scope_villes(scope)
            if not villes:
                return "Je n’ai pas cette information."
            return "📍 Les villes concernées sont :\n- " + "\n- ".join(villes)

    dep = best_dep_match(user_text)
    if dep:
        items = projects_in_dep(dep)
        set_scope(items)
        if items:
            return pretty_list(items, f"Voici les projets dans le {dep} :", limit=25)
        return "Je n’ai pas cette information."

    city = best_city_match(user_text)
    if city:
        if any(k in q for k in ["ma region", "ma région", "autour", "près", "pres", "dans le departement", "département"]):
            inferred = infer_dep_from_city(city)
            if inferred:
                items = projects_in_dep(inferred)
                set_scope(items)
                if items:
                    return pretty_list(items, f"Autour de {city} :", limit=25)

        items = projects_in_city(city)
        set_scope(items)
        if items:
            return pretty_list(items, f"Voici les projets à {city} :", limit=25)
        return f"Je n’ai pas trouvé de projet pour {city}. Tu veux regarder le département ? 😊"

    hits = search_projects(user_text, limit=TOP_K)
    if not hits:
        return "Je n’ai pas cette information. Tu peux me dire une **ville**, un **département**, ou un **thème** 😊"

    set_scope(hits)
    ctx = {"projets": [compact_project(p) for p in hits]}
    return llm_answer(user_text, ctx)

# =========================
# SOUMISSION GUIDÉE
# =========================
SUBMIT_STEPS = [
    ("titre", "Super ✨ Quel est le **titre** de ton projet (5–8 mots) ?"),
    ("ville", "Dans quelle **ville** se déroule le projet ?"),
    ("description", "Décris-le en **2–4 phrases** (quoi, pourquoi, comment)."),
    ("objectif", "Quel est l’**objectif** principal ? (ex: aider, sensibiliser, améliorer…)"),
    ("beneficiaires", "Qui sont les **bénéficiaires** (public visé) ?"),
    ("etat_avancement", "Il en est où aujourd’hui ? (idée / en cours / déjà lancé / terminé)"),
    ("contact", "Tu veux ajouter un **contact** (email/téléphone) ? (sinon réponds: non)"),
]

def submit_next_question(state: Dict[str, Any]) -> str:
    step = int(state.get("_step", 0))
    if step >= len(SUBMIT_STEPS):
        return "Merci !"
    return SUBMIT_STEPS[step][1]

def submit_apply_answer(state: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    out = dict(state)
    step = int(out.get("_step", 0))
    if step >= len(SUBMIT_STEPS):
        return out

    key, _q = SUBMIT_STEPS[step]
    ans = user_text.strip()

    if key == "contact" and norm(ans) in ("non", "non merci", "no", "nop"):
        out[key] = None
    else:
        out[key] = ans

    out["_step"] = step + 1
    return out

def submit_complete(state: Dict[str, Any]) -> bool:
    required = ["titre", "ville", "description", "objectif", "beneficiaires", "etat_avancement", "contact"]
    return all(k in state for k in required)

def submit_text_summary(state: Dict[str, Any]) -> str:
    titre = state.get("titre", "à préciser")
    ville = state.get("ville", "à préciser")
    objectif = state.get("objectif", "à préciser")
    beneficiaires = state.get("beneficiaires", "à préciser")
    avancement = state.get("etat_avancement", "à préciser")
    description = state.get("description", "à préciser")
    contact = state.get("contact", None)

    lines = [
        "Parfait 🙌 Voilà ton projet résumé :",
        f"- **{titre}** ({ville})",
        f"- Objectif : {objectif}",
        f"- Pour : {beneficiaires}",
        f"- Avancement : {avancement}",
        f"- Détails : {description}",
    ]
    if contact:
        lines.append(f"- Contact : {contact}")
    else:
        lines.append("- Contact : non renseigné")

    lines.append("\nSi tu veux, je peux aussi t’aider à le reformuler en version “affiche” ou “réseaux sociaux” 😊")
    return "\n".join(lines)

def wants_submit(text: str) -> bool:
    t = norm(text)
    triggers = [
        "soumettre", "proposer", "ajouter", "deposer", "déposer", "soumission",
        "mettre un projet", "soumettre un projet", "proposer un projet"
    ]
    return any(k in t for k in triggers)

# =========================
# SESSION STATE
# =========================
if "chat" not in st.session_state:
    st.session_state["chat"] = [
        {"role": "assistant", "content": "Salut 😊 Dis-moi ta ville, un département, ou un thème (santé, jeunesse…), et je t’aide avec plaisir !"}
    ]
if "mode" not in st.session_state:
    st.session_state["mode"] = "qa"
if "submit" not in st.session_state:
    st.session_state["submit"] = {"_step": 0}
if "scope_ids" not in st.session_state:
    st.session_state["scope_ids"] = []

# =========================
# RENDER CHAT
# =========================
for m in st.session_state["chat"]:
    with st.chat_message("assistant" if m["role"] == "assistant" else "user"):
        st.markdown(m["content"])

user_input = st.chat_input("Écris ton message…")
if user_input:
    st.session_state["chat"].append({"role": "user", "content": user_input})
    q = norm(user_input)

    if st.session_state["mode"] == "submit" and q in ("annuler", "stop", "retour", "quitter"):
        st.session_state["mode"] = "qa"
        st.session_state["submit"] = {"_step": 0}
        st.session_state["chat"].append({"role": "assistant", "content": "Pas de souci 😊 On revient aux questions."})
        st.rerun()

    if st.session_state["mode"] != "submit" and wants_submit(user_input):
        st.session_state["mode"] = "submit"
        st.session_state["submit"] = {"_step": 0}
        st.session_state["chat"].append(
            {"role": "assistant", "content": "Avec plaisir ✨ Je te guide étape par étape. (Tu peux dire “annuler” à tout moment.)\n\n" + submit_next_question(st.session_state["submit"])}
        )
        st.rerun()

    if st.session_state["mode"] == "submit":
        s = st.session_state["submit"]
        s = submit_apply_answer(s, user_input)
        st.session_state["submit"] = s

        if submit_complete(s):
            st.session_state["chat"].append({"role": "assistant", "content": submit_text_summary(s)})
            st.session_state["mode"] = "qa"
            st.session_state["submit"] = {"_step": 0}
            st.rerun()
        else:
            st.session_state["chat"].append({"role": "assistant", "content": submit_next_question(s)})
            st.rerun()
    else:
        ans = answer_qa(user_input)
        st.session_state["chat"].append({"role": "assistant", "content": ans})
        st.rerun()