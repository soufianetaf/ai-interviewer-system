# AI Interviewer System

Système d'entretien technique automatisé basé sur **LangGraph**, **RAG (FAISS)** et des agents LLM **Qwen2.5**. Le système génère des questions, évalue les réponses du candidat, puis supervise l'évaluation via un Juge indépendant.

---

## Architecture

┌─────────────────────────────────────────────────────────┐
│                    PHASE 1 : GÉNÉRATION                  │
│  START → Recruteur (génère N questions) → STOP           │
│          (le candidat répond manuellement)               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    PHASE 2 : ÉVALUATION                  │
│                                                          │
│  START → RAG → Recruteur → JugeAveugle → JugeComparateur │
│                    ↑__________________________|           │
│                    (si rejeté, max 3 itérations)         │
└─────────────────────────────────────────────────────────┘

---

## Agents

### Agent 1 — Recruteur (Qwen2.5-7B-Instruct)
- Phase 1 : génère N questions sur le sujet donné
- Phase 2 : évalue chaque réponse avec une note /5 en comparant à la théorie RAG
- Reçoit les critiques du Juge en cas de rejet et corrige son évaluation
- Règle de décision : score ≥ 60% → ADMIS, sinon RECALÉ

### Agent 2A — Juge Aveugle (Qwen2.5-72B-Instruct)
- Évalue les réponses du candidat sans voir l'évaluation du Recruteur
- Produit une évaluation totalement indépendante avec ses propres notes /5

### Agent 2B — Juge Comparateur (Qwen2.5-72B-Instruct)
- Compare l'évaluation du Juge Aveugle avec celle du Recruteur
- Valide si : même décision ET écart de notes ≤ 3 points
- Rejette sinon et renvoie le Recruteur corriger son évaluation

### RAG — Base de connaissances (FAISS)
- Recherche la théorie de référence pour chaque question
- Modèle d'embedding : all-MiniLM-L6-v2
- Base vectorielle locale : FAISS

---

## Structure du projet

ai-interviewer-system/
├── config/
│   └── settings.py              # Chemins et paramètres (FAISS, embeddings)
├── src/
│   └── interview_graph.py       # Graphe LangGraph + agents
├── notebooks/
│   ├── 01_build_rag_db.ipynb    # Construction de la base FAISS
│   └── 02_run_interview.ipynb   # Exécution de l'entretien
└── README.md

---

## État partagé (InterviewState)

| Champ                   | Type        | Description                                        |
|-------------------------|-------------|---------------------------------------------------|
| sujet_entretien         | str         | Sujet de l'entretien (ex: "SQL et Bases de Données") |
| nombre_questions        | int         | Nombre de questions à générer                     |
| questions               | List[str]   | Questions générées par le Recruteur               |
| reponses                | List[str]   | Réponses saisies par le candidat                  |
| contextes_rag           | List[str]   | Théorie récupérée par FAISS pour chaque question  |
| feedback_recruteur      | str         | Bilan détaillé du Recruteur                       |
| passed_recruteur        | bool        | Décision du Recruteur (True = ADMIS)              |
| notes_juge_independant  | str         | Évaluation aveugle du Juge                        |
| verdict_juge            | str         | Verdict final du Juge Comparateur                 |
| juge_est_daccord        | bool        | True si le Juge valide le Recruteur               |
| iteration               | int         | Compteur de boucles Recruteur ↔ Juge              |
| historique_juge         | List[str]   | Log des rejets du Juge avec justifications        |
| ecart_total             | int         | Écart de notation entre Juge et Recruteur         |

---

## Installation

pip install langgraph langchain-community langchain-huggingface faiss-cpu sentence-transformers
pip install -U typing_extensions pydantic langchain-core

---

## Configuration

Modifier config/settings.py :

FAISS_INDEX_PATH    = "/chemin/vers/vector_store/faiss_index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME     = "connaissances_techniques"

---

## Utilisation

### Étape 1 — Construire la base de connaissances
Exécuter notebooks/01_build_rag_db.ipynb

### Étape 2 — Lancer l'entretien
Exécuter notebooks/02_run_interview.ipynb

etat_initial = {
    "sujet_entretien": "SQL et Bases de Données",
    "nombre_questions": 3,
    "questions": [], "reponses": [], "contextes_rag": [],
    "feedback_recruteur": "", "passed_recruteur": False,
    "notes_juge_independant": "", "verdict_juge": "",
    "juge_est_daccord": False, "iteration": 0,
    "historique_juge": [], "ecart_total": 0,
}

---

## Flux de supervision

Recruteur → JugeAveugle (seul) → JugeComparateur
                                        │
                   ┌────────────────────┤
                   │                    │
             Écart ≤ 3 et          Écart > 3 ou
           même décision        décisions différentes
                   │                    │
            SUPERVISION:          SUPERVISION:
              VALIDE ✅             REJETE ❌
                   │                    │
                  END         Recruteur corrige (max 3x)

---

## Règle de décision

Seuil : score ≥ 60% du score maximum → ADMIS

| Score | %   | Décision  |
|-------|-----|-----------|
| 9/15  | 60% | ✅ ADMIS  |
| 8/15  | 53% | ❌ RECALÉ |

---

## Améliorations futures

- Enrichir la base FAISS (20+ documents par sujet)
- Remplacer input() par dbutils.widgets sur Databricks
- Sauvegarder les résultats en JSON ou base de données
- Ajouter des tests unitaires sur chaque nœud
- Afficher le score numérique final dans le notebook

##Soufiane TAFAHI