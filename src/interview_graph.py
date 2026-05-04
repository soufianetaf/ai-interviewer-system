from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate


# --- ÉTAT PARTAGÉ ---
class InterviewState(TypedDict):
    sujet_entretien: str
    nombre_questions: int
    questions: List[str]
    reponses: List[str]
    contextes_rag: List[str]
    feedback_recruteur: str
    passed_recruteur: bool
    verdict_juge: str
    juge_est_daccord: bool
    iteration: int
    historique_juge: List[str]
    notes_juge_independant: str   # ← NEW : évaluation aveugle du Juge
    ecart_total: int              # ← NEW : écart calculé en Python

# --- NŒUD 1 : RAG EN LOT ---
def agent_rag(state: InterviewState):
    print(" RAG : Recherche de la théorie pour CHAQUE question...")
    from config.settings import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    contextes_trouves = []

    try:
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        for q in state["questions"]:
            if q.strip():
                docs = vectorstore.similarity_search(q, k=1)
                contextes_trouves.append(
                    docs[0].page_content if docs else "Théorie non trouvée."
                )
    except Exception as e:
        contextes_trouves = [f"Erreur : {str(e)}"] * len(state["questions"])

    return {"contextes_rag": contextes_trouves}


# --- NŒUD 2 : L'AGENT 1 (RECRUTEUR) ---
def agent_1_recruteur(state: InterviewState):
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=1024,
        temperature=0.3,
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    # PHASE 1 : GÉNÉRATION (aucune réponse → on génère les questions)
    if not state.get("reponses"):
        print(f" Agent 1 : Génération de {state['nombre_questions']} questions sur '{state['sujet_entretien']}'...")

        template = """Tu es un recruteur technique. Génère {n} questions sur {sujet}.
Format STRICT : une question par ligne commençant par 'Q:'. Rien d'autre."""

        prompt = PromptTemplate(input_variables=["n", "sujet"], template=template)
        reponse = (prompt | llm).invoke({
            "n": state["nombre_questions"],
            "sujet": state["sujet_entretien"],
        })

        questions = [
            q.replace("Q:", "").strip()
            for q in reponse.content.split("\n")
            if "Q:" in q
        ]
        return {"questions": questions, "iteration": 0, "historique_juge": []}

    # PHASE 2 : ÉVALUATION
    iteration = state.get("iteration", 0) + 1
    print(f"\n ITÉRATION #{iteration} — Agent 1 : analyse des réponses...")

    # Garde-fou : stop après 3 boucles
    if iteration > 3:
        print("  Limite de 3 itérations atteinte, arrêt forcé.")
        return {
            "feedback_recruteur": state.get("feedback_recruteur", "Limite atteinte."),
            "passed_recruteur": state.get("passed_recruteur", False),
            "iteration": iteration,
        }

    nb_a_analyser = min(
        len(state["questions"]),
        len(state["reponses"]),
        len(state["contextes_rag"]),
    )
    score_max = nb_a_analyser * 5

    transcript = ""
    for i in range(nb_a_analyser):
        transcript += (
            f"\n--- Question {i+1} ---"
            f"\nQuestion    : {state['questions'][i]}"
            f"\nRéponse     : {state['reponses'][i]}"
            f"\nThéorie RAG : {state['contextes_rag'][i]}\n"
        )

    # Inclure les critiques du Juge si rejet précédent
    historique = state.get("historique_juge", [])
    feedback_juge_precedent = ""
    if historique:
        feedback_juge_precedent = (
            "\n Le Juge a rejeté ta précédente évaluation. Voici ses critiques :\n"
            + "\n".join(f"- {h}" for h in historique)
            + "\nCorrige ces erreurs dans cette nouvelle évaluation.\n"
        )

    template_eval = """Tu es un recruteur technique expert et rigoureux.
Évalue chaque réponse du candidat en la comparant à la théorie de référence.
{feedback_juge_precedent}
{transcript}

Pour CHAQUE question, fournis :
  - Note : X/5
  - Points corrects : ce que le candidat a bien dit
  - Points manquants ou erronés : ce qui est faux ou absent par rapport à la théorie
  - Commentaire : appréciation courte

Ensuite, donne un bilan global :
  - Score total : X/{score_max}
  - Forces du candidat : ...
  - Faiblesses du candidat : ...
  - Recommandation : ...
Règle de décision : calcule (score total / score maximum) * 100. Si ce résultat est supérieur ou ÉGAL à 60, termine par DECISION: TRUE. Si inférieur à 60, termine par DECISION: FALSE. Exemple : 9/15 = 60% → DECISION: TRUE.

Évaluation :"""

    prompt = PromptTemplate(
        input_variables=["transcript", "score_max", "feedback_juge_precedent"],
        template=template_eval,
    )
    reponse_ia = (prompt | llm).invoke({
        "transcript": transcript,
        "score_max": score_max,
        "feedback_juge_precedent": feedback_juge_precedent,
    })
    text = reponse_ia.content
    decision = "DECISION: TRUE" in text.upper()
    print(f"   → Décision Recruteur : {' EMBAUCHE' if decision else ' REJET'}")

    return {
        "feedback_recruteur": text.replace("DECISION: TRUE", "").replace("DECISION: FALSE", "").strip(),
        "passed_recruteur": decision,
        "iteration": iteration,
    }


# --- NŒUD 3 : LE JUGE (SUPERVISION INDÉPENDANTE) ---
def agent_juge_aveugle(state: InterviewState):
    iteration = state.get("iteration", 1)
    print(f" Juge Aveugle (itération #{iteration}) : évaluation indépendante...")

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        max_new_tokens=1024,
        temperature=0.2,
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    nb = min(len(state["questions"]), len(state["reponses"]), len(state["contextes_rag"]))
    score_max = nb * 5

    transcript = ""
    for i in range(nb):
        transcript += (
            f"\n--- Question {i+1} ---"
            f"\nQuestion    : {state['questions'][i]}"
            f"\nRéponse     : {state['reponses'][i]}"
            f"\nThéorie RAG : {state['contextes_rag'][i]}\n"
        )

    template = """Tu es un évaluateur indépendant. Tu ne connais pas l'avis du Recruteur.
Évalue chaque réponse UNIQUEMENT en te basant sur la théorie de référence.

{transcript}

Pour chaque question, donne :
  - Ma note : X/5
  - Justification courte

Ensuite :
  - Mon score total : X/{score_max}
  - Ma décision : EMBAUCHE si (score / score_max * 100) >= 60, sinon REJET

Évaluation indépendante :"""

    prompt = PromptTemplate(input_variables=["transcript", "score_max"], template=template)
    result = (prompt | llm).invoke({"transcript": transcript, "score_max": score_max})

    return {"notes_juge_independant": result.content}
def agent_juge_comparateur(state: InterviewState):
    iteration = state.get("iteration", 1)
    print(f"  Juge Comparateur (itération #{iteration}) : comparaison en cours...")

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        max_new_tokens=1024,
        temperature=0.1,
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    template = """Tu es le Directeur RH. Compare ces deux évaluations indépendantes.

=== TON ÉVALUATION (faite sans voir le Recruteur) ===
{notes_juge_independant}

=== ÉVALUATION DU RECRUTEUR ===
Décision : {decision_recruteur}
Bilan : {feedback_recruteur}

Ta mission :
1. Identifie les écarts de notation entre toi et le Recruteur
2. Vérifie si vos décisions finales sont identiques
3. Signale toute incohérence ou biais dans l'évaluation du Recruteur

Règle stricte :
- "SUPERVISION: VALIDE" si vos décisions sont identiques ET l'écart total <= 3 points
- "SUPERVISION: REJETE" sinon, avec explication précise des corrections à faire

Verdict :"""

    prompt = PromptTemplate(
        input_variables=["notes_juge_independant", "decision_recruteur", "feedback_recruteur"],
        template=template,
    )
    result = (prompt | llm).invoke({
        "notes_juge_independant": state["notes_juge_independant"],
        "decision_recruteur": "EMBAUCHE" if state["passed_recruteur"] else "REJET",
        "feedback_recruteur": state["feedback_recruteur"],
    })

    verdict_text = result.content if hasattr(result, "content") else str(result)
    juge_ok = "SUPERVISION: VALIDE" in verdict_text.upper()
    verdict_propre = (
        verdict_text
        .replace("SUPERVISION: VALIDE", "")
        .replace("SUPERVISION: REJETE", "")
        .strip()
    )

    historique = list(state.get("historique_juge", []))
    if juge_ok:
        print(f"    Juge a VALIDÉ (itération {iteration}).")
    else:
        historique.append(f"Itération {iteration} : REJETÉ — {verdict_propre[:200].replace(chr(10), ' ')}")
        print(f"    Juge a REJETÉ (itération {iteration}). Renvoi au Recruteur.")

    return {
        "verdict_juge": verdict_propre,
        "juge_est_daccord": juge_ok,
        "historique_juge": historique,
    }

def route_depuis_start(state: InterviewState):
    # Phase 2 : réponses présentes → RAG d'abord
    if state.get("questions") and state.get("reponses"):
        return "RAG"
    # Phase 1 : génération des questions
    return "Recruteur"


def route_apres_recruteur(state: InterviewState):
    # Phase 1 : questions générées, pas encore de réponses → STOP
    if state.get("questions") and not state.get("reponses"):
        return END
    # Phase 2 : évaluation faite → Juge
    return "Juge"


def route_apres_juge(state: InterviewState):
    if state["juge_est_daccord"]:
        return END
    # Garde-fou : stop forcé après 3 itérations
    if state.get("iteration", 0) >= 3:
        print("  Limite de 3 itérations atteinte. Arrêt forcé.")
        return END
    return "Recruteur"


# --- CONSTRUCTION DU GRAPHE ---
def build_interview_graph():
    workflow = StateGraph(InterviewState)

    workflow.add_node("Recruteur", agent_1_recruteur)
    workflow.add_node("RAG", agent_rag)
    workflow.add_node("JugeAveugle", agent_juge_aveugle)       # ← NEW
    workflow.add_node("JugeComparateur", agent_juge_comparateur)  # ← NEW

    workflow.add_conditional_edges(
        START,
        route_depuis_start,
        {"Recruteur": "Recruteur", "RAG": "RAG"}
    )

    workflow.add_edge("RAG", "Recruteur")

    workflow.add_conditional_edges(
        "Recruteur",
        route_apres_recruteur,
        {END: END, "Juge": "JugeAveugle"}   # ← pointe vers JugeAveugle
    )

    workflow.add_edge("JugeAveugle", "JugeComparateur")  # ← NEW : aveugle → comparateur

    workflow.add_conditional_edges(
        "JugeComparateur",
        route_apres_juge,
        {END: END, "Recruteur": "Recruteur"}
    )

    return workflow.compile()