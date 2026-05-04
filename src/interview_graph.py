from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

# 1. La mémoire : On accepte maintenant des LISTES de questions et réponses
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

# --- NŒUD 1 : RAG EN LOT ---
def agent_rag(state: InterviewState):
    print(" RAG : Recherche de la théorie pour CHAQUE question...")
    from config.settings import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    contextes_trouves = []

    try:
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        for q in state["questions"]:
            if q.strip():
                docs = vectorstore.similarity_search(q, k=1)
                contextes_trouves.append(docs[0].page_content if docs else "Théorie non trouvée.")
    except Exception as e:
        contextes_trouves = [f"Erreur : {str(e)}"] * len(state["questions"])

    return {"contextes_rag": contextes_trouves}
# --- NŒUD 2 : L'AGENT 1 (MAÎTRE DE L'ENTRETIEN) ---
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

        questions = [q.replace("Q:", "").strip() for q in reponse.content.split("\n") if "Q:" in q]
        return {"questions": questions, "iteration": 0}

    # PHASE 2 : ÉVALUATION
    iteration = state.get("iteration", 0) + 1
    print(f"\n🔁 ITÉRATION #{iteration} — Agent 1 : analyse des réponses...")

    # Garde-fou : stop après 3 boucles
    if iteration > 3:
        print("  Limite de 3 itérations atteinte, arrêt forcé.")
        return {
            "feedback_recruteur": state.get("feedback_recruteur", "Limite d'itérations atteinte."),
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

    # On inclut l'historique des rejets du Juge pour que le Recruteur s'améliore
    historique = state.get("historique_juge", [])
    feedback_juge_precedent = ""
    if historique:
        feedback_juge_precedent = (
            "\n Le Juge a rejeté ta précédente évaluation. Voici ses critiques :\n"
            + "\n".join(f"- {h}" for h in historique)
            + "\nCorrige tes erreurs dans cette nouvelle évaluation.\n"
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

Règle de décision : si le score total est >= 60% du score maximum, termine par DECISION: TRUE, sinon DECISION: FALSE.

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
def agent_2_juge(state: InterviewState):
    iteration = state.get("iteration", 1)
    print(f"  Juge (itération #{iteration}) : évaluation indépendante en cours...")

    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=1500,
        temperature=0.2,   # un peu plus haut pour favoriser une opinion propre
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    nb = min(
        len(state["questions"]),
        len(state["reponses"]),
        len(state["contextes_rag"]),
    )
    score_max = nb * 5

    transcript = ""
    for i in range(nb):
        transcript += (
            f"\n--- Question {i+1} ---"
            f"\nQuestion    : {state['questions'][i]}"
            f"\nRéponse     : {state['reponses'][i]}"
            f"\nThéorie RAG : {state['contextes_rag'][i]}\n"
        )

    template = """Tu es le Directeur RH. Tu dois superviser un Recruteur, mais ATTENTION :
ton rôle n'est PAS de simplement valider son travail. Tu dois être CRITIQUE et INDÉPENDANT.

==================== ÉTAPE 1 : TON ÉVALUATION INDÉPENDANTE ====================
AVANT de regarder ce qu'a dit le Recruteur, fais TA PROPRE évaluation des réponses.
Note chaque réponse de 0 à 5 en te basant UNIQUEMENT sur la théorie de référence.

Transcription de l'entretien :
{transcript}

Donne tes propres notes :
- Q1 : ma note = X/5 — justification courte
- Q2 : ma note = X/5 — justification courte
- ... (etc pour chaque question)
- Mon score total indépendant = X/{score_max}
- Ma décision indépendante : EMBAUCHE (si >= 60%) ou REJET

==================== ÉTAPE 2 : COMPARAISON AVEC LE RECRUTEUR ====================
Maintenant, voici ce qu'a dit le Recruteur :
- Sa décision : {decision_recruteur}
- Son bilan : {feedback_recruteur}

Compare TES notes avec les siennes :
- Y a-t-il des écarts dans la notation ? Lesquels précisément ?
- Le Recruteur a-t-il été trop sévère ou trop indulgent sur certaines questions ?
- Sa décision finale correspond-elle à ce que tu aurais décidé ?

==================== ÉTAPE 3 : VERDICT ====================
Règle stricte de validation :
- "SUPERVISION: VALIDE" SEULEMENT si :
    1) Ta décision finale et celle du Recruteur sont identiques
    2) L'écart total de notation entre tes notes et les siennes est ≤ 3 points
- "SUPERVISION: REJETE" sinon. Dans ce cas, indique CLAIREMENT :
    - Quelles questions le Recruteur doit revoir
    - Si la décision finale est trop sévère ou trop indulgente

Verdict du Directeur RH :"""

    prompt = PromptTemplate(
        input_variables=["transcript", "feedback_recruteur", "decision_recruteur", "score_max"],
        template=template,
    )
    verdict_ia = (prompt | llm).invoke({
        "transcript": transcript,
        "feedback_recruteur": state["feedback_recruteur"],
        "decision_recruteur": "EMBAUCHE" if state["passed_recruteur"] else "REJET",
        "score_max": score_max,
    })

    verdict_text = verdict_ia.content if hasattr(verdict_ia, "content") else str(verdict_ia)

    juge_ok = "SUPERVISION: VALIDE" in verdict_text.upper()
    verdict_propre = (
        verdict_text
        .replace("SUPERVISION: VALIDE", "")
        .replace("SUPERVISION: REJETE", "")
        .strip()
    )

    # Logger le verdict
    historique = list(state.get("historique_juge", []))
    if juge_ok:
        print(f"    Juge a VALIDÉ (itération {iteration}).")
    else:
        # On garde un résumé court du rejet
        resume_rejet = verdict_propre[:300].replace("\n", " ")
        historique.append(f"Itération {iteration} : REJETÉ — {resume_rejet}")
        print(f"    Juge a REJETÉ (itération {iteration}). Renvoi au Recruteur.")

    return {
        "verdict_juge": verdict_propre,
        "juge_est_daccord": juge_ok,
        "historique_juge": historique,
    }
# --- AIGUILLAGE ---
def route_apres_juge(state: InterviewState):
    if state["juge_est_daccord"]:
        return END
    if state.get("iteration", 0) >= 3:   # ← stop après 3 boucles
        print(" Max d'itérations atteint, arrêt forcé.")
        return END
    return "Recruteur"

def route_depuis_start(state: InterviewState):
    """Décide du point d'entrée selon la phase."""
    # Phase 2 : on a déjà des questions ET des réponses → on commence par le RAG
    if state.get("questions") and state.get("reponses"):
        return "RAG"
    # Phase 1 : génération des questions
    return "Recruteur"


def route_apres_recruteur(state: InterviewState):
    # Phase 1 : questions générées sans réponses → STOP (on attend l'humain)
    if state.get("questions") and not state.get("reponses"):
        return END
    # Phase 2 : évaluation faite → on va au Juge
    return "Juge"


# --- CONSTRUCTION DU GRAPHE ---
def build_interview_graph():
    workflow = StateGraph(InterviewState)

    workflow.add_node("Recruteur", agent_1_recruteur)
    workflow.add_node("RAG", agent_rag)
    workflow.add_node("Juge", agent_2_juge)

    # Routage initial : Phase 1 (gen) ou Phase 2 (éval avec RAG d'abord)
    workflow.add_conditional_edges(
        START,
        route_depuis_start,
        {"Recruteur": "Recruteur", "RAG": "RAG"}
    )

    # RAG → Recruteur (pour l'évaluation avec les contextes)
    workflow.add_edge("RAG", "Recruteur")

    # Recruteur → END (Phase 1) ou Juge (Phase 2)
    workflow.add_conditional_edges(
        "Recruteur",
        route_apres_recruteur,
        {END: END, "Juge": "Juge"}
    )

    # Juge → END ou retour Recruteur (re-évaluation, RAG déjà fait)
    workflow.add_conditional_edges(
        "Juge",
        route_apres_juge,
        {END: END, "Recruteur": "Recruteur"}
    )

    return workflow.compile()