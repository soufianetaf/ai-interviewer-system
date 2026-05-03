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
        # On boucle sur chaque question pour trouver la théorie correspondante
        for q in state["questions"]:
            docs = vectorstore.similarity_search(q, k=1)
            contextes_trouves.append(docs[0].page_content if docs else "Aucune info.")
    except Exception as e:
        contextes_trouves = [f"Erreur : {str(e)}"] * len(state["questions"])
        
    contextes_trouves = []
    # On boucle sur chaque question présente dans l'état
    for q in state["questions"]:
        if q.strip(): # On ignore les lignes vides
            docs = vectorstore.similarity_search(q, k=1)
            contextes_trouves.append(docs[0].page_content if docs else "Théorie non trouvée.")
    
    return {"contextes_rag": contextes_trouves}



# --- NŒUD 2 : L'AGENT 1 (MAÎTRE DE L'ENTRETIEN) ---
def agent_1_recruteur(state: InterviewState):
    llm_endpoint = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=512, temperature=0.3)
    llm = ChatHuggingFace(llm=llm_endpoint)

    # PHASE 1 : GÉNÉRATION (Si aucune réponse n'est présente)
    if not state.get("reponses") :
        print(f" Agent 1 : Génération de {state['nombre_questions']} questions sur '{state['sujet_entretien']}'...")
        
        template = """Tu es un recruteur technique. Génère {n} questions sur {sujet}.
        Format STRICT : une question par ligne commençant par 'Q:'. Rien d'autre."""
        
        prompt = PromptTemplate(input_variables=["n", "sujet"], template=template)
        reponse = (prompt | llm).invoke({"n": state["nombre_questions"], "sujet": state["sujet_entretien"]})
        
        questions = [q.replace("Q:", "").strip() for q in reponse.content.split("\n") if "Q:" in q]
        return {"questions": questions}

    # PHASE 2 : ÉVALUATION (Si les réponses sont là)
    else:
        print(" Agent 1 : Analyse des réponses et verdict final...")
        
        nb_a_analyser = min(len(state["questions"]), len(state["reponses"]), len(state["contextes_rag"]))
        
        transcript = ""
        for i in range(nb_a_analyser):
            transcript += f"\nQ: {state['questions'][i]}\nR: {state['reponses'][i]}\nThéorie: {state['contextes_rag'][i]}\n"
        
        template_eval = f"""Évalue cet entretien. Sois précis.
        {transcript}
        Termine par DECISION: TRUE ou DECISION: FALSE.
        Bilan :"""
        
        reponse_ia = llm.invoke(template_eval)
        text = reponse_ia.content
        return {
            "feedback_recruteur": text.replace("DECISION: TRUE", "").replace("DECISION: FALSE", "").strip(),
            "passed_recruteur": "DECISION: TRUE" in text.upper()
        }

# --- NŒUD 3 : LE JUGE (SUPERVISION FINALE) ---
def agent_2_juge(state: InterviewState):
    print(" Agent 2 (Juge) : Évaluation de la décision finale du Recruteur...")
    
    llm_endpoint = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=512, temperature=0.1)
    llm = ChatHuggingFace(llm=llm_endpoint)
    
    transcript = ""
    for i in range(len(state["questions"])):
        transcript += f"\nQ{i+1}: {state['questions'][i]}\nRéponse: {state['reponses'][i]}\nThéorie: {state['contextes_rag'][i]}\n"
    
    template = """Tu es le Directeur RH (Juge). Tu supervises l'Agent 1.
Le recruteur a décidé que le candidat était globalement : {decision_recruteur} (True = Embauché, False = Recalé).
Voici son bilan : "{feedback_recruteur}"

Voici la transcription de l'entretien :
{transcript}

Le recruteur a-t-il pris la bonne décision finale ? Son bilan est-il juste au vu de toutes les réponses ?
Termine OBLIGATOIREMENT par "SUPERVISION: VALIDE" s'il a bien fait son travail, ou "SUPERVISION: REJETE" s'il a pris une mauvaise décision d'embauche.

Ton verdict :"""
    
    prompt = PromptTemplate(input_variables=["decision_recruteur", "feedback_recruteur", "transcript"], template=template)
    verdict_ia = (prompt | llm).invoke({
        "decision_recruteur": str(state["passed_recruteur"]),
        "feedback_recruteur": state["feedback_recruteur"],
        "transcript": transcript
    })
    
    verdict_text = verdict_ia.content if hasattr(verdict_ia, 'content') else str(verdict_ia)
    
    juge_ok = True if "SUPERVISION: VALIDE" in verdict_text.upper() else False
    verdict_propre = verdict_text.replace("SUPERVISION: VALIDE", "").replace("SUPERVISION: REJETE", "").strip()
        
    return {"verdict_juge": verdict_propre, "juge_est_daccord": juge_ok}

# --- AIGUILLAGE ---
def route_apres_juge(state: InterviewState):
    if state["juge_est_daccord"] == True:
        return END
    else:
        return "Recruteur"


def route_apres_recruteur(state: InterviewState):
    # Si on vient de générer les questions et qu'on n'a pas encore de réponses -> STOP
    if state.get("questions") and not state.get("reponses"):
        return END
    # Sinon, on continue vers le RAG puis le Juge
    return "RAG"
# --- CONSTRUCTION DU GRAPHE ---
def build_interview_graph():
    workflow = StateGraph(InterviewState)
    
    workflow.add_node("Recruteur", agent_1_recruteur)
    workflow.add_node("RAG", agent_rag)
    workflow.add_node("Juge", agent_2_juge)
    
    workflow.add_edge(START, "Recruteur")
    
    # Aiguillage : Stop après génération OU continue vers évaluation
    workflow.add_conditional_edges(
        "Recruteur",
        route_apres_recruteur,
        {"RAG": "RAG", END: END}
    )
    
    workflow.add_edge("RAG", "Juge")
    workflow.add_conditional_edges("Juge", route_apres_juge, {END: END, "Recruteur": "Recruteur"})
    
    return workflow.compile()