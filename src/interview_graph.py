from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

# 1. La mémoire : On accepte maintenant des LISTES de questions et réponses
class InterviewState(TypedDict):
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
        
    return {"contextes_rag": contextes_trouves}

# --- NŒUD 2 : LE RECRUTEUR (ÉVALUATION GLOBALE) ---
def agent_1_recruteur(state: InterviewState):
    print(" Agent 1 (Recruteur) : Analyse globale de tout l'entretien...")
    
    llm_endpoint = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=512, temperature=0.1)
    llm = ChatHuggingFace(llm=llm_endpoint)
    
    # On prépare le format texte de l'entretien
    transcript = ""
    for i in range(len(state["questions"])):
        transcript += f"\nQ{i+1}: {state['questions'][i]}\nRéponse: {state['reponses'][i]}\nThéorie: {state['contextes_rag'][i]}\n"
    
    remarque_juge = ""
    if state.get("verdict_juge", "") != "":
        remarque_juge = f"\n Le Juge a rejeté ton évaluation globale avec cette critique :\n'{state['verdict_juge']}'\nCorrige ton verdict !\n"
    
    template = f"""Tu es un recruteur technique principal. Évalue l'ensemble de l'entretien du candidat. {remarque_juge}
Voici la transcription de l'entretien complet (Questions, Réponses du candidat, et la Théorie officielle) :
{transcript}

Mission :
1. Fais un bilan global au candidat sur ses forces et ses erreurs durant cet entretien.
2. À la toute fin, écris "DECISION: TRUE" si le candidat mérite le poste globalement (plus de bonnes que de mauvaises réponses), ou "DECISION: FALSE" s'il a trop de lacunes.

Ton bilan global :"""
    
    prompt = PromptTemplate(input_variables=[], template=template)
    reponse_ia = (prompt | llm).invoke({})
    
    reponse_text = reponse_ia.content if hasattr(reponse_ia, 'content') else str(reponse_ia)
    
    decision_bool = True if "DECISION: TRUE" in reponse_text.upper() else False
    feedback_propre = reponse_text.replace("DECISION: TRUE", "").replace("DECISION: FALSE", "").strip()
        
    return {"feedback_recruteur": feedback_propre, "passed_recruteur": decision_bool}

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

# --- CONSTRUCTION DU GRAPHE ---
def build_interview_graph():
    workflow = StateGraph(InterviewState)
    workflow.add_node("RAG", agent_rag)
    workflow.add_node("Recruteur", agent_1_recruteur)
    workflow.add_node("Juge", agent_2_juge)
    
    workflow.add_edge(START, "RAG")
    workflow.add_edge("RAG", "Recruteur")
    workflow.add_edge("Recruteur", "Juge") 
    workflow.add_conditional_edges("Juge", route_apres_juge, {END: END, "Recruteur": "Recruteur"})
    
    return workflow.compile()