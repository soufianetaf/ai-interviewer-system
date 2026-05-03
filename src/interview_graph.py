from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

# 1. On ajoute un booléen pour savoir si le Juge valide le travail du Recruteur
class InterviewState(TypedDict):
    question_posee: str
    reponse_candidat: str
    contexte_rag: str
    feedback_recruteur: str
    passed_recruteur: bool
    verdict_juge: str
    juge_est_daccord: bool # <-- NOUVEAUTÉ

# --- NŒUD 1 : RAG ---
def agent_rag(state: InterviewState):
    print(" RAG : Recherche de la théorie exacte dans FAISS...")
    from config.settings import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    try:
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(state["question_posee"], k=1)
        vrai_contexte = docs[0].page_content if docs else "Aucune info trouvée."
    except Exception as e:
        vrai_contexte = f"Erreur : {str(e)}"
        
    return {"contexte_rag": vrai_contexte}

# --- NŒUD 2 : LE RECRUTEUR (QUI PEUT ÊTRE GRONDÉ PAR LE JUGE) ---
def agent_1_recruteur(state: InterviewState):
    print(" Agent 1 (Recruteur) : Analyse et rédaction du feedback...")
    
    llm_endpoint = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=256, temperature=0.1)
    llm = ChatHuggingFace(llm=llm_endpoint)
    
    # LA MAGIE : Si on est dans une boucle et que le Juge a fait une critique, on l'injecte au Recruteur !
    remarque_juge = ""
    if state.get("verdict_juge", "") != "":
        remarque_juge = f"\n ATTENTION ! Ton superviseur (Le Juge) a rejeté ton évaluation précédente avec cette critique :\n'{state['verdict_juge']}'\nTu DOIS corriger ton feedback pour le candidat en prenant en compte cette remarque.\n"
    
    template = f"""Tu es un recruteur technique. Évalue le candidat. {remarque_juge}
1. Parle au candidat pour lui donner un feedback.
2. À la fin, écris "DECISION: TRUE" s'il a bon, ou "DECISION: FALSE" s'il a faux.

Théorie : {{verite}}
Réponse du candidat : {{reponse}}

Ton feedback :"""
    
    prompt = PromptTemplate(input_variables=["verite", "reponse"], template=template)
    reponse_ia = (prompt | llm).invoke({"verite": state["contexte_rag"], "reponse": state["reponse_candidat"]})
    
    reponse_text = reponse_ia.content if hasattr(reponse_ia, 'content') else str(reponse_ia)
    
    decision_bool = True if "DECISION: TRUE" in reponse_text.upper() else False
    feedback_propre = reponse_text.replace("DECISION: TRUE", "").replace("DECISION: FALSE", "").strip()
        
    return {"feedback_recruteur": feedback_propre, "passed_recruteur": decision_bool}

# --- NŒUD 3 : LE JUGE (QUI PEUT VALIDER OU REJETER LE RECRUTEUR) ---
def agent_2_juge(state: InterviewState):
    print(" Agent 2 (Juge) : Évaluation stricte du travail du Recruteur...")
    
    llm_endpoint = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=256, temperature=0.1)
    llm = ChatHuggingFace(llm=llm_endpoint)
    
    template = """Tu es le Juge Principal. Tu dois évaluer le travail d'un recruteur (Agent 1).
Le recruteur a décidé que le candidat était : {decision_recruteur}.
Voici ce que le recruteur a dit au candidat : "{feedback_recruteur}"

Théorie officielle : "{verite}"
Réponse du candidat : "{reponse}"

Le recruteur a-t-il pris la bonne décision ? Son feedback est-il juste et précis ?
1. Donne ton analyse.
2. Termine OBLIGATOIREMENT par "SUPERVISION: VALIDE" si le recruteur a bien fait son travail, ou "SUPERVISION: REJETE" s'il s'est trompé, a été injuste ou incomplet.

Ton verdict :"""
    
    prompt = PromptTemplate(input_variables=["decision_recruteur", "feedback_recruteur", "verite", "reponse"], template=template)
    verdict_ia = (prompt | llm).invoke({
        "decision_recruteur": str(state["passed_recruteur"]),
        "feedback_recruteur": state["feedback_recruteur"],
        "verite": state["contexte_rag"],
        "reponse": state["reponse_candidat"]
    })
    
    verdict_text = verdict_ia.content if hasattr(verdict_ia, 'content') else str(verdict_ia)
    
    # Parsing de l'accord du Juge
    juge_ok = True if "SUPERVISION: VALIDE" in verdict_text.upper() else False
    verdict_propre = verdict_text.replace("SUPERVISION: VALIDE", "").replace("SUPERVISION: REJETE", "").strip()
        
    return {"verdict_juge": verdict_propre, "juge_est_daccord": juge_ok}

# --- LA FONCTION D'AIGUILLAGE (QUI CRÉE LA BOUCLE) ---
def route_apres_juge(state: InterviewState):
    if state["juge_est_daccord"] == True:
        print(" [ROUTAGE] Le Juge a validé le travail. Fin du processus.")
        return END
    else:
        print(" [ROUTAGE] Le Juge a REJETÉ l'évaluation ! Renvoi au Recruteur pour correction...")
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
    
    # ON PLACE LA CONDITION APRÈS LE JUGE !
    workflow.add_conditional_edges(
        "Juge",
        route_apres_juge,
        {
            END: END,
            "Recruteur": "Recruteur" # La fameuse boucle !
        }
    )
    
    return workflow.compile()