from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

# 1. La mémoire : On stocke les actions du Recruteur ET du Juge
class InterviewState(TypedDict):
    question_posee: str
    reponse_candidat: str
    contexte_rag: str
    
    # Sorties de l'Agent 1 (Recruteur)
    feedback_recruteur: str
    passed_recruteur: bool
    
    # Sorties de l'Agent 2 (Juge)
    verdict_juge: str

# --- NŒUD 1 : LA RECHERCHE DE VÉRITÉ (RAG) ---
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

# --- NŒUD 2 : L'AGENT 1 (LE RECRUTEUR) ---
def agent_1_recruteur(state: InterviewState):
    print(" Agent 1 (Recruteur) : Analyse et prise de décision...")
    
    # Pour les modèles conversationnels, utiliser ChatHuggingFace wrapper
    llm_endpoint = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=256, temperature=0.1)
    llm = ChatHuggingFace(llm=llm_endpoint)
    
    template = """Tu es un recruteur technique. Évalue le candidat.
1. Parle au candidat pour lui donner un feedback.
2. À la fin, écris "DECISION: TRUE" s'il a bon, ou "DECISION: FALSE" s'il a faux.

Théorie : {verite}
Réponse du candidat : {reponse}

Ton feedback :"""
    
    prompt = PromptTemplate(input_variables=["verite", "reponse"], template=template)
    reponse_ia = (prompt | llm).invoke({"verite": state["contexte_rag"], "reponse": state["reponse_candidat"]})
    
    # Extraction du contenu de la réponse
    reponse_text = reponse_ia.content if hasattr(reponse_ia, 'content') else str(reponse_ia)
    
    # Parsing de la décision
    decision_bool = True if "DECISION: TRUE" in reponse_text.upper() else False
    feedback_propre = reponse_text.replace("DECISION: TRUE", "").replace("DECISION: FALSE", "").strip()
        
    return {"feedback_recruteur": feedback_propre, "passed_recruteur": decision_bool}

# --- NŒUD 3 : L'AGENT 2 (LE JUGE DU RECRUTEUR) ---
def agent_2_juge(state: InterviewState):
    print(" Agent 2 (Juge) : Évaluation du travail du Recruteur...")
    
    # Pour les modèles conversationnels, utiliser ChatHuggingFace wrapper
    llm_endpoint = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=256, temperature=0.1)
    llm = ChatHuggingFace(llm=llm_endpoint)
    
    template = """Tu es le Juge Principal. Tu dois évaluer le travail d'un recruteur (Agent 1).
Le recruteur a décidé que le candidat était : {decision_recruteur}.
Voici ce que le recruteur a dit au candidat : "{feedback_recruteur}"

Sachant que la théorie officielle est : "{verite}"
Et que le candidat a répondu : "{reponse}"

Le recruteur a-t-il pris la bonne décision (True/False) et donné un bon feedback ? Juge le recruteur sévèrement.

Ton verdict sur le recruteur :"""
    
    prompt = PromptTemplate(input_variables=["decision_recruteur", "feedback_recruteur", "verite", "reponse"], template=template)
    
    verdict_ia = (prompt | llm).invoke({
        "decision_recruteur": str(state["passed_recruteur"]),
        "feedback_recruteur": state["feedback_recruteur"],
        "verite": state["contexte_rag"],
        "reponse": state["reponse_candidat"]
    })
    
    # Extraction du contenu de la réponse
    verdict_text = verdict_ia.content if hasattr(verdict_ia, 'content') else str(verdict_ia)
        
    return {"verdict_juge": verdict_text.strip()}

# --- CONSTRUCTION DU GRAPHE ---
def build_interview_graph():
    workflow = StateGraph(InterviewState)
    
    workflow.add_node("RAG", agent_rag)
    workflow.add_node("Recruteur", agent_1_recruteur)
    workflow.add_node("Juge", agent_2_juge)
    
    # L'ordre d'exécution !
    workflow.add_edge(START, "RAG")
    workflow.add_edge("RAG", "Recruteur")
    workflow.add_edge("Recruteur", "Juge") # Le Juge passe toujours après le Recruteur
    workflow.add_edge("Juge", END)
    
    return workflow.compile()