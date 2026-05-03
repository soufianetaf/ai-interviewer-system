# src/interview_graph.py

from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Définition de la mémoire partagée (L'État du Graphe)
class InterviewState(TypedDict):
    question_posee: str
    reponse_candidat: str
    contexte_rag: str
    evaluation_finale: str

# 2. Le Nœud du Juge (Le RAG)
def agent_juge_rag(state: InterviewState):
    print("👨 Juge : Recherche de la vérité dans la base de connaissances (FAISS)...")
    
    # On charge la base FAISS que l'on a créée tout à l'heure
    from config.settings import FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    try:
        # allow_dangerous_deserialization est requis pour lire les fichiers .pkl locaux en toute sécurité
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # On cherche la vraie réponse par rapport à la question posée
        docs = vectorstore.similarity_search(state["question_posee"], k=1)
        vrai_contexte = docs[0].page_content if docs else "Aucune info trouvée."
    except Exception as e:
        vrai_contexte = f"Erreur de lecture de la base : {str(e)}"
        
    # On met à jour la mémoire avec ce qu'on a trouvé
    return {"contexte_rag": vrai_contexte}

# 3. Le Nœud du Recruteur (L'évaluateur LLM)
def agent_recruteur_llm(state: InterviewState):
    print(" Recruteur : Analyse de la réponse du candidat...")
    
    # Ici, nous simulerons l'appel au LLM (HuggingFace) pour l'instant
    # Dans la prochaine étape, on branchera le vrai modèle !
    
    reponse_candidat = state["reponse_candidat"]
    verite = state["contexte_rag"]
    
    # Logique simplifiée en attendant le LLM
    if len(reponse_candidat) > 10:
        feedback = f"Bonne tentative. \n La théorie exacte dit : '{verite}'"
    else:
        feedback = f"Réponse trop courte. \n La théorie exacte dit : '{verite}'"
        
    return {"evaluation_finale": feedback}

# 4. Construction du Circuit (Le Graphe)
def build_interview_graph():
    # On initialise le graphe avec notre structure de mémoire
    workflow = StateGraph(InterviewState)
    
    # On ajoute nos deux agents
    workflow.add_node("Juge", agent_juge_rag)
    workflow.add_node("Recruteur", agent_recruteur_llm)
    
    # On définit le sens de circulation
    workflow.add_edge(START, "Juge")        # 1. On commence par chercher la vérité
    workflow.add_edge("Juge", "Recruteur")  # 2. Puis on donne la vérité et la réponse au recruteur
    workflow.add_edge("Recruteur", END)     # 3. Fin de l'évaluation
    
    # On compile le tout
    return workflow.compile()