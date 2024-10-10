import os
import re
import streamlit as st
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import TokenTextSplitter
from py2neo import Graph, Node

# Configurer la clé API OpenAI
os.environ["OPENAI_API_KEY"] = "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"

# Connecter à la base de données Neo4j
graph = Graph("neo4j+s://2a8ea069.databases.neo4j.io", auth=("neo4j", "AG5BTuhBpYS88ZtkCkz92FsMXwgu0TENpL3tBAQH4cE"))


# Définir la classe GraphRAG pour le domaine médical
class GraphRAG:
    def __init__(self):
        self.graph = graph  # Connexion à Neo4j
        self.llm = ChatOpenAI(temperature=0.5)  # Initialiser l'instance LLM ici
        self.create_fulltext_index()  # Créer l'index de texte intégral

    def create_fulltext_index(self):
        """Crée un index de texte intégral pour les entités dans Neo4j."""
        try:
            self.graph.run("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]")
        except Exception as e:
            print(f"Error creating fulltext index: {e}")

    def retriever(self, question):
        # Récupérer des entités médicales et le contexte dans le graphe Neo4j
        result = self.structured_retriever(question)
        context = "Contexte récupéré : " + str(result)
        return context

    def structured_retriever(self, question: str) -> str:
        """Récupère le contexte médical pertinent à partir du graphe de connaissances."""
        # Extraire les entités médicales dans la question via un simple prompt
        entities = self.extract_entities(question)
        
        # Effectuer des requêtes sur Neo4j pour récupérer les nœuds et relations liées aux entités
        context = ""
        for entity in entities:
            clean_entity = self.clean_query(entity)  # Nettoyer l'entité avant de l'utiliser dans la requête
            response = self.graph.run(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node, score
                CALL {
                    WITH node
                    MATCH (node)-[r]->(neighbor)
                    RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                    UNION ALL
                    WITH node
                    MATCH (node)<-[r]-(neighbor)
                    RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50""",
                {"query": clean_entity}
            )
            context += "\n".join([str(el['output']) for el in response if el['output'] is not None])
        return context

    def extract_entities(self, question: str) -> list:
        """Utilise un prompt pour extraire les entités médicales de la question."""
        prompt = ChatPromptTemplate.from_template(
            """Étant donné une question ou une phrase, extrayez toutes les entités médicales importantes telles que les symptômes, les maladies et les traitements.
            Question : {question}
            Répondez avec une liste de mots ou de phrases représentant ces entités."""
        )
        # Obtenir la liste des entités
        response = self.llm.predict(prompt.format(question=question))
        entities = response.split(",")  # Hypothèse : les entités sont séparées par des virgules
        return [entity.strip() for entity in entities]

    def clean_query(self, query: str) -> str:
        """Nettoie la chaîne de requête en supprimant les caractères spéciaux."""
        query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
        return query

# Fonction pour générer la réponse
def get_response(question: str) -> str:
    """Génère une réponse basée sur une question et le contexte médical du graphe de connaissances."""
    chat = ChatOpenAI(temperature=0.5)
    
    # Créer le prompt
    template = """Répondez à la question en vous basant uniquement sur le contexte médical suivant :
    {context}
    
    Question : {question}
    Utilisez un langage médical précis et soyez concis.
    Réponse : """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Récupérer le contexte médical
    rag = GraphRAG()
    context = rag.retriever(question)

    # Générer la réponse
    response = chat.predict(prompt.format(question=question, context=context))
    return response

# Fonction pour l'interface utilisateur
def init_ui():
    """Initialise l'interface utilisateur Streamlit pour le contexte médical."""
    st.set_page_config(page_title="GraphRAG Bot Médical", layout="wide")
    st.title("GraphRAG Bot Médical")

    # Initialiser l'état de session pour stocker l'historique du chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Champ d'entrée pour les questions
    user_query = st.chat_input("Posez une question médicale...")
    if user_query:
        # Ajouter la question de l'utilisateur à l'historique du chat
        st.session_state.chat_history.append({"user": user_query})
        
        # Obtenir la réponse du modèle
        response = get_response(user_query)
        
        # Ajouter la réponse du bot à l'historique du chat
        st.session_state.chat_history.append({"bot": response})
    
    # Afficher l'historique du chat
    for chat in st.session_state.chat_history:
        if "user" in chat:
            with st.chat_message("user"):
                st.write(chat["user"])
        if "bot" in chat:
            with st.chat_message("bot"):
                st.write(chat["bot"])

# Lancer l'application Streamlit
if __name__ == "__main__":
    init_ui()
