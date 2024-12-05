import os
import re
from py2neo import Graph, Node
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter

# Configurer la connexion à la base de données Neo4j
graph = Graph("neo4j+s://your-neo4j-url", auth=("neo4j", "your-password"))

def importer_pdf_et_creer_noeuds(pdf_path):
    """Importe le contenu d'un fichier PDF et crée des nœuds dans Neo4j."""
    try:
        # Charger le PDF
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()
        
        # Diviser le contenu en morceaux
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        docs = text_splitter.split_documents(raw_docs)
        
        # Ajouter chaque morceau de texte comme nœud dans Neo4j
        for doc in docs:
            content = doc.page_content
            node = Node("Document", content=content)
            graph.create(node)
        
        print("Importation réussie et nœuds créés dans Neo4j.")

    except Exception as e:
        print(f"Erreur lors de l'importation du PDF : {e}")

# Spécifier le chemin du fichier PDF
pdf_path = "C:\\Users\\abd_o\\Downloads\\OTBwb3lNdWNWczFzdEk0aUhxRnNOQT09.pdf"

# Appeler la fonction pour importer le PDF et créer des nœuds
importer_pdf_et_creer_noeuds(pdf_path)


