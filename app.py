import os
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from py2neo import Graph

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


# Connect to the Neo4j database
graph = Graph("neo4j+s://your-neo4j-url", auth=("neo4j", "your-password"))

# Define a function to create a modern, clickable card with a professional icon from Font Awesome in the sidebar
def create_clickable_card_sidebar(title, description, icon_class, key):
    hover_style = """
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');
    .card {
        display: block;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        background-color: white;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        text-decoration: none;
        color: black;
        font-family: 'Arial', sans-serif;
        text-align: center;
    }
    .card:hover {
        background-color: #f5f5f5;
        transform: translateY(-5px);
        box-shadow: 0px 6px 18px rgba(0, 0, 0, 0.2);
        cursor: pointer;
    }
    .card h3 {
        margin: 10px 0;
        font-size: 1.3em;
        font-weight: bold;
        color: #333;
    }
    .card p {
        margin: 5px 0 0;
        font-size: 0.95em;
        color: #777;
    }
    .card-icon {
        font-size: 40px;
        margin-bottom: 10px;
        color: #4CAF50; /* You can change the color for each card if needed */
    }
    .logo-container {
        text-align: center;
        margin-bottom: 30px;
    }
    .logo-container img {
        width: 150px; /* Adjust size for best look */
        height: auto;
        border-radius: 50%; /* Optionally make it rounded */
    }
    </style>
    """
    st.sidebar.markdown(hover_style, unsafe_allow_html=True)

    # Create a card with a professional Font Awesome icon centered at the top
    card_html = f"""
    <div class="card" onclick="window.location.href='/?card={key}'">
        <div class="card-icon"><i class="{icon_class}"></i></div>
        <h3>{title}</h3>
        <p>{description}</p>
    </div>
    """
    st.sidebar.markdown(card_html, unsafe_allow_html=True)

# Define the GraphRAG class for retrieving context from Neo4j
class GraphRAG:
    def __init__(self):
        self.graph = graph  # Connection to Neo4j
        self.llm = ChatOpenAI(temperature=0.5)  # Initialize LLM instance here
        self.create_fulltext_index()  # Create a full-text index

    def create_fulltext_index(self):
        """Create a full-text index for entities in Neo4j."""
        try:
            self.graph.run("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]")
        except Exception as e:
            print(f"Error creating fulltext index: {e}")

    def retriever(self, question):
        # Retrieve medical entities and context from the Neo4j graph
        result = self.structured_retriever(question)
        context = "Retrieved context: " + str(result)
        return context

    def structured_retriever(self, question: str) -> str:
        """Retrieve relevant medical context from the knowledge graph."""
        entities = self.extract_entities(question)
        context = ""
        for entity in entities:
            clean_entity = self.clean_query(entity)
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
        """Use a prompt to extract medical entities from the question."""
        prompt = ChatPromptTemplate.from_template(
            """Given a question or phrase, extract all important medical entities such as symptoms, diseases, and treatments.
            Question: {question}
            Respond with a list of words or phrases representing these entities."""
        )
        response = self.llm.predict(prompt.format(question=question))
        entities = response.split(",")
        return [entity.strip() for entity in entities]

    def clean_query(self, query: str) -> str:
        """Clean the query string by removing special characters."""
        query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
        return query

# Function to generate a response based on the question and retrieved context
def get_response(question: str) -> str:
    chat = ChatOpenAI(temperature=0.5)
    template = """Respond to the question based solely on the following medical context:
    {context}
    
    Question: {question}
    Use precise medical language and be concise.
    Response: """
    
    prompt = ChatPromptTemplate.from_template(template)
    rag = GraphRAG()
    context = rag.retriever(question)
    response = chat.predict(prompt.format(question=question, context=context))
    return response

# Function for the interactive user interface
def init_ui():
    st.set_page_config(page_title="D&A MedLabs Converse", layout="wide")
    st.title("D&A MedLabs Converse")

    # Sidebar logo
    st.sidebar.markdown("""
    <div class="logo-container">
        <img src="/home/ec2-user/TeamDataFiles/malick/malicklogo.png" alt="Logo">
    </div>
    """, unsafe_allow_html=True)

    # Sidebar title
    st.sidebar.title("Explore Predefined Agents")
 
    # Create clickable cards in the sidebar for each agent with centered icons from Font Awesome
    create_clickable_card_sidebar("Cosmetics Agent", "Interact with a dataset on cosmetics ingredients and their effects", "fas fa-flask", key="cosmetics")
    create_clickable_card_sidebar("Healthcare Agent", "Interact with healthcare data including patient histories", "fas fa-stethoscope", key="healthcare")
    create_clickable_card_sidebar("FoodTech Agent", "Interact with data on food technology innovations", "fas fa-utensils", key="foodtech")

    # Check URL parameters to identify which card was clicked
    query_params = st.experimental_get_query_params()
    selected_card = query_params.get('card', [None])[0]

    # Display appropriate chat interface based on selected card
    if selected_card == "cosmetics":
        st.write("You are interacting with **Cosmetics Agent**. Please ask your question.")
    elif selected_card == "healthcare":
        st.write("You are interacting with **Healthcare Agent**. Please ask your question.")
    elif selected_card == "foodtech":
        st.write("You are interacting with **FoodTech Agent**. Please ask your question.")
    
    # Chat history initialization
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input field for medical questions
    user_query = st.chat_input("Posez une question médicale...")
    if user_query:
        st.session_state.chat_history.append({"user": user_query})
        response = get_response(user_query)
        st.session_state.chat_history.append({"bot": response})

    # Display the chat history
    for chat in st.session_state.chat_history:
        if "user" in chat:
            with st.chat_message("user"):
                st.write(chat["user"])
        if "bot" in chat:
            with st.chat_message("bot"):
                st.write(chat["bot"])

# Launch the Streamlit application
if __name__ == "__main__":
    init_ui()
