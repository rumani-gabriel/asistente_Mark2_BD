import streamlit as st
import sqlite3
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# import pickle

# Carga las variables de entorno y configura la API de Google
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Inicializa embeddings una sola vez
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Función para inicializar la base de datos
def init_db():
    conn = sqlite3.connect('knowledge_base.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, name TEXT, content TEXT)''')
    conn.commit()
    return conn

# Función para verificar si un documento ya existe en la base de datos
def document_exists(conn, name):
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM documents WHERE name = ?", (name,))
    return c.fetchone()[0] > 0

# Función para guardar documento en la base de datos
def save_document(conn, name, content):
    c = conn.cursor()
    c.execute("INSERT INTO documents (name, content) VALUES (?, ?)", (name, content))
    conn.commit()

# Función para obtener todos los documentos de la base de datos
def get_all_documents(conn):
    c = conn.cursor()
    c.execute("SELECT name, content FROM documents")
    return c.fetchall()

# Función para extraer texto de los PDFs
@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            text += f"\n--- Página {i+1} del documento {pdf.name} ---\n"
            text += page.extract_text()
    return text

# Divide el texto en chunks más pequeños
@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Crea y guarda el almacén de vectores
@st.cache_resource
def get_vector_store(_embeddings, text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=_embeddings)
    return vector_store

# Configura la cadena de conversación
def get_conversational_chain():
    prompt_template = """
    Basándote en el contexto proporcionado y tu conocimiento general, responde la pregunta de manera detallada y expresiva. 
    Asegúrate de incluir todos los detalles relevantes del contexto y complementa con tu conocimiento si es necesario.
    Si la respuesta no está en el contexto, indica que no está disponible pero ofrece información relacionada si es posible.
    Al final de tu respuesta, menciona específicamente en qué documento(s) y página(s) se encuentra la información utilizada.

    Contexto:
    {context}

    Pregunta: {question}

    Respuesta detallada:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Procesa la entrada del usuario y genera una respuesta
def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain.invoke(
        {"input_documents": docs, "question": user_question}
    )

    st.write("Respuesta:", response["output_text"])

# Función principal que configura la interfaz de Streamlit
def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    
    # Estilo personalizado (mantenido igual que en la versión anterior)
    st.markdown(
        """
        <style>
         @keyframes colorChange {
            0% {
                color: #b979d9;
            }
            25% {
                color: #cb2f0e;
            }
            50% {
                color: #5be5cc;
            }
            75% {
                color: #201edc;
            }
            100% {
                color: #d7c3e1;
            }
        }
        body {
            background-color: #1e1e1e;
            color: #e8e8e8;
        }
        .css-1d391kg {
            background-color: #1e1e1e;
            color: #e8e8e8;
        }
        .css-1offfwp {
            background-color: #1e1e1e;
            color: #e8e8e8;
        }
        .stButton button {
            background-color: #3b3b3b;
            color: #e8e8e8;
            border-radius: 5px;
        }
        .stTextInput input {
            background-color: #faf9fa;
            color: #383638;
        }
        .stFileUploader label {
            background-color: #3b3b3b;
            color: #e8e8e8;
        }
        .stTextInput div {
            background-color: #faf9fa;
            color: #383638;
        }
        .css-1n543e5 {
            background-color: #1e1e1e;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <h1 style="
        text-align: center;
        animation: colorChange 5s infinite;
        -webkit-animation: colorChange 5s infinite;
        -moz-animation: colorChange 5s infinite;
        -o-animation: colorChange 5s infinite;
        color: #D5D3D6;
    ">Hola, soy Mark2, asistente inteligente con el proósito de facilitar desiciones</h1>
        <h3 style='text-align: center; color: #888888;'>¿En qué puedo ayudarte?</h3>
        """,
        unsafe_allow_html=True
    )

    # Inicializar la base de datos
    conn = init_db()

    # Obtener la lista de documentos ya procesados
    processed_docs = get_all_documents(conn)

    # Mostrar documentos procesados en una burbuja informativa
    if processed_docs:
        st.info(f"Documentos ya procesados: {', '.join([doc[0] for doc in processed_docs])}")

    # Opción para trabajar con conocimiento existente o cargar nuevo
    option = st.radio(
        "Elige una opción:",
        ("Trabajar con conocimiento existente", "Cargar nuevo conocimiento")
    )

    embeddings = get_embeddings()

    if option == "Trabajar con conocimiento existente":
        if processed_docs:
            st.success("Listo para responder preguntas sobre el conocimiento existente.")
            user_question = st.text_input("Dame una tarea en base al conocimiento cargado:")
            if user_question:
                try:
                    # Reconstruir el vector store desde la base de datos
                    all_text = "\n".join([doc[1] for doc in processed_docs])
                    text_chunks = get_text_chunks(all_text)
                    vector_store = get_vector_store(embeddings, text_chunks)
                    user_input(user_question, vector_store)
                except Exception as e:
                    st.error(f"Error al procesar la pregunta: {str(e)}")
        else:
            st.warning("No hay documentos procesados. Por favor, carga nuevo conocimiento primero.")

    else:  # Cargar nuevo conocimiento
        st.subheader("Cargar nuevo conocimiento")
        pdf_docs = st.file_uploader("Sube tus archivos PDF", accept_multiple_files=True)
        if st.button("Procesar PDFs"):
            if pdf_docs:
                with st.spinner("Procesando PDFs... Por favor, espera."):
                    for pdf in pdf_docs:
                        if document_exists(conn, pdf.name):
                            st.warning(f"El documento {pdf.name} ya ha sido procesado anteriormente.")
                        else:
                            raw_text = get_pdf_text([pdf])
                            text_chunks = get_text_chunks(raw_text)
                            vector_store = get_vector_store(embeddings, text_chunks)
                            
                            # Guardar en la base de datos
                            save_document(conn, pdf.name, raw_text)
                    
                    st.success("¡PDFs procesados con éxito! Ahora puedes hacer preguntas.")
            else:
                st.warning("Por favor, sube al menos un archivo PDF antes de procesar.")


    # Cerrar la conexión a la base de datos
    conn.close()

if __name__ == "__main__":
    main()