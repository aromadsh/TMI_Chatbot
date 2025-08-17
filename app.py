import streamlit as st
import requests
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

@st.cache_resource
def get_vectorstore():
    embedding = SentenceTransformerEmbeddings(model_name="BM-K/KoSimCSE-roberta-multitask")
    return Chroma(persist_directory="./db", embedding_function=embedding)

data_embedding = get_vectorstore()

def ask_ollama_with_context(question, database):

    db = database
    
    docs = db.similarity_search(question, k=2)
    context = "\n".join([doc.page_content for doc in docs])

    full_prompt = f"""
    다음은 한 개인에 대한 정보입니다:    
    {context}
    ------------------------
    
    사용자의 질문: {question}
    위 정보를 바탕으로 한 개인으로서 자신을 이야기를 하듯이 친절하고 정확하게 답변을 해주세요.
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "hf.co/Bllossom/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M",
            "prompt": full_prompt,
            "stream": False,
        }
    )
    return response.json()["response"]

st.title("TMI")

user_input = st.text_input("무엇이 궁금  한가요?")

if user_input:
    with st.spinner("답변 생성 중..."):
        answer = ask_ollama_with_context(user_input, data_embedding)
        st.markdown(f"{answer}")