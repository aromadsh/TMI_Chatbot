from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


loader = TextLoader("intro.txt", encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embedding = SentenceTransformerEmbeddings(
    model_name="BM-K/KoSimCSE-roberta-multitask"
)

db = Chroma.from_documents(docs, embedding, persist_directory="./db")
db.persist()

print("임베딩 완료 및 db 저장 완료.")
