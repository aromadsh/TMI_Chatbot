# from langchain.vectorstores import Chroma
# from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter

# # 1. 문서 로드 및 분할
# loader = TextLoader("intro.txt", encoding='utf-8')
# documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
# docs = text_splitter.split_documents(documents)

# # 2. 임베딩 모델 로드
# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# # 3. 벡터 DB 생성 및 저장
# db = Chroma.from_documents(docs, embedding, persist_directory="./db")
# db.persist()

# print("✅ intro.txt 임베딩 완료 및 db 저장 완료.")

from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# 1. 문서 로드 및 분할
loader = TextLoader("intro.txt", encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# 2. KoSimCSE 임베딩 모델 로드
# BM-K/KoSimCSE-roberta-multitask 모델 사용
embedding = SentenceTransformerEmbeddings(
    model_name="BM-K/KoSimCSE-roberta-multitask"
)

# 3. 벡터 DB 생성 및 저장
db = Chroma.from_documents(docs, embedding, persist_directory="./db")
db.persist()

print("✅ intro.txt 임베딩 완료 (KoSimCSE) 및 db 저장 완료.")
