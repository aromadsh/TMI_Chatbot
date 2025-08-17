# 나를 소개하는 TMI Chatbot

나를 소개해주는 Chatbot, TMI  
> 나의 이야기를 담아두면, TMI가 나를 대신해서 소개해줍니다!

---

## 🚀 실행 방법

```bash
# 1. intro.txt 파일 내에 자신의 이야기를 작성합니다 (많을수록 좋아요!)

# 2. 필요 라이브러리 설치
pip install -r requirements.txt

# 3. 임베딩 DB 생성 (최초 1회)
python embed_build.py

# 4. 모델 실행
ollama run gemma3

# 5. 웹 실행
streamlit run app.py
