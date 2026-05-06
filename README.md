# 🍲 기억의 도서관 짓기: 스마트 하이브리드 레시피 추천 시스템

## 1. 프로젝트 소개 및 아키텍처
본 프로젝트는 ChromaDB 벡터 데이터베이스를 활용하여 구축된 지능형 레시피 검색 및 추천 시스템입니다. 기존의 단순 키워드 매칭(파일 기반 탐색)의 한계를 극복하기 위해, 사용자의 자연어 질의(문맥)를 이해하는 **순수 시맨틱 검색**과, 조리시간, 맵기, 카테고리 등의 조건을 결합한 **하이브리드 검색(Hybrid Search)**을 동시에 제공하고 비교합니다.

### 🏗 시스템 아키텍처 및 데이터 흐름
본 시스템은 다음과 같은 5단계의 흐름으로 작동합니다.
1. **데이터 로딩 (Data Loading):** `recipes.json` 파일에서 240개의 다국적 레시피와 메타데이터를 로드합니다[cite: 8, 10].
2. **임베딩 (Embedding):** 커스텀 임베딩 함수인 OpenAI의 `text-embedding-3-large` 모델을 호출하여, 레시피의 `이름`, `설명`, `재료`, `조리법`을 병합한 텍스트를 고차원 벡터로 변환합니다.
3. **DB 저장 (DB Storage):** ChromaDB의 `PersistentClient`를 활용하여 생성된 임베딩 벡터와 메타데이터를 로컬 디스크(`.chroma_db/`)에 영구적으로 안전하게 저장합니다.
4. **쿼리 (Querying):** Streamlit UI에서 입력받은 자연어 검색어와 메타데이터 필터 조건(`$lte`, `$gte`, `$eq`, `$and`)을 조합하여 ChromaDB에 쿼리를 전송합니다.
5. **결과 후처리 (Post-processing):** 반환된 결과(유사도 거리 점수, 메타데이터 등)를 가공하여 순수 시맨틱 검색과 하이브리드 검색의 차이를 한눈에 비교할 수 있도록 UI 카드로 렌더링합니다.

---

## 2. 선택한 벡터 DB와 선택 이유
* **선택한 벡터 DB:** **ChromaDB**[cite: 8]
* **선택 이유:**
  1. **영속성(Persistent Storage) 확보:** 클라우드 서버에 의존하지 않고 로컬 스토리지에 데이터를 SQLite/Parquet 형태로 영구 저장하여, 재실행 시 임베딩 비용과 시간을 절약할 수 있습니다.
  2. **강력한 필터링 문법:** 쿼리 실행 시 딕셔너리 형태의 `where` 절을 통해 `$and`, `$eq`, `$gte`, `$lte` 등 선언적인 메타데이터 사전 필터링(Pre-filtering)을 완벽하게 지원합니다[cite: 9].
  3. **Python 생태계 최적화:** 외부 REST API 통신 없이 Python 프로세스 내에서 가볍게 동작하므로, Streamlit 기반의 대시보드 및 데이터 파이프라인과 유기적으로 결합하기 가장 적합합니다[cite: 9].

---

## 3. 권장 모듈 구조 (디렉토리 트리)
유지보수 및 확장을 고려하여 다음과 같이 모듈을 분리하여 구성했습니다.
```text
📦 Recipe_Recommendation_Project
 ┣ 📂 data
 ┃ ┗ 📜 recipes_2.json            # 240개의 레시피 데이터셋 및 메타데이터
 ┣ 📂 db
 ┃ ┗ 📜 init_db_2.py              # 데이터 로딩 및 DB 초기화 배치 스크립트[cite: 8]
 ┣ 📂 ui
 ┃ ┗ 📜 app.py                    # Streamlit 기반 검색, CRUD, 시각화, 벤치마크 통합 UI[cite: 9]
 ┣ 📜 .env                        # OpenAI API 키 저장 (보안)
 ┣ 📜 .gitignore                  # 보안 및 대용량 파일 업로드 방지
 ┗ 📜 README.md                   # 프로젝트 명세서
```

## 4. 실행 방법

### Step 1: 필수 의존성 라이브러리 설치
터미널에서 아래 명령어를 실행하여 필수 패키지를 설치합니다.
```bash
pip install streamlit chromadb python-dotenv pandas numpy scikit-learn plotly openai
