# 🍲 기억의 도서관 짓기: 스마트 하이브리드 레시피 추천 시스템

## 1. 프로젝트 소개 및 아키텍처

본 프로젝트는 ChromaDB 벡터 데이터베이스를 활용한 지능형 레시피 검색·추천 시스템입니다.
기존 파일 기반(NumPy + Pickle) 방식의 세 가지 한계를 해결합니다.

- **검색 비효율** — 전체 벡터를 매번 순회하는 O(n) 코사인 유사도 → HNSW 기반 ANN O(log n)으로 대체
- **필터링 불편** — Python for 루프 수동 필터링 → `where` 절 선언적 메타데이터 Pre-filtering
- **영속성 불안정** — Pickle 파일 수동 관리 + 중복 임베딩 낭비 → PersistentClient 자동 저장 + ID 기반 중복 체크

### 🏗 시스템 아키텍처 및 데이터 흐름

시스템은 **최초 1회 실행하는 데이터 파이프라인**과 **매 요청마다 실행되는 실시간 검색 흐름**으로 구분됩니다.
┌─────────────────────────── 데이터 파이프라인 (최초 1회) ───────────────────────────┐
│                                                                                    │
│  ① 데이터 로딩  →  ② 임베딩 생성  →  ③ DB 저장                                  │
│  recipes.json      text-embedding       PersistentClient                           │
│  (240개 레시피)    -3-large             → ./chroma_db/                             │
│                    (3,072차원)                                                     │
└────────────────────────────────────────────────────────────────────────────────────┘
↓ (저장 완료 후)
┌─────────────────────────── 실시간 검색 흐름 ───────────────────────────────────────┐
│                                                                                    │
│  ④ Streamlit UI  →  ChromaDB (ANN 검색)  →  ⑤ 결과 렌더링                       │
│  자연어 쿼리         where 절 Pre-filtering     시맨틱 vs 하이브리드 비교 카드     │
│  + 슬라이더 필터     ($lte · $gte · $eq · $and)                                   │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
#### 단계별 설명

| 단계 | 모듈 | 역할 |
|------|------|------|
| ① 데이터 로딩 | `init_db.py` / `app.py` | `recipes.json`에서 240개 레시피 파싱. `id`, `name`, `description`, `ingredients`, `instructions`, `metadata` 추출 |
| ② 임베딩 생성 | `OpenAIEmbeddingFunction` | 4개 필드를 하나의 문서 문자열로 병합 후 `text-embedding-3-large` 호출 → 3,072차원 벡터 변환 |
| ③ DB 저장 | `chromadb.PersistentClient` | 임베딩 벡터·메타데이터·원본 문서를 `./chroma_db/`에 SQLite/Parquet 형태로 영구 저장. ID 기반 중복 체크로 API 재호출 방지 |
| ④ 쿼리 실행 | Streamlit UI + ChromaDB | 자연어 입력과 슬라이더·드롭다운 필터를 `$and` 조건으로 결합하여 ChromaDB HNSW 인덱스에 쿼리 |
| ⑤ 결과 렌더링 | Streamlit UI | 코사인 거리 점수·메타데이터를 포함한 결과 카드. 순수 시맨틱 검색과 하이브리드 검색 결과를 나란히 비교 |

#### 핵심 설계 결정

**임베딩 입력 텍스트 구성**
```python
doc = f"{name} - {description} 재료: {ingredients} 조리법: {instructions}"
```
단일 벡터로 레시피의 의미 전체를 포착하기 위해 4개 필드를 병합합니다.

**ID 기반 중복 체크 (API 비용 절감)**
```python
existing = collection.get(ids=all_ids, include=[])
existing_ids = set(existing["ids"])
new_recipes = [r for r in recipes if r["id"] not in existing_ids]
# → 새 항목만 임베딩 API 호출, 기존 항목 재사용
```

**하이브리드 검색 where 절 동적 조합**
```python
conditions = [
    {"cook_time":    {"$lte": max_time}},   # 조리시간 이하
    {"spicy_level":  {"$gte": min_spicy}},  # 맵기 이상
]
if difficulty != "전체":
    conditions.append({"difficulty": {"$eq": difficulty}})
if category != "전체":
    conditions.append({"category": {"$eq": category}})

where_clause = {"$and": conditions} if len(conditions) > 1 else conditions[0]

results = collection.query(
    query_texts=[query],
    n_results=3,
    where=where_clause   # 필터 통과 항목 내에서만 ANN 검색 실행
)
```
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
```

### Step 2: 환경 변수 설정(API 키 보호)
보안을 위해 API 키는 코드에 하드코딩하지 않습니다. 프로젝트 루트 폴더에 .env 파일을 생성하고 아래와 같이 OpenAI API 키를 입력하세요.
```bash
OPENAI_API_KEY=sk-본인의_실제_API_키를_입력하세요
```

### Step 3: 데이터베이스 초기화 및 적재
터미널에서 아래 명령어를 통해 JSON 데이터를 읽어 임베딩하고 DB를 구축합니다. (약 1~2분 소요)
```bash
python db/init_db_2.py

# 💡 팁: 기존 DB를 날리고 강제로 덮어쓰려면 `--reset` 옵션을 추가하세요.
# python db/init_db_2.py --reset
```

### Step 4: 웹 애플리케이션 실행
데이터 적재가 완료되면 아래 명령어로 Streamlit UI를 실행합니다.
```bash
streamlit run ui/app.py
```

## 5. 데이터셋 출처 및 메타데이터 스키마 설명

### 데이터셋 출처: AI도구를 활용해 만든 240개의 다국적 레시피 JSON 데이터셋
### 메타데이터 스키마: 필터링과 하이브리드 검색 고도화를 위해 아래 6가지의 의미 있는 메타데이터 필드를 포함합니다.
1. category (문자열): 요리 카테고리 (한식, 일식, 중식, 양식, 분식, 디저트 등)
2. cook_time (정수): 예상 조리 소요 시간 (분 단위)
3. difficulty (문자열): 조리 난이도 (쉬움, 보통, 어려움)
4. spicy_level (정수): 매운맛 강도 (0 ~ 5 단계)
5. calories (정수): 1인분 기준 예상 칼로리 (kcal)
6. rating (실수): 레시피 예상 평점 (0.0 ~ 5.0)

## 6. 주요 확장 기능 및 구현 성과
### 핵심 필수 요구사항 외에 아래와 같은 4가지 확장 기능을 시스템에 유기적으로 통합했습니다.
1. 커스텀 임베딩 함수: ChromaDB의 기본 모델 대신 OpenAIEmbeddingFunction을 활용하여 text-embedding-3-large 모델을 컬렉션에 적용, 검색의 의미론적 정확도를 극대화했습니다.
2. 고급 UI 대시보드: Streamlit을 활용하여 슬라이더, 드롭다운 필터, 확장/축소(Expander)형 결과 카드를 갖춘 직관적인 UI를 개발했습니다.
3. 벡터 2D 축소 시각화: sklearn.manifold.TSNE와 plotly.express를 결합하여 고차원 임베딩을 2D 평면에 뿌리고, 카테고리별로 색상을 매핑하여 데이터 간의 의미적 군집을 시각적으로 분석할 수 있게 하였습니다.
4. 실시간 검색 속도 벤치마크: 파일 기반(Numpy 선형 탐색)과 벡터 DB(HNSW 인덱싱)의 성능 차이를 증명하기 위해, 메모리 DB 상에서 데이터를 100개에서 5,000개까지 증식시키며 쿼리 응답 시간(ms)을 동적으로 측정 및 그래프화하는 기능을 구현했습니다.  
