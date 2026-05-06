import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import json
import os
import time
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

# --- A. 환경 설정 및 커스텀 임베딩 (OpenAI) ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.error("OpenAI API Key가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    st.stop()

custom_embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key,
    model_name="text-embedding-3-large"
)

# ChromaDB Persistent Client 초기화
client = chromadb.PersistentClient(path="./chroma_db")
COLLECTION_NAME = "recipe_collection_v3"  # v3로 변경하여 기존 오염된 DB를 버리고 새로 생성

def get_or_create_collection():
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=custom_embedding_function,
        metadata={"hnsw:space": "cosine"} # 기본값(l2) 대신 코사인 유사도 거리로 명확히 지정
    )

collection = get_or_create_collection()

# --- 데이터 초기화 ---
def initialize_data():
    with open("recipes.json", "r", encoding="utf-8") as f:
        recipes = json.load(f)

    # ✅ ID 기반 중복 체크: DB에 이미 있는 ID는 건너뜀
    all_ids = [r["id"] for r in recipes]
    existing = collection.get(ids=all_ids, include=[])
    existing_ids = set(existing["ids"])
    
    new_recipes = [r for r in recipes if r["id"] not in existing_ids]

    if not new_recipes:
        return  # 모두 이미 존재 → 임베딩 API 호출 없음

    with st.spinner(f"최초 실행: {len(new_recipes)}개 레시피를 임베딩 중입니다... (약 1~2분 소요)"):
        ids, documents, metadatas = [], [], []
        for recipe in new_recipes:
            ids.append(recipe["id"])
            doc = f"{recipe['name']} - {recipe['description']} 재료: {recipe['ingredients']} 조리법: {recipe['instructions']}"
            documents.append(doc)
            meta = recipe["metadata"].copy()
            meta["name"] = recipe["name"]
            meta["description"] = recipe["description"]
            meta["ingredients"] = recipe["ingredients"]
            meta["instructions"] = recipe["instructions"]
            metadatas.append(meta)

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    st.success(f"임베딩 완료! ({len(new_recipes)}개 신규 추가)")

initialize_data()

# --- Streamlit Session State ---
if "search_query" not in st.session_state:
    st.session_state.search_query = ""

def set_rainy_day_query():
    st.session_state.search_query = "비오는 날에 어울리는 얼큰한 국물 요리"

def render_recipe_card(meta, dist, index):
    with st.container():
        st.markdown(f"#### {index+1}. {meta['name']}")
        st.caption(f"**유사도(거리):** {dist:.4f} | **카테고리:** {meta['category']} | **조리시간:** {meta['cook_time']}분 | **난이도:** {meta['difficulty']} | **맵기:** 🌶️{meta['spicy_level']}")
        st.write(f"**📝 설명:** {meta['description']}")
        with st.expander("재료 및 조리법 보기"):
            st.write(f"**🛒 재료:** {meta['ingredients']}")
            st.info(f"**👨‍🍳 조리법:**\n{meta['instructions']}")
        st.markdown("---")

# --- UI 레이아웃 설정 ---
st.set_page_config(page_title="스마트 레시피 추천기", layout="wide")
st.title("🍲 스마트 하이브리드 레시피 추천기 (ChromaDB)")

tab1, tab2, tab3, tab4 = st.tabs(["🔍 검색 결과 비교", "⚙️ CRUD 운영 테스트", "📊 벡터 데이터 시각화", "⏱️ 파일 vs 벡터 DB 벤치마크 (요구사항 D)"])

# =========================================================
# 탭 1: 하이브리드 검색 및 레시피 추천
# =========================================================
with tab1:
    st.header("조건 및 상황에 맞는 레시피 추천")
    st.button("🌧️ '비오는 날에 어울리는 얼큰한 국물 요리' 추천받기", on_click=set_rainy_day_query)
    st.markdown("---")
    
    col_filter, col_search = st.columns([1, 3])
    with col_filter:
        st.subheader("메타데이터 필터")
        max_time = st.slider("최대 조리 시간 (분)", 5, 240, 60, step=5)
        difficulty = st.selectbox("난이도", ["전체", "쉬움", "보통", "어려움"], index=0)
        category = st.selectbox("카테고리", ["전체", "한식", "중식", "일식", "양식", "분식", "디저트"], index=0)
        min_spicy = st.slider("최소 맵기 단계 (0~5)", 0, 5, 0)

    with col_search:
        query_text = st.text_input("자연어 검색어 (상황, 재료, 먹고 싶은 맛 등)", key="search_query")
        execute_search = st.button("검색 실행")
        
    if execute_search or query_text:
        with st.spinner("최적의 레시피를 찾고 있습니다..."):
            conditions = []
            conditions.append({"cook_time": {"$lte": max_time}})
            conditions.append({"spicy_level": {"$gte": min_spicy}})
            if difficulty != "전체": conditions.append({"difficulty": {"$eq": difficulty}})
            if category != "전체": conditions.append({"category": {"$eq": category}})
                
            where_clause = None
            if len(conditions) > 1: where_clause = {"$and": conditions}
            elif len(conditions) == 1: where_clause = conditions[0]
            
            # A. 순수 시맨틱 검색
            semantic_results = collection.query(query_texts=[query_text], n_results=3)
            # B. 하이브리드 검색
            try:
                hybrid_results = collection.query(query_texts=[query_text], n_results=3, where=where_clause)
            except Exception:
                hybrid_results = {"ids": [[]], "metadatas": [[]], "distances": [[]]}
            
            st.markdown("---")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("🧠 순수 시맨틱 검색")
                if semantic_results['ids'][0]:
                    for i in range(len(semantic_results['ids'][0])):
                        render_recipe_card(semantic_results['metadatas'][0][i], semantic_results['distances'][0][i], i)
                else: st.warning("결과 없음")
                    
            with res_col2:
                st.subheader("🎛️ 하이브리드 검색 (필터 적용)")
                if hybrid_results['ids'][0]:
                    for i in range(len(hybrid_results['ids'][0])):
                        render_recipe_card(hybrid_results['metadatas'][0][i], hybrid_results['distances'][0][i], i)
                else: st.warning("필터 조건을 만족하는 결과가 없습니다.")

# =========================================================
# 탭 2 & 3: CRUD 및 시각화 (기존 코드 유지)
# =========================================================
with tab2:
    st.header("벡터 DB CRUD 운영 테스트")
    crud_id = st.text_input("테스트 대상 ID", "recipe_999")
    crud_doc = st.text_area("문서 내용", "테스트용 문서")
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        if st.button("Read"):
            res = collection.get(ids=[crud_id])
            if res['ids']:
                st.success(res['documents'][0])
            else:
                st.warning("없음")
                
    with c2:
        if st.button("Update"):
            try: 
                collection.update(ids=[crud_id], documents=[crud_doc], metadatas=[{"category":"테스트"}])
                st.success("업데이트 성공")
            except Exception as e: 
                st.error(f"실패: {e}")
                
    with c3:
        if st.button("Upsert"):
            collection.upsert(ids=[crud_id], documents=[crud_doc], metadatas=[{"category":"테스트", "cook_time":10, "spicy_level":0}])
            st.success("Upsert 성공")
            
    with c4:
        if st.button("Delete"):
            try: 
                collection.delete(ids=[crud_id])
                st.success("삭제 완료")
            except Exception: 
                st.warning("삭제할 데이터가 없습니다.")

with tab3:
    st.header("2D 시각화 (t-SNE)")
    if st.button("시각화 생성"):
        with st.spinner("생성 중..."):
            all_data = collection.get(include=['embeddings', 'metadatas'])
            emb = np.array(all_data['embeddings'])
            cat = [m['category'] for m in all_data['metadatas']]
            names = [m['name'] for m in all_data['metadatas']]
            reduced = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(emb)
            df_tsne = pd.DataFrame({'X': reduced[:,0], 'Y': reduced[:,1], 'Category': cat, 'Name': names})
            st.plotly_chart(px.scatter(df_tsne, x='X', y='Y', color='Category', hover_name='Name'))

# =========================================================
# 탭 4: ⏱️ 파일 vs 벡터 DB 비교 및 벤치마크 (핵심 요구사항 D)
# =========================================================
with tab4:
    st.header("1. 실시간 검색 속도 벤치마크 (데이터 규모별 비교)")
    st.info("""
    이전 과제에서 사용한 Numpy 기반 방식과 ChromaDB의 검색+필터링 속도를 
    동일한 데이터셋을 배수로 확장(100개 -> 5,000개)시켜가며 실시간으로 측정합니다.
    """)
    
    if st.button("▶ 데이터 규모별 벤치마크 실행 (약 10초 소요)"):
        with st.spinner("기존 데이터를 복제하여 시뮬레이션을 진행 중입니다..."):
            # 기존 임베딩 가져오기
            base_data = collection.get(include=["embeddings", "metadatas"])
            
            # ⚠️ Numpy 배열일 경우를 대비해 명시적으로 파이썬 리스트로 변환합니다.
            base_emb = list(base_data["embeddings"])
            base_meta = list(base_data["metadatas"])
            query_emb = base_emb[0]  # 테스트용 쿼리 벡터 (김치찌개)
            
            sizes = [100, 500, 1000, 5000]
            benchmark_results = []
            
            # 독립된 메모리 DB로 Chroma 테스트 (본 DB 오염 방지)
            temp_client = chromadb.EphemeralClient()
            
            for size in sizes:
                # 데이터 배수 확장 시뮬레이션
                mult = (size // len(base_emb)) + 1
                test_embs = (base_emb * mult)[:size]
                test_metas = (base_meta * mult)[:size]
                test_ids = [f"id_{i}" for i in range(size)]
                
                # --- [과제1 방식] Numpy + For Loop 하이브리드 검색 시뮬레이션 ---
                start_np = time.time()
                # 1. 메타데이터 필터링 (Python for loop)
                filtered_indices = [i for i, m in enumerate(test_metas) if m.get("category") == "한식"]
                # 2. 코사인 유사도 계산 (Numpy)
                if filtered_indices:
                    filtered_embs = np.array([test_embs[i] for i in filtered_indices])
                    q_vec = np.array(query_emb)
                    norms = np.linalg.norm(filtered_embs, axis=1) * np.linalg.norm(q_vec)
                    norms[norms == 0] = 1e-10
                    sims = np.dot(filtered_embs, q_vec) / norms
                    top_k = np.argsort(sims)[::-1][:5]
                end_np = time.time()
                np_time = (end_np - start_np) * 1000
                
                # --- [과제2 방식] ChromaDB 검색 ---
                try:
                    temp_client.delete_collection(f"bench_{size}")
                except Exception:
                    pass
                    
                temp_col = temp_client.create_collection(f"bench_{size}")
                temp_col.add(ids=test_ids, embeddings=test_embs, metadatas=test_metas)
                
                # HNSW 인덱스 빌드용 웜업
                temp_col.query(query_embeddings=[query_emb], n_results=1)
                
                start_ch = time.time()
                temp_col.query(
                    query_embeddings=[query_emb],
                    n_results=5,
                    where={"category": "한식"}
                )
                end_ch = time.time()
                ch_time = (end_ch - start_ch) * 1000
                
                benchmark_results.append({
                    "Data Size": size,
                    "Numpy (과제1)": np_time,
                    "ChromaDB (과제2)": ch_time
                })
            
            df_bench = pd.DataFrame(benchmark_results).set_index("Data Size")
            st.success("벤치마크 완료!")
            
            # 차트 렌더링
            st.line_chart(df_bench, use_container_width=True)
            st.dataframe(df_bench.style.format("{:.2f} ms"))
            st.caption("* Numpy는 메모리에서 수식 연산만 하여 소규모에서는 매우 빠르지만, 데이터가 수십만 건을 넘어가면 HNSW 인덱스를 사용하는 벡터DB에 역전당합니다. 벡터DB는 네트워크/파싱 오버헤드로 인해 소규모에선 시간 고정값이 발생합니다.")

    st.markdown("---")
    st.header("2. 기능 및 구현 난이도 비교 리포트")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📝 과제1: 파일 기반 (Pickle + Numpy)")
        st.markdown("""
        **1. 메타데이터 필터링 (불편함):**
        * 파이썬의 `if target_time != "무관": sim -= 0.3` 처럼 유사도 점수에 페널티를 주거나 직접 For문을 돌며 데이터를 걸러내야 했습니다.
        * 조건이 추가될 때마다 코드가 기하급수적으로 길어집니다.
        
        **2. 데이터 추가/수정의 복잡도 (높음):**
        * 항목 1개를 수정하더라도 DataFrame을 갱신하고, 임베딩을 다시 추출하여 전체 `.pkl` 파일을 `pickle.dump`로 덮어써야 했습니다.
        
        **3. 영속성 (불안정):**
        * 앱을 켤 때 파일 존재 여부를 `os.path.exists`로 확인 후 전체를 RAM에 올려야 함.
        """)
        
    with col_b:
        st.subheader("💡 과제2: 벡터 DB (ChromaDB)")
        st.markdown("""
        **1. 메타데이터 필터링 (매우 간편함):**
        * 데이터베이스 엔진 단에서 `where={"$and": [{"category": "한식"}]}` 쿼리 객체 하나로 완벽한 Pre-filtering을 지원합니다. 코드 복잡도가 극도로 낮아졌습니다.
        
        **2. 데이터 추가/수정의 복잡도 (낮음):**
        * 단 한 줄의 `collection.upsert(ids=[...])` 호출로 부분 업데이트가 완벽히 보장됩니다(멱등성).
        
        **3. 영속성 (안정적):**
        * `PersistentClient`를 통해 데이터를 삽입하는 즉시 SQLite 및 Parquet 형태로 디스크에 자동 영구 보존됩니다.
        """)
