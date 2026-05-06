"""
데이터베이스 초기화 스크립트 (init_db.py) - 단독 실행 가능 버전

사용법:
    python init_db.py          # 레시피 전체 삽입 (비어있을 경우)
    python init_db.py --reset  # 기존 DB 초기화 후 재삽입
    python init_db.py --stats  # DB 통계만 출력
"""

import sys
import time
import argparse
import json
import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# 환경변수 로드 (.env 파일에서 OPENAI_API_KEY 가져오기)
load_dotenv()

# app.py와 동일한 DB 설정 공유
DB_PATH = "./chroma_db"
COLLECTION_NAME = "recipe_collection_v2"

def load_recipes(file_path="recipes.json"):
    """JSON 파일에서 레시피 데이터를 로드합니다."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ {file_path} 파일을 찾을 수 없습니다. 파일 위치를 확인해주세요.")
        sys.exit(1)

class RecipeVectorDB:
    """ChromaDB를 직접 제어하는 래퍼 클래스"""
    def __init__(self, embedding_model="text-embedding-3-small"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            sys.exit(1)

        # OpenAI 임베딩 함수 초기화
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embedding_model
        )
        
        # ChromaDB 클라이언트 및 컬렉션 연동
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )

    def reset_database(self):
        """기존 컬렉션을 삭제하고 새로 생성합니다."""
        try:
            self.client.delete_collection(name=COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            print("✅ 기존 컬렉션 삭제 및 초기화 완료")
        except ValueError:
            print("⚠️ 삭제할 컬렉션이 없습니다. 새로 생성합니다.")

    def count(self):
        """현재 DB에 저장된 문서 개수를 반환합니다."""
        return self.collection.count()

    def create_batch(self, recipes, batch_size=50):
        """레시피 데이터를 배치 단위로 DB에 삽입합니다."""
        total_inserted = 0
        
        for i in range(0, len(recipes), batch_size):
            batch = recipes[i:i + batch_size]
            
            ids = []
            documents = []
            metadatas = []
            
            for recipe in batch:
                ids.append(recipe["id"])
                # app.py와 동일한 검색 최적화 문서 구조 구성
                doc = f"{recipe['name']} - {recipe['description']} 재료: {recipe['ingredients']} 조리법: {recipe['instructions']}"
                documents.append(doc)
                
                # UI용 메타데이터 병합
                meta = recipe["metadata"].copy()
                meta["name"] = recipe["name"]
                meta["description"] = recipe["description"]
                meta["ingredients"] = recipe["ingredients"]
                meta["instructions"] = recipe["instructions"]
                metadatas.append(meta)
                
            # DB 삽입 실행
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            total_inserted += len(batch)
            print(f"   ... {total_inserted}/{len(recipes)} 개 삽입 완료")
            
        return total_inserted

    def get_collection_stats(self):
        """DB 통계(카테고리별 개수 등)를 반환합니다."""
        all_data = self.collection.get(include=['metadatas'])
        categories = {}
        
        if all_data and 'metadatas' in all_data and all_data['metadatas']:
            for meta in all_data['metadatas']:
                cat = meta.get('category', '알 수 없음')
                categories[cat] = categories.get(cat, 0) + 1
                
        return {
            "main": self.count(),
            "categories": categories
        }
        
    def list_all_collections(self):
        """생성된 모든 컬렉션 목록을 반환합니다."""
        return [col.name for col in self.client.list_collections()]


def main():
    parser = argparse.ArgumentParser(description="레시피 벡터 DB 초기화")
    parser.add_argument("--reset", action="store_true", help="기존 데이터 삭제 후 재삽입")
    parser.add_argument("--stats", action="store_true", help="DB 통계만 출력")
    parser.add_argument("--model", default="text-embedding-3-small",
                        help="임베딩 모델 (기본: text-embedding-3-small)")
    args = parser.parse_args()

    print("=" * 60)
    print("🍳 레시피 벡터 데이터베이스 관리자")
    print("=" * 60)

    # DB 및 임베딩 설정 초기화
    db = RecipeVectorDB(embedding_model=args.model)

    if args.stats:
        print_stats(db)
        return

    if args.reset:
        print("\n⚠️  기존 데이터를 삭제하고 재삽입합니다...")
        db.reset_database()

    # 레시피 로드 및 수량 비교
    recipes = load_recipes()
    total_recipes = len(recipes)
    current_count = db.count()
    
    print(f"\n현재 DB 항목 수: {current_count}")
    print(f"삽입할 레시피 수: {total_recipes}")

    if current_count >= total_recipes and not args.reset:
        print("✅ 이미 모든 레시피가 DB에 저장되어 있습니다.")
        print("   강제로 덮어쓰려면 '--reset' 옵션을 추가하여 실행하세요.")
        print_stats(db)
        return

    print("\n📚 레시피 데이터 로드 중...")
    print(f"   {total_recipes}개 레시피 준비 완료")

    print(f"\n🔢 벡터 DB에 삽입 중 (모델: {args.model})...")
    start_time = time.time()

    try:
        inserted = db.create_batch(recipes, batch_size=50)
        elapsed = time.time() - start_time

        print(f"\n✅ 삽입 완료!")
        print(f"   총 삽입 수: {inserted}개")
        print(f"   소요 시간: {elapsed:.1f}초 ({elapsed/inserted*1000:.1f}ms/레시피)")

    except Exception as e:
        print(f"\n❌ 삽입 실패: {e}")
        raise

    # 최종 통계 출력
    print_stats(db)

    print("\n🎉 초기화 완료! 이제 앱을 실행할 수 있습니다.")
    print("▶ 실행 명령어: streamlit run app.py")


def print_stats(db: RecipeVectorDB):
    """DB 통계 출력"""
    print("\n" + "=" * 40)
    print("📊 데이터베이스 통계")
    print("=" * 40)

    stats = db.get_collection_stats()
    print(f"{'메인 컬렉션':<20} {stats['main']:>6}개")
    print("-" * 40)
    
    if stats['categories']:
        for cat, count in stats["categories"].items():
            print(f"  {cat:<18} {count:>6}개")
    else:
        print("  데이터가 없습니다.")
        
    print("-" * 40)
    total_in_cats = sum(stats["categories"].values())
    print(f"{'카테고리 합계':<20} {total_in_cats:>6}개")
    print("=" * 40)

    print("\n📋 컬렉션 목록:")
    for col_name in db.list_all_collections():
        print(f"  - {col_name}")


if __name__ == "__main__":
    main()