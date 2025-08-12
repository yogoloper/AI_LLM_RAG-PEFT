"""
간소화된 RAG 시스템
"""

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
from datasets import load_dataset
from typing import List, Dict, Tuple
from api_client import VLLMAPIClient

class SimpleRAGSystem:
    """간소화된 RAG 시스템"""
    
    def __init__(self, api_client: VLLMAPIClient):
        self.api_client = api_client
        self.chunks = []
        self.chunk_metadata = []
        self.vector_index = None
        self.embedder = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.initialized = False
    
    @st.cache_resource
    def load_embedding_model(_self):
        """임베딩 모델 로드 (캐시)"""
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def initialize(self):
        """시스템 초기화"""
        if self.initialized:
            return
        
        try:
            # 대안 데이터셋 사용 (더 적은 데이터로 테스트)
            print("📦 대안 데이터셋 사용...")
            knowledge_data = [
                {
                    'index': 0,
                    'user_message': 'How do I reset my password and update my profile?',
                    'context': 'To reset your password, go to the login page and click "Forgot Password". Enter your email to receive reset instructions. To update your profile, log into your account and navigate to Settings or Profile section where you can edit your information.',
                    'response': 'You can reset your password by clicking "Forgot Password" on the login page and following the email instructions. To update your profile, access the Settings section after logging in.'
                },
                {
                    'index': 1,
                    'user_message': 'What is Argilla and how does it work?',
                    'context': 'Argilla is a comprehensive Software as a Service (SaaS) solution for data labeling and curation. The service is designed to meet the needs of businesses seeking a reliable, secure, and user-friendly platform for data management. Argilla provides tools for annotation, quality control, and data workflow management.',
                    'response': 'Argilla is a SaaS platform for data labeling and curation that helps businesses manage their data annotation workflows efficiently.'
                },
                {
                    'index': 2,
                    'user_message': 'How can I contact customer support?',
                    'context': 'Customer support can be reached through multiple channels: email support is available 24/7, live chat during business hours (9 AM - 6 PM), and phone support for urgent issues. You can also submit tickets through the help desk system.',
                    'response': 'You can contact customer support via email (24/7), live chat (business hours), phone for urgent issues, or through our help desk ticketing system.'
                },
                {
                    'index': 3,
                    'user_message': 'What are the backup and recovery procedures?',
                    'context': 'Argilla Cloud provides comprehensive backup and recovery protocol with daily backups. The service has a recovery point objective (RPO) of 24 hours and recovery time objective (RTO) designed to minimize data loss and ensure swift recovery in case of disruption.',
                    'response': 'We perform daily backups with 24-hour RPO and have established RTO procedures to ensure quick data recovery in case of any service disruption.'
                },
                {
                    'index': 4,
                    'user_message': 'How do I manage user access and permissions?',
                    'context': 'User access management is handled through the administrator panel where you can invite team members, assign roles, and set permissions. The client administrator has full control over their teams access and can manage workspace settings efficiently.',
                    'response': 'Access the administrator panel to invite users, assign roles, and manage permissions. Client administrators have full control over team access and workspace management.'
                }
            ]
            print(f"✅ {len(knowledge_data)}개 고품질 데이터 준비됨")
            
        except Exception as e:
            print(f"❌ 데이터 준비 실패: {e}")
            raise e
        
        # 데이터 확인
        if not knowledge_data:
            raise ValueError("지식 데이터가 없습니다.")
        
        # 임베딩 모델 로드
        print("📦 임베딩 모델 로딩 중...")
        self.embedder = self.load_embedding_model()
        
        # Context 청킹
        print("📝 텍스트 청킹 중...")
        self._chunk_contexts(knowledge_data)
        
        # 데이터 확인
        if not self.chunks:
            raise ValueError("청킹 후에도 데이터가 없습니다.")
            
        print(f"📊 생성된 청크 수: {len(self.chunks)}")
        
        # 벡터 인덱스 구축
        print("🔍 벡터 인덱스 구축 중...")
        self._build_vector_index()
        
        # TF-IDF 인덱스 구축
        print("📈 TF-IDF 인덱스 구축 중...")
        self._build_tfidf_index()
        
        print("✅ RAG 시스템 초기화 완료!")
        self.initialized = True
    
    def _chunk_contexts(self, knowledge_data: List[Dict]):
        """Context를 청크로 분할"""
        chunk_size = 300
        self.chunks = []
        self.chunk_metadata = []
        
        for item in knowledge_data:
            context = item['context']
            # 간단한 청킹 (문장 단위)
            sentences = context.split('. ')
            
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk + sentence) < chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk:
                        self.chunks.append(current_chunk.strip())
                        self.chunk_metadata.append({
                            'source_index': item['index'],
                            'user_message': item['user_message'],
                            'response': item['response']
                        })
                    current_chunk = sentence + ". "
            
            # 마지막 청크 추가
            if current_chunk:
                self.chunks.append(current_chunk.strip())
                self.chunk_metadata.append({
                    'source_index': item['index'],
                    'user_message': item['user_message'],
                    'response': item['response']
                })
    
    def _build_vector_index(self):
        """벡터 인덱스 구축"""
        embeddings = self.embedder.encode(self.chunks)
        
        # FAISS 인덱스 구축
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(embeddings.astype('float32'))
    
    def _build_tfidf_index(self):
        """TF-IDF 인덱스 구축"""
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
    
    def semantic_search(self, query: str, k: int = 3) -> List[Tuple[str, Dict, float]]:
        """의미적 검색"""
        query_embedding = self.embedder.encode([query])
        distances, indices = self.vector_index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.chunks):
                similarity = 1 / (1 + distance)  # 거리를 유사도로 변환
                results.append((
                    self.chunks[idx],
                    self.chunk_metadata[idx],
                    similarity
                ))
        
        return results
    
    def keyword_search(self, query: str, k: int = 3) -> List[Tuple[str, Dict, float]]:
        """키워드 검색"""
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # 상위 k개 인덱스
        top_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 유사도가 0보다 큰 경우만
                results.append((
                    self.chunks[idx],
                    self.chunk_metadata[idx],
                    similarities[idx]
                ))
        
        return results
    
    def hybrid_search(self, query: str, k: int = 3) -> List[Tuple[str, Dict, float]]:
        """하이브리드 검색"""
        # 의미적 검색과 키워드 검색 결합
        semantic_results = self.semantic_search(query, k)
        keyword_results = self.keyword_search(query, k)
        
        # 결과 통합 (간단한 방식)
        all_results = {}
        
        # 의미적 검색 결과 (가중치 0.6)
        for chunk, metadata, score in semantic_results:
            key = chunk[:50]  # 청크의 처음 50자를 키로 사용
            all_results[key] = (chunk, metadata, score * 0.6)
        
        # 키워드 검색 결과 (가중치 0.4)
        for chunk, metadata, score in keyword_results:
            key = chunk[:50]
            if key in all_results:
                # 이미 있는 경우 점수 합산
                existing_chunk, existing_metadata, existing_score = all_results[key]
                all_results[key] = (existing_chunk, existing_metadata, existing_score + score * 0.4)
            else:
                all_results[key] = (chunk, metadata, score * 0.4)
        
        # 점수순 정렬
        sorted_results = sorted(all_results.values(), key=lambda x: x[2], reverse=True)
        return sorted_results[:k]
    
    def generate_answer(self, query: str, search_method: str = "hybrid") -> Tuple[str, List[Dict]]:
        """답변 생성"""
        if not self.initialized:
            return "시스템이 초기화되지 않았습니다.", []
        
        # 검색 수행
        if search_method == "semantic":
            search_results = self.semantic_search(query)
        elif search_method == "keyword":
            search_results = self.keyword_search(query)
        else:
            search_results = self.hybrid_search(query)
        
        if not search_results:
            return "관련 정보를 찾을 수 없습니다.", []
        
        # 컨텍스트 구성
        context_parts = []
        source_docs = []
        
        for chunk, metadata, score in search_results:
            context_parts.append(f"- {chunk}")
            source_docs.append({
                'chunk': chunk,
                'score': score,
                'metadata': metadata
            })
        
        context = "\n".join(context_parts)
        
        # 컨텍스트 길이 제한 (토큰 오버플로우 방지)
        max_context_length = 800  # 충분한 여유를 두고 설정
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            print(f"⚠️ 컨텍스트가 너무 길어서 {max_context_length}자로 잘렸습니다.")
        
        # 프롬프트 구성 (간소화)
        prompt = f"""참고 정보: {context}

질문: {query}

답변:"""
        
        # API 호출
        answer = self.api_client.simple_chat(prompt)
        
        return answer, source_docs