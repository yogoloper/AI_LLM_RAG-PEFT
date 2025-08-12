"""
간소화된 AI Assistant 웹앱
- RAG 기반 질의응답
- 기본 추론 기능
- 최소한의 UI
"""

import streamlit as st
import time
import os
from api_client import VLLMAPIClient
from rag_system import SimpleRAGSystem

# 페이지 설정 - 원격 서버용
st.set_page_config(
    page_title="AI Assistant", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """세션 상태 초기화"""
    if "api_client" not in st.session_state:
        st.session_state.api_client = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def check_server_status():
    """서버 상태 확인 및 연결"""
    if st.session_state.api_client is None:
        client = VLLMAPIClient()
        if client.health_check():
            st.session_state.api_client = client
            return True
        return False
    return st.session_state.api_client.health_check()

def init_rag_system():
    """RAG 시스템 초기화"""
    if st.session_state.rag_system is None and st.session_state.api_client:
        with st.spinner("RAG 시스템 초기화 중..."):
            rag_system = SimpleRAGSystem(st.session_state.api_client)
            rag_system.initialize()
            st.session_state.rag_system = rag_system
            st.success("✅ RAG 시스템 준비 완료!")
            # 페이지 새로고침으로 상태 반영
            st.rerun()

def main():
    """메인 애플리케이션"""
    init_session_state()
    
    st.title("🤖 AI Assistant")
    st.caption("튜닝된 모델 + RAG 시스템")
    
    # 사이드바 - 시스템 상태
    with st.sidebar:
        st.header("⚙️ 시스템 상태")
        
        # 서버 상태 확인
        server_ok = check_server_status()
        status_color = "🟢" if server_ok else "🔴"
        st.metric("VLLM 서버", f"{status_color} {'연결됨' if server_ok else '연결 안됨'}")
        
        # RAG 시스템 상태
        rag_ok = st.session_state.rag_system is not None
        rag_color = "🟢" if rag_ok else "🟡"
        st.metric("RAG 시스템", f"{rag_color} {'준비됨' if rag_ok else '미준비'}")
        
        st.divider()
        
        # 시스템 설정
        st.subheader("🔧 설정")
        
        # RAG 시스템 상태에 따른 버튼 표시
        if not rag_ok:
            if st.button("🚀 RAG 시스템 초기화"):
                if server_ok:
                    init_rag_system()
                else:
                    st.error("먼저 VLLM 서버를 시작해주세요.")
        else:
            st.info("✅ RAG 시스템이 준비되었습니다!")
        
        if st.button("🔄 시스템 재연결"):
            st.session_state.api_client = None
            st.session_state.rag_system = None
            st.rerun()
    
    # 서버 연결 안된 경우
    if not server_ok:
        st.error("❌ VLLM 서버에 연결할 수 없습니다.")
        st.info("""
        VLLM 서버를 시작해주세요:
        
        ```bash
        # 새 터미널에서 VLLM 서버 시작
        bash start_vllm_server.sh
        ```
        
        **포트 정보:**
        - VLLM 서버: localhost:8000
        - 웹앱: localhost:8501
        """)
        return
    
    # 메인 탭들
    tab1, tab2 = st.tabs(["💬 RAG 채팅", "🔍 기본 추론"])
    
    # RAG 채팅 탭
    with tab1:
        st.subheader("📚 RAG 기반 질의응답")
        
        if not rag_ok:
            st.warning("RAG 시스템을 먼저 초기화해주세요.")
            col1, col2 = st.columns([1, 2])
            with col1:
                if st.button("🚀 지금 초기화하기", type="primary"):
                    init_rag_system()
            with col2:
                st.info("💡 사이드바에서도 초기화할 수 있습니다")
        else:
            # 검색 방법 선택
            search_method = st.selectbox(
                "검색 방법",
                ["hybrid", "semantic", "keyword"],
                index=0,
                help="Hybrid: 의미적 + 키워드 검색 결합"
            )
            
            # 질문 입력
            query = st.text_area(
                "질문을 입력하세요:",
                placeholder="예: Argilla가 무엇인가요?",
                height=100
            )
            
            if st.button("🔍 답변 생성", type="primary"):
                if query.strip():
                    with st.spinner("답변 생성 중..."):
                        answer, docs = st.session_state.rag_system.generate_answer(
                            query, search_method
                        )
                        
                        # 답변 표시
                        st.subheader("🤖 AI 답변")
                        st.write(answer)
                        
                        # 참고 문서들
                        if docs:
                            with st.expander("📄 참고 문서들", expanded=True):
                                for i, doc in enumerate(docs):
                                    st.markdown(f"**문서 {i+1}** (유사도: {doc['score']:.3f})")
                                    st.text(doc['chunk'][:300] + "...")
                                    st.divider()
                else:
                    st.warning("질문을 입력해주세요.")
    
    # 기본 추론 탭
    with tab2:
        st.subheader("💭 기본 추론")
        st.caption("RAG 없이 모델만 사용")
        
        # 채팅 기록 표시
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # 사용자 입력
        if prompt := st.chat_input("메시지를 입력하세요..."):
            # 사용자 메시지 추가
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    response = st.session_state.api_client.simple_chat(prompt)
                    st.write(response)
                    # 응답 기록에 추가
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()