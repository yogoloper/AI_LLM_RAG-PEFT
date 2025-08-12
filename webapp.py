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
    tab1, tab2, tab3 = st.tabs(["💬 RAG 채팅", "📄 PDF 관리", "🔍 기본 추론"])
    
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
                                    metadata = doc['metadata']
                                    source_type = metadata.get('source_type', 'basic')
                                    
                                    if source_type == 'pdf':
                                        # PDF 문서인 경우
                                        st.markdown(f"**📄 PDF 문서 {i+1}** (유사도: {doc['score']:.3f})")
                                        st.markdown(f"*파일명: {metadata.get('filename', 'Unknown')}*")
                                        st.markdown(f"*업로드: {metadata.get('upload_time', 'Unknown')}*")
                                    else:
                                        # 기본 데이터인 경우
                                        st.markdown(f"**📚 기본 문서 {i+1}** (유사도: {doc['score']:.3f})")
                                        if 'user_message' in metadata:
                                            st.markdown(f"*관련 질문: {metadata['user_message']}*")
                                    
                                    st.text(doc['chunk'][:300] + "...")
                                    st.divider()
                else:
                    st.warning("질문을 입력해주세요.")
    
    # PDF 관리 탭
    with tab2:
        st.subheader("📄 PDF 문서 관리")
        
        if not rag_ok:
            st.warning("RAG 시스템을 먼저 초기화해주세요.")
        else:
            # PDF 업로드 섹션
            st.markdown("### 📤 PDF 업로드")
            uploaded_files = st.file_uploader(
                "PDF 파일을 선택하세요",
                type=['pdf'],
                accept_multiple_files=True,
                help="여러 PDF 파일을 동시에 업로드할 수 있습니다."
            )
            
            if uploaded_files:
                upload_col1, upload_col2 = st.columns([1, 1])
                
                with upload_col1:
                    if st.button("📤 업로드 시작", type="primary"):
                        success_count = 0
                        for uploaded_file in uploaded_files:
                            with st.spinner(f"'{uploaded_file.name}' 처리 중..."):
                                # 중복 파일명 체크
                                existing_files = [doc['filename'] for doc in st.session_state.rag_system.pdf_documents]
                                if uploaded_file.name in existing_files:
                                    st.warning(f"⚠️ '{uploaded_file.name}'은 이미 업로드된 파일입니다.")
                                    continue
                                
                                # PDF 추가
                                if st.session_state.rag_system.add_pdf_document(uploaded_file, uploaded_file.name):
                                    success_count += 1
                        
                        if success_count > 0:
                            st.success(f"✅ {success_count}개 파일 업로드 완료!")
                            st.rerun()  # 페이지 새로고침으로 상태 업데이트
                
                with upload_col2:
                    st.info(f"선택된 파일: {len(uploaded_files)}개")
                    for file in uploaded_files:
                        st.text(f"• {file.name} ({file.size:,} bytes)")
            
            st.divider()
            
            # 업로드된 PDF 목록
            st.markdown("### 📋 업로드된 PDF 목록")
            pdf_summary = st.session_state.rag_system.get_pdf_summary()
            
            if pdf_summary['total_pdfs'] == 0:
                st.info("업로드된 PDF 문서가 없습니다.")
            else:
                # 요약 정보
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 PDF 수", pdf_summary['total_pdfs'])
                with col2:
                    st.metric("총 청크 수", pdf_summary['total_pdf_chunks'])
                with col3:
                    basic_chunks = len(st.session_state.rag_system.chunks) - pdf_summary['total_pdf_chunks']
                    st.metric("기본 데이터 청크", basic_chunks)
                
                st.divider()
                
                # PDF 문서 목록
                for i, doc in enumerate(pdf_summary['documents']):
                    with st.expander(f"📄 {doc['filename']}", expanded=False):
                        info_col1, info_col2, action_col = st.columns([2, 2, 1])
                        
                        with info_col1:
                            st.text(f"업로드 시간: {doc['upload_time']}")
                            st.text(f"청크 수: {doc['chunk_count']}개")
                        
                        with info_col2:
                            st.text(f"텍스트 길이: {len(doc['text']):,}자")
                            st.text(f"텍스트 미리보기:")
                            st.text(doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'])
                        
                        with action_col:
                            if st.button("🗑️ 삭제", key=f"delete_{i}"):
                                if st.session_state.rag_system.remove_pdf_document(doc['filename']):
                                    st.rerun()
    
    # 기본 추론 탭
    with tab3:
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