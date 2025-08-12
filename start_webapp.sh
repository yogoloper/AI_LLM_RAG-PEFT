#!/bin/bash

# Streamlit 웹앱 시작 스크립트 (CORS 문제 해결)

echo "🌐 Streamlit 웹앱 시작 (CORS 해결)..."

# 가상환경 활성화 확인
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "./venv" ]; then
        echo "🔄 가상환경 활성화 중..."
        source ./venv/bin/activate
        echo "✅ 가상환경 활성화됨"
    else
        echo "❌ 가상환경을 찾을 수 없습니다."
        echo "💡 먼저 setup.sh를 실행하세요."
        exit 1
    fi
fi

# 기존 Streamlit 프로세스 정리
echo "🧹 기존 Streamlit 프로세스 정리..."
pkill -f "streamlit run webapp.py" 2>/dev/null || true
sleep 2

# VLLM 서버 연결 확인
echo "🔗 VLLM 서버 연결 확인..."
if curl -s http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "✅ VLLM 서버 연결 성공"
else
    echo "⚠️  VLLM 서버에 연결할 수 없습니다."
    echo "💡 먼저 다른 터미널에서 VLLM 서버를 시작해주세요:"
    echo "   bash start_vllm_server.sh"
fi

echo "\n🚀 Streamlit 웹앱 시작 (CORS 비활성화)..."
echo "📱 웹 브라우저에서 다음 주소로 접속하세요:"
echo "   http://localhost:8501"
echo "\n[Ctrl+C로 종료]"

# Streamlit 실행 (CORS 비활성화)
streamlit run webapp.py \
    --server.port=8501 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false

echo "\n🏁 Streamlit 웹앱이 종료되었습니다."