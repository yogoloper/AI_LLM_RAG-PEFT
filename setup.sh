#!/bin/bash

# AI Assistant 웹앱 설정 스크립트
# model_tuning.ipynb와 함께 사용

echo "🚀 AI Assistant 웹앱 설정 시작..."

# 1. Python 버전 확인
if command -v python3 &> /dev/null; then
    PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo "✅ Python 버전: $PYTHON_VER"
else
    echo "❌ Python3가 설치되어 있지 않습니다."
    exit 1
fi

# 2. 가상환경 생성
VENV_NAME="venv"
if [ -d "$VENV_NAME" ]; then
    echo "⚠️ 기존 가상환경이 존재합니다. 재생성합니다."
    rm -rf "$VENV_NAME"
fi

echo "📦 가상환경 생성 중..."
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# 3. pip 업그레이드
echo "⬆️ pip 업그레이드..."
pip install --upgrade pip

# 4. PyTorch 설치 (CUDA 지원)
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 CUDA 지원 PyTorch 설치..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 CPU 전용 PyTorch 설치..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 5. 기본 의존성 설치
echo "📚 기본 의존성 설치..."
pip install \
    streamlit \
    requests \
    sentence-transformers \
    scikit-learn \
    faiss-cpu \
    datasets \
    numpy \
    pandas

# 6. vLLM 설치
echo "⚡ vLLM 설치..."
pip install vllm

# 7. 설치 확인
echo "✅ 설치 완료! 패키지 확인 중..."
python3 -c "
import streamlit
import sentence_transformers
import sklearn
import faiss
import datasets
import vllm
print('🎉 모든 패키지가 정상적으로 설치되었습니다!')
"

echo "
🎯 설치가 완료되었습니다!

사용 방법:
1. 가상환경 활성화: source venv/bin/activate
2. VLLM 서버 시작: bash start_vllm_server.sh
3. 웹앱 실행: bash start_webapp.sh

주의사항:
- model_tuning.ipynb를 먼저 실행하여 모델을 훈련해주세요
- final-tuned-model/ 디렉터리가 있어야 합니다
"