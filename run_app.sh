#!/bin/bash

# A100 서버에서 튜닝된 모델 + RAG 웹앱 실행 스크립트

echo "🚀 튜닝된 모델 RAG 웹앱 시작"
echo "================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로깅 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# GPU 확인
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log_error "nvidia-smi를 찾을 수 없습니다. CUDA 환경을 확인하세요."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
log_success "GPU ${GPU_COUNT}개 감지됨"

# 튜닝된 모델 확인
MODEL_PATHS=("./final-tuned-model" "./sft-final-model")
MODEL_PATH=""

for path in "${MODEL_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/adapter_config.json" ]; then
        MODEL_PATH="$path"
        log_success "튜닝된 LoRA 모델 발견: $MODEL_PATH"
        break
    fi
done

if [ -z "$MODEL_PATH" ]; then
    log_warning "튜닝된 모델을 찾을 수 없습니다."
    log_info "먼저 model_tuning.ipynb를 실행하여 모델을 튜닝하세요."
    exit 1
fi

# 프로세스 정리
log_info "기존 프로세스 정리..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -f "streamlit run webapp.py" 2>/dev/null || true
sleep 3

echo ""
log_info "🔧 실행 안내:"
echo "1️⃣  먼저 VLLM 서버를 시작합니다 (터미널 1)"
echo "2️⃣  그 다음 웹앱을 시작합니다 (터미널 2)"
echo ""

echo "터미널 1에서 다음 명령어를 실행하세요:"
echo -e "${GREEN}bash start_vllm_server.sh${NC}"
echo ""

echo "VLLM 서버가 준비되면 터미널 2에서 다음 명령어를 실행하세요:"
echo -e "${GREEN}bash start_webapp.sh${NC}"
echo ""

log_info "📱 웹앱 접속: http://localhost:8501"
log_info "🤖 VLLM API: http://localhost:8000"

echo ""
log_success "CORS 문제가 해결된 설정으로 실행됩니다!"
echo "🎉 RAG 기능과 inference 기능을 모두 사용할 수 있습니다."