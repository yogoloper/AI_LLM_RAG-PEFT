#!/bin/bash

# A100 서버용 VLLM 서버 시작 스크립트
# model_tuning.ipynb에서 훈련한 튜닝된 모델 사용

echo "🚀 A100 서버에서 VLLM 서버 시작..."

# 가상환경 활성화 확인 및 자동 활성화
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "./venv" ]; then
        echo "🔄 가상환경 활성화 중..."
        source ./venv/bin/activate
        echo "✅ 가상환경 활성화됨: $VIRTUAL_ENV"
    else
        echo "❌ 가상환경을 찾을 수 없습니다."
        echo "💡 먼저 setup.sh를 실행하여 환경을 설정하세요:"
        echo "   bash setup.sh"
        exit 1
    fi
else
    echo "✅ 가상환경 이미 활성화됨: $VIRTUAL_ENV"
fi

# vLLM 설치 확인
if ! python -c "import vllm" >/dev/null 2>&1; then
    echo "❌ vLLM이 설치되지 않았습니다."
    echo "💡 가상환경에서 vLLM을 설치하세요:"
    echo "   pip install vllm"
    exit 1
fi

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# 프로세스 정리 함수
cleanup() {
    echo "\n🛑 서버 종료 중..."
    pkill -f "vllm.entrypoints.openai.api_server"
    exit 0
}

# 신호 처리
trap cleanup SIGINT SIGTERM

# 기존 VLLM 프로세스 종료
echo "🧹 기존 VLLM 프로세스 정리..."
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

# 튜닝된 모델 경로 확인 (순서대로 체크)
MODEL_PATHS=("./final-tuned-model" "./sft-final-model" "./sft-model-advanced" "./dpo-alternative-advanced")
MODEL_PATH=""

for path in "${MODEL_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/adapter_config.json" ]; then
        MODEL_PATH="$path"
        echo "✅ 튜닝된 LoRA 모델 발견: $MODEL_PATH"
        break
    fi
done

if [ -z "$MODEL_PATH" ]; then
    echo "❌ 사용 가능한 모델을 찾을 수 없습니다."
    echo "다음 경로들을 확인했습니다:"
    for path in "${MODEL_PATHS[@]}"; do
        echo "  - $path"
    done
    echo "💡 model_tuning.ipynb를 먼저 실행해주세요."
    exit 1
fi

echo "📦 사용할 모델: $MODEL_PATH"

# GPU 상태 확인
echo "🖥️  GPU 상태 확인:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits

echo "\n🔧 VLLM 서버 설정 (메모리 최적화):"
echo "   - 베이스 모델: meta-llama/Llama-2-7b-chat-hf"
echo "   - LoRA 어댑터: $MODEL_PATH"
echo "   - 포트: 8000"
echo "   - GPU 메모리 사용률: 70% (0.7)"
echo "   - 최대 토큰: 2048"
echo "   - 최대 동시 시퀀스: 32개"
echo "   - CORS: 기본 설정"

# 포트 사용 확인
if command -v lsof >/dev/null 2>&1 && lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  포트 8000이 이미 사용 중입니다. 기존 프로세스를 종료합니다."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 2
fi

echo "\n🚀 VLLM 서버 시작 중..."
echo "📡 접근URL: http://localhost:8000"
echo "🔗 Health Check: http://localhost:8000/v1/models"
echo "\n[Ctrl+C로 종료]"

# VLLM 서버 시작 (A100 최적화 설정 + LoRA 어댑터)
python -m vllm.entrypoints.openai.api_server \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --enable-lora \
    --lora-modules tuned-model="$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name "tuned-model" \
    --max-model-len 2048 \
    --tensor-parallel-size 1 \
    --dtype auto \
    --trust-remote-code \
    --disable-log-stats \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 32 \
    --swap-space 2 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048

echo "\n🏁 VLLM 서버가 종료되었습니다."