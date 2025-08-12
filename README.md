# 🤖 튜닝된 모델 RAG + Inference 챗봇 웹앱

A100 서버에서 `model_tuning.ipynb`로 훈련한 튜닝된 모델을 활용한 RAG 시스템과 기본 추론 기능을 제공하는 챗봇 웹앱입니다.

## 📋 주요 기능

- **🧠 튜닝된 모델**: SFT + DPO 훈련으로 고품질 응답 생성
- **📚 RAG 시스템**: 문서 검색 기반 지식 질의응답 (Hybrid/Semantic/Keyword 검색)
- **💬 기본 추론**: 모델과 직접 대화
- **⚡ A100 최적화**: GPU 메모리 효율성 및 빠른 추론

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 허깅페이스 토큰 설정 (필요시)
export HF_TOKEN=hf_your_token_here

# 리눅스의 경우 dos2unix 설치 
apt update && apt install -y dos2unix
dos2unix setup.sh run_app.sh start_vllm_server.sh start_webapp.sh

# 자동 환경 설정
bash setup.sh
```

### 2. 모델 훈련 (필수)
```bash
# Jupyter 실행하여 모델 튜닝
jupyter lab model_tuning.ipynb

# 또는 Jupyter Notebook
jupyter notebook model_tuning.ipynb
```
> ⚠️ **중요**: `final-tuned-model/` 폴더가 생성될 때까지 훈련을 완료해야 합니다.

### 3. 웹앱 실행
```bash
bash run_app.sh
```

```bash
# 터미널 1: VLLM 서버 시작 (가상환경 자동 활성화)
bash start_vllm_server.sh

# 터미널 2: 웹앱 시작
bash start_webapp.sh
```

### 4. 웹앱 접속
브라우저에서 `http://localhost:8501` 접속

## 📁 파일 구조

```
AI LLM RAG+PEFT/
├── model_tuning.ipynb          # 모델 훈련 노트북 (SFT + DPO)
├── final-tuned-model/          # 훈련된 최종 모델 (자동 생성)
├── webapp.py                   # Streamlit 웹앱 메인 파일
├── api_client.py              # VLLM API 클라이언트 (CORS 해결)
├── rag_system.py              # RAG 시스템 (Hybrid 검색)
├── start_vllm_server.sh       # VLLM 서버 시작 스크립트
├── run_app.sh                 # 통합 실행 안내 스크립트
├── setup.sh                   # 환경 설정 자동화 스크립트
├── webapp_requirements.txt    # 웹앱 필요 패키지 목록
├── dataset.txt               # RAG용 샘플 데이터
└── README.md                 # 이 파일
```

## 🔧 웹앱 사용 방법

### 📚 RAG 채팅 탭
1. **RAG 시스템 초기화**: "🚀 RAG 시스템 초기화" 버튼 클릭
2. **검색 방법 선택**: 
   - `hybrid`: 의미적 + 키워드 검색 결합 (권장)
   - `semantic`: 의미적 검색만
   - `keyword`: 키워드 검색만
3. **질문 입력**: 텍스트 영역에 질문 작성
4. **답변 생성**: "🔍 답변 생성" 버튼 클릭
5. **참고 문서 확인**: 답변 하단의 "📄 참고 문서들" 섹션 확인

### 💭 기본 추론 탭
1. **탭 전환**: "🔍 기본 추론" 탭으로 이동
2. **채팅**: 하단 채팅창에 메시지 입력하여 대화

## ⚠️ 문제 해결

### 🔴 VLLM 서버 연결 실패
```bash
# LoRA 모델 확인 (adapter_config.json 파일 존재 여부)
ls -la final-tuned-model/
ls -la sft-final-model/

# GPU 메모리 확인
nvidia-smi

# 포트 8000 사용 확인
lsof -i :8000

# 서버 재시작
pkill -f vllm
bash start_vllm_server.sh
```

### 🔴 RAG 시스템 초기화 실패
```bash
# 인터넷 연결 확인 (Argilla 데이터셋 다운로드 필요)
ping google.com

# 메모리 부족시 Python 프로세스 재시작
pkill -f streamlit
streamlit run webapp.py
```

### 🔴 CORS 문제
이미 해결되었지만, 문제 발생시:
```bash
# VLLM 서버에 CORS 헤더가 포함되었는지 확인
curl -I http://localhost:8000/v1/models
```

### 🔴 의존성 오류
```bash
# 가상환경 완전 재생성
rm -rf venv
bash setup.sh

# 또는 개별 패키지 재설치
source venv/bin/activate
pip install -r webapp_requirements.txt
```

## 🎯 A100 서버 최적화 설정

### GPU 메모리 최적화
- `gpu-memory-utilization`: 0.9 (90% 사용)
- `max-num-seqs`: 64 (동시 처리 시퀀스)
- `tensor-parallel-size`: 1 (단일 GPU 최적화)

### 성능 최적화
- `enable-chunked-prefill`: 청킹 프리필 활성화
- `max-model-len`: 2048 토큰 (긴 컨텍스트 지원)
- `VLLM_ATTENTION_BACKEND`: FLASH_ATTN 사용

### CORS 해결
- `allowed-origins`: "*" (모든 오리진 허용)
- `allowed-methods`: "*" (모든 HTTP 메서드 허용)
- `allowed-headers`: "*" (모든 헤더 허용)

## 🚀 고급 사용법

### 환경 변수 설정
```bash
# HuggingFace 토큰 (비공개 모델 사용시)
export HF_TOKEN=hf_your_token_here

# GPU 디바이스 선택
export CUDA_VISIBLE_DEVICES=0

# VLLM 백엔드 설정
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

### 모델 경로 커스터마이징
`start_vllm_server.sh`에서 `MODEL_PATHS` 배열 수정:
```bash
MODEL_PATHS=("./your-custom-model" "./final-tuned-model")
```

---