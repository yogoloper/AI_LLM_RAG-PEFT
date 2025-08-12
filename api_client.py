"""
간소화된 API 클라이언트
"""

import requests
import json
import time
import os
from typing import List, Dict, Any, Optional

class VLLMAPIClient:
    """vLLM OpenAI 호환 API 클라이언트"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": "Bearer sk-no-auth-needed",
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    
    def health_check(self) -> bool:
        """서버 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/v1/models", 
                                  headers=self.headers, timeout=10)
            if response.status_code == 200:
                # 사용 가능한 모델 목록 출력
                models = response.json()
                print(f"🤖 사용 가능한 모델: {[model['id'] for model in models.get('data', [])]}")
            return response.status_code == 200
        except:
            return False
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "tuned-model",
        temperature: float = 0.7,
        max_tokens: int = 256,  # 토큰 수 줄임
        **kwargs
    ) -> str:
        """채팅 완성 API 호출"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        try:
            print(f"🔍 API 요청 디버그:")
            print(f"   URL: {self.base_url}/v1/chat/completions")
            print(f"   Model: {model}")
            print(f"   Messages: {len(messages)}개")
            print(f"   첫 메시지: {messages[0] if messages else 'None'}")
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            print(f"   응답 코드: {response.status_code}")
            
            if response.status_code != 200:
                print(f"   응답 내용: {response.text}")
                
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            try:
                error_detail = response.json()
            except:
                error_detail = response.text
            return f"오류: {str(e)}\n상세: {error_detail}"
        except Exception as e:
            return f"오류: {str(e)}"
    
    def simple_chat(self, user_message: str, system_message: str = "답변하세요.") -> str:
        """간단한 채팅"""
        # 시스템 메시지 없이 사용자 메시지만 사용 (토큰 절약)
        messages = [
            {"role": "user", "content": user_message}
        ]
        return self.chat_completion(messages)