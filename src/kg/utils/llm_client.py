#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM客户端模块
用于与各种大语言模型API进行交互
"""

import logging
import time
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import os
import hashlib
import threading
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_TRACE_LOCK = threading.Lock()


def _trace_enabled() -> bool:
    v = os.getenv("KG_LLM_TRACE", "").strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    return False


def _default_trace_path() -> str:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / "llm_trace.jsonl")


def _trace_path() -> str:
    p = os.getenv("KG_LLM_TRACE_PATH", "").strip()
    if p:
        try:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p
    return _default_trace_path()


def _sha256_text(text: str) -> str:
    try:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    except Exception:
        return ""


def _append_trace_record(record: Dict[str, Any]) -> None:
    if not _trace_enabled():
        return
    try:
        path = _trace_path()
        line = json.dumps(record, ensure_ascii=False)
        with _TRACE_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        return


@dataclass
class LLMResponse:
    """LLM响应类"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    response_time: float


class LLMClient:
    """LLM客户端类"""
    
    def __init__(self, api_key: str, model: str = "glm-4-flash", 
                 max_tokens: int = 4096, temperature: float = 0.3,
                 timeout: int = 30, max_retries: int = 3):
        """初始化LLM客户端
        
        Args:
            api_key: API密钥
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 智谱AI API配置
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"LLM客户端初始化完成，模型: {model}")
    
    def invoke(self, prompt: str, system_message: str = None) -> LLMResponse:
        """调用LLM API
        
        Args:
            prompt: 用户提示词
            system_message: 系统消息（可选）
            
        Returns:
            LLM响应
        """
        start_time = time.time()
        
        # 构建消息
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # 构建请求数据
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False
        }
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"调用LLM API，尝试 {attempt + 1}/{self.max_retries}")
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=data,
                    timeout=self.timeout
                )
                
                # 检查响应状态
                response.raise_for_status()
                
                # 解析响应
                result = response.json()
                
                # 提取内容
                content = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                finish_reason = result["choices"][0].get("finish_reason", "stop")
                
                response_time = time.time() - start_time
                
                logger.debug(f"LLM API调用成功，耗时: {response_time:.2f}秒")
                _append_trace_record({
                    "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "provider": "zhipu",
                    "endpoint": "chat.completions",
                    "model": self.model,
                    "system_message": system_message,
                    "prompt_len": len(prompt or ""),
                    "prompt_sha256": _sha256_text(prompt or ""),
                    "response_text": content,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "response_time_sec": response_time,
                })
                
                return LLMResponse(
                    content=content,
                    model=self.model,
                    usage=usage,
                    finish_reason=finish_reason,
                    response_time=response_time
                )
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"LLM API调用失败，尝试 {attempt + 1}/{self.max_retries}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    # 指数退避
                    wait_time = 2 ** attempt
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM API调用失败，已达到最大重试次数: {str(e)}")
                    raise
            
            except Exception as e:
                logger.error(f"LLM API调用发生未知错误: {str(e)}")
                raise
        
        # 如果所有重试都失败，抛出异常
        raise RuntimeError("LLM API调用失败，已达到最大重试次数")
    
    def batch_invoke(self, prompts: List[str], system_message: str = None) -> List[LLMResponse]:
        """批量调用LLM API
        
        Args:
            prompts: 提示词列表
            system_message: 系统消息（可选）
            
        Returns:
            响应列表
        """
        responses = []
        
        for i, prompt in enumerate(prompts):
            try:
                logger.debug(f"批量调用LLM API，第 {i + 1}/{len(prompts)} 个")
                response = self.invoke(prompt, system_message)
                responses.append(response)
                
                # API调用间隔
                if i < len(prompts) - 1:
                    time.sleep(1)  # 避免过于频繁的API调用
                    
            except Exception as e:
                logger.error(f"批量调用LLM API失败，第 {i + 1} 个: {str(e)}")
                # 添加错误响应
                responses.append(LLMResponse(
                    content=f"Error: {str(e)}",
                    model=self.model,
                    usage={},
                    finish_reason="error",
                    response_time=0
                ))
        
        return responses
    
    def embed(self, inputs: List[str], dimensions: Optional[int] = None) -> List[List[float]]:
        start_time = time.time()
        data: Dict[str, Any] = {
            "model": self.model if "embedding" in self.model else "embedding-3",
            "input": inputs
        }
        if dimensions:
            data["dimensions"] = dimensions
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()
                vecs = [item.get("embedding", []) for item in result.get("data", [])]
                response_time = time.time() - start_time
                _append_trace_record({
                    "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "provider": "zhipu",
                    "endpoint": "embeddings",
                    "model": data.get("model"),
                    "input_count": len(inputs or []),
                    "dimensions": dimensions,
                    "response_time_sec": response_time,
                    "vector_sizes": [len(v) for v in vecs[:10]],
                })
                return vecs
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                raise
    
    def test_connection(self) -> bool:
        """测试API连接
        
        Returns:
            是否连接成功
        """
        try:
            response = self.invoke("你好，这是一个测试消息。请回复'连接成功'。")
            return "连接成功" in response.content or "成功" in response.content
        except Exception as e:
            logger.error(f"LLM API连接测试失败: {str(e)}")
            return False


class AliyunBailianClient:
    """阿里云百炼客户端类 - 使用OpenAI兼容模式"""
    
    def __init__(self, api_key: str, model: str = "deepseek-v3.2-exp", 
                 max_tokens: int = 4096, temperature: float = 0.3,
                 timeout: int = 30, max_retries: int = 3):
        """初始化阿里云百炼客户端
        
        Args:
            api_key: API密钥
            model: 模型名称（默认deepseek-v3.2-exp）
            max_tokens: 最大token数
            temperature: 温度参数
            timeout: 超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        
        # 使用OpenAI兼容模式
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.use_openai_mode = True
            logger.info(f"阿里云百炼客户端初始化成功（OpenAI兼容模式），模型: {model}")
        except ImportError:
            # 如果openai库不可用，使用原生HTTP模式
            self.use_openai_mode = False
            self.base_url = "https://dashscope.aliyuncs.com/api/v1"
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            logger.info(f"阿里云百炼客户端初始化成功（HTTP模式），模型: {model}")
    
    def invoke(self, prompt: str, system_message: str = None) -> LLMResponse:
        """调用阿里云百炼API
        
        Args:
            prompt: 用户提示词
            system_message: 系统消息（可选）
            
        Returns:
            LLM响应
        """
        start_time = time.time()
        
        if self.use_openai_mode:
            return self._invoke_openai_mode(prompt, system_message, start_time)
        else:
            return self._invoke_http_mode(prompt, system_message, start_time)
    
    def _invoke_openai_mode(self, prompt: str, system_message: str = None, start_time: float = 0) -> LLMResponse:
        """使用OpenAI兼容模式调用"""
        try:
            logger.debug(f"使用OpenAI兼容模式调用阿里云百炼API")
            
            # 构建消息
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # 调用API - 使用参考代码中的参数格式
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                # 添加enable_thinking参数（仅对deepseek-v3.2-exp和deepseek-v3.1有效）
                extra_body={"enable_thinking": True} if "deepseek-v3.2" in self.model or "deepseek-v3.1" in self.model else None
            )
            
            choice0 = response.choices[0]
            msg = getattr(choice0, "message", None)
            content = getattr(msg, "content", None) if msg is not None else None
            if isinstance(content, dict):
                content = json.dumps(content, ensure_ascii=False)
            if isinstance(content, list):
                content = "\n".join(str(x) for x in content)

            def _iter_strings(x):
                if x is None:
                    return
                if isinstance(x, str):
                    yield x
                    return
                if isinstance(x, dict):
                    for v in x.values():
                        yield from _iter_strings(v)
                    return
                if isinstance(x, (list, tuple)):
                    for v in x:
                        yield from _iter_strings(v)
                    return

            def _pick_best_string(strings):
                best = ""
                best_score = -1
                for s in strings:
                    if not isinstance(s, str):
                        continue
                    st = s.strip()
                    if not st:
                        continue
                    score = 0
                    if "entities" in st and "relations" in st:
                        score += 5
                    if "alias_map" in st or "bridge_relations" in st:
                        score += 4
                    if "{" in st and "}" in st:
                        score += 2
                    score += min(len(st), 20000) / 20000
                    if score > best_score:
                        best_score = score
                        best = st
                return best

            if not content or not str(content).strip():
                dump_obj = None
                try:
                    dump_obj = choice0.model_dump()
                except Exception:
                    try:
                        dump_obj = msg.model_dump() if msg is not None and hasattr(msg, "model_dump") else None
                    except Exception:
                        dump_obj = getattr(msg, "__dict__", None)
                content = _pick_best_string(_iter_strings(dump_obj)) if dump_obj is not None else ""
                if not content and dump_obj is not None:
                    try:
                        content = json.dumps(dump_obj, ensure_ascii=False)
                    except Exception:
                        content = str(dump_obj)

            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            finish_reason = choice0.finish_reason or "stop"
            response_time = time.time() - start_time
            
            logger.debug(f"阿里云百炼API调用成功（OpenAI模式），耗时: {response_time:.2f}秒")
            _append_trace_record({
                "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "provider": "aliyun",
                "endpoint": "chat.completions",
                "mode": "openai_compatible",
                "model": self.model,
                "system_message": system_message,
                "prompt_len": len(prompt or ""),
                "prompt_sha256": _sha256_text(prompt or ""),
                "response_text": content,
                "finish_reason": finish_reason,
                "usage": usage,
                "response_time_sec": response_time,
            })
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                finish_reason=finish_reason,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"OpenAI兼容模式调用失败: {str(e)}")
            raise
    
    def _invoke_http_mode(self, prompt: str, system_message: str = None, start_time: float = 0) -> LLMResponse:
        """使用HTTP模式调用"""
        # 构建消息
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # 构建请求数据
        data = {
            "model": self.model,
            "input": {
                "messages": messages
            },
            "parameters": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "result_format": "message",
                "enable_search": False,
                "incremental_output": False
            }
        }
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"使用HTTP模式调用阿里云百炼API，尝试 {attempt + 1}/{self.max_retries}")
                
                response = requests.post(
                    f"{self.base_url}/services/aigc/text-generation/generation",
                    headers=self.headers,
                    json=data,
                    timeout=self.timeout
                )
                
                # 检查响应状态
                response.raise_for_status()
                
                # 解析响应
                result = response.json()
                
                # 检查响应状态码
                if result.get("status_code") != 200:
                    raise Exception(f"API返回错误: {result.get('message', '未知错误')}")
                
                # 提取内容
                content = result["output"]["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                finish_reason = result["output"]["choices"][0].get("finish_reason", "stop")
                
                response_time = time.time() - start_time
                
                logger.debug(f"阿里云百炼API调用成功（HTTP模式），耗时: {response_time:.2f}秒")
                _append_trace_record({
                    "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "provider": "aliyun",
                    "endpoint": "text-generation.generation",
                    "mode": "http",
                    "model": self.model,
                    "system_message": system_message,
                    "prompt_len": len(prompt or ""),
                    "prompt_sha256": _sha256_text(prompt or ""),
                    "response_text": content,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "response_time_sec": response_time,
                })
                
                return LLMResponse(
                    content=content,
                    model=self.model,
                    usage=usage,
                    finish_reason=finish_reason,
                    response_time=response_time
                )
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"阿里云百炼API调用失败，尝试 {attempt + 1}/{self.max_retries}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    # 指数退避
                    wait_time = 2 ** attempt
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"阿里云百炼API调用失败，已达到最大重试次数: {str(e)}")
                    raise
            
            except Exception as e:
                logger.error(f"阿里云百炼API调用发生未知错误: {str(e)}")
                raise
        
        # 如果所有重试都失败，抛出异常
        raise RuntimeError("阿里云百炼API调用失败，已达到最大重试次数")
    
    def test_connection(self) -> bool:
        """测试API连接
        
        Returns:
            是否连接成功
        """
        try:
            response = self.invoke("你好，这是一个测试消息。请回复'连接成功'。")
            return "连接成功" in response.content or "成功" in response.content
        except Exception as e:
            logger.error(f"阿里云百炼API连接测试失败: {str(e)}")
            return False


# 统一的LLM客户端工厂
class LLMClientFactory:
    """LLM客户端工厂类"""
    
    @staticmethod
    def create_client(provider: str, api_key: str, **kwargs) -> LLMClient:
        """创建LLM客户端
        
        Args:
            provider: 提供商（"zhipu", "dashscope", "aliyun"）
            api_key: API密钥
            **kwargs: 其他参数
            
        Returns:
            LLM客户端实例
        """
        if provider.lower() == "zhipu":
            return LLMClient(api_key=api_key, **kwargs)
        elif provider.lower() == "dashscope":
            return DashScopeClient(api_key=api_key, **kwargs)
        elif provider.lower() == "aliyun":
            return AliyunBailianClient(api_key=api_key, **kwargs)
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")
