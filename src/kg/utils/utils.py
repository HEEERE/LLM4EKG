#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具类模块
包含文本处理、日志配置、文件操作等工具函数
"""

import os
import json
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import tiktoken
import re


def setup_logging(log_file: str = None, level: str = "INFO", 
                 format_string: str = None) -> logging.Logger:
    """设置日志配置
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
        format_string: 日志格式字符串
        
    Returns:
        日志记录器
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有的处理器
    logger.handlers = []
    
    # 创建格式化器
    formatter = logging.Formatter(format_string)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """加载JSON数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        数据列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保返回列表格式
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"不支持的JSON格式: {type(data)}")
            
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {str(e)}")
    except Exception as e:
        raise ValueError(f"加载数据失败: {str(e)}")


def save_json_data(data: Any, file_path: str, indent: int = 2):
    """保存JSON数据
    
    Args:
        data: 要保存的数据
        file_path: 输出文件路径
        indent: JSON缩进
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            
    except Exception as e:
        raise ValueError(f"保存数据失败: {str(e)}")


def calculate_text_hash(text: str) -> str:
    """计算文本的哈希值
    
    Args:
        text: 输入文本
        
    Returns:
        MD5哈希值
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """计算文本的token数量
    
    Args:
        text: 输入文本
        encoding_name: 编码名称
        
    Returns:
        token数量
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        # 如果tiktoken不可用，使用简单的字符计数作为近似
        return len(text) // 2  # 粗略估计：中文字符约占2个token


def chunk_text(text: str, chunk_size: int = 2048, overlap: int = 50) -> List[str]:
    """将文本分块
    
    Args:
        text: 输入文本
        chunk_size: 每个块的最大token数
        overlap: 块之间的重叠token数
        
    Returns:
        文本块列表
    """
    if not text:
        return []
    
    # 计算文本的token数量
    total_tokens = count_tokens(text)
    
    # 如果文本token数小于块大小，直接返回
    if total_tokens <= chunk_size:
        return [text]
    
    # 按句子分割文本
    sentences = re.split(r'[。！？；]\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        
        # 如果当前句子加上重叠部分超过了块大小，开始新块
        if current_tokens + sentence_tokens > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
            
            # 新块包含重叠部分
            if chunks and overlap > 0:
                # 从上一个块中取重叠部分
                overlap_text = get_overlap_text(chunks[-1], overlap)
                current_chunk = overlap_text + sentence
                current_tokens = count_tokens(current_chunk)
            else:
                current_chunk = sentence
                current_tokens = sentence_tokens
        else:
            # 添加到当前块
            if current_chunk:
                current_chunk += "。" + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_tokens
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def get_overlap_text(text: str, overlap_tokens: int) -> str:
    """获取文本的重叠部分
    
    Args:
        text: 输入文本
        overlap_tokens: 重叠token数
        
    Returns:
        重叠文本
    """
    if not text or overlap_tokens <= 0:
        return ""
    
    # 按句子分割
    sentences = re.split(r'[。！？；]\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    overlap_text = ""
    total_tokens = 0
    
    # 从后往前取句子，直到达到重叠token数
    for sentence in reversed(sentences):
        sentence_tokens = count_tokens(sentence)
        if total_tokens + sentence_tokens <= overlap_tokens:
            if overlap_text:
                overlap_text = sentence + "。" + overlap_text
            else:
                overlap_text = sentence
            total_tokens += sentence_tokens
        else:
            break
    
    return overlap_text + "。" if overlap_text else ""


def clean_text(text: str) -> str:
    """清理文本
    
    Args:
        text: 输入文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符（保留中文、英文、数字和常用标点）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s。，！？；：""''（）()【】\[\]<>《》]', '', text)
    
    # 移除多余的标点
    text = re.sub(r'[。，！？；：]{2,}', '。', text)
    
    # 移除首尾空白
    text = text.strip()
    
    return text


def validate_entity(entity_name: str, entity_type: str) -> bool:
    """验证实体
    
    Args:
        entity_name: 实体名称
        entity_type: 实体类型
        
    Returns:
        是否有效
    """
    if not entity_name or len(entity_name.strip()) < 2:
        return False
    
    # 实体名称不能全是数字
    if entity_name.isdigit():
        return False
    
    # 实体名称不能包含过多特殊字符
    special_chars = len(re.findall(r'[^\u4e00-\u9fa5a-zA-Z0-9]', entity_name))
    if special_chars > len(entity_name) * 0.3:  # 特殊字符超过30%
        return False
    
    return True


def validate_relation(subject: str, predicate: str, obj: str) -> bool:
    """验证关系
    
    Args:
        subject: 主体
        predicate: 谓词
        obj: 客体
        
    Returns:
        是否有效
    """
    if not subject or not predicate or not obj:
        return False
    
    if len(subject.strip()) < 2 or len(predicate.strip()) < 2 or len(obj.strip()) < 2:
        return False
    
    # 主体和客体不能相同
    if subject.strip() == obj.strip():
        return False
    
    return True


def format_timestamp(timestamp: float = None) -> str:
    """格式化时间戳
    
    Args:
        timestamp: 时间戳（可选，默认为当前时间）
        
    Returns:
        格式化的时间字符串
    """
    if timestamp is None:
        timestamp = time.time()
    
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """安全获取字典值
    
    Args:
        dictionary: 字典
        key: 键
        default: 默认值
        
    Returns:
        键对应的值或默认值
    """
    try:
        return dictionary.get(key, default)
    except (AttributeError, TypeError):
        return default


def batch_process(items: List[Any], batch_size: int = 10) -> List[List[Any]]:
    """批量处理
    
    Args:
        items: 项目列表
        batch_size: 批量大小
        
    Returns:
        批量列表
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def normalize_text(text: str) -> str:
    """标准化文本
    
    Args:
        text: 输入文本
        
    Returns:
        标准化后的文本
    """
    if not text:
        return ""
    
    # 转换为半角字符
    text = fullwidth_to_halfwidth(text)
    
    # 统一标点符号
    text = unify_punctuation(text)
    
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def fullwidth_to_halfwidth(text: str) -> str:
    """全角转半角
    
    Args:
        text: 输入文本
        
    Returns:
        转换后的文本
    """
    result = ""
    for char in text:
        code = ord(char)
        # 全角空格直接转换
        if code == 0x3000:
            result += " "
        # 全角字符（除空格）根据关系转化
        elif 0xFF01 <= code <= 0xFF5E:
            result += chr(code - 0xFEE0)
        else:
            result += char
    return result


def unify_punctuation(text: str) -> str:
    """统一标点符号
    
    Args:
        text: 输入文本
        
    Returns:
        统一后的文本
    """
    # 统一引号
    text = re.sub(r'["""]', '"', text)
    text = re.sub(r"['']", "'", text)
    
    # 统一括号
    text = re.sub(r'[（]', '(', text)
    text = re.sub(r'[）]', ')', text)
    
    # 统一破折号
    text = re.sub(r'[—–]', '-', text)
    
    return text


def extract_text_features(text: str) -> Dict[str, Any]:
    """提取文本特征
    
    Args:
        text: 输入文本
        
    Returns:
        文本特征字典
    """
    if not text:
        return {}
    
    # 基本统计
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len(re.split(r'[。！？；]', text))
    
    # 中文统计
    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    chinese_ratio = chinese_chars / char_count if char_count > 0 else 0
    
    # 数字统计
    numbers = len(re.findall(r'\d+', text))
    number_ratio = numbers / word_count if word_count > 0 else 0
    
    # 标点统计
    punctuations = len(re.findall(r'[，。！？；：""''（）()【】[\]<>《》]', text))
    punctuation_ratio = punctuations / char_count if char_count > 0 else 0
    
    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "chinese_chars": chinese_chars,
        "chinese_ratio": chinese_ratio,
        "numbers": numbers,
        "number_ratio": number_ratio,
        "punctuations": punctuations,
        "punctuation_ratio": punctuation_ratio
    }


# 使用示例
if __name__ == "__main__":
    # 测试文本处理函数
    test_text = "这是一个测试文本。包含中文、English、123数字和标点符号！"
    
    print("原始文本:", test_text)
    print("清理后文本:", clean_text(test_text))
    print("Token数量:", count_tokens(test_text))
    print("文本特征:", extract_text_features(test_text))
    
    # 测试分块
    long_text = "这是第一段。这是第二段。这是第三段。这是第四段。这是第五段。"
    chunks = chunk_text(long_text, chunk_size=10, overlap=5)
    print("分块结果:", chunks)
    
    # 测试实体验证
    print("实体验证:", validate_entity("应急管理部", "组织机构"))
    print("关系验证:", validate_relation("应急管理部", "负责", "全国应急管理工作"))