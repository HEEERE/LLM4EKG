#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应急条例知识图谱重构版配置文件

包含所有项目配置参数，便于统一管理和修改。
"""

import os
from pathlib import Path
from typing import Dict, Any

KG_DIR = Path(__file__).resolve().parents[1]
_default_data_dir = str(KG_DIR / "data")
_default_output_dir = str(KG_DIR / "outputs")
_output_dir = os.getenv("KG_OUTPUT_DIR", _default_output_dir)
_log_dir = os.getenv("KG_LOG_DIR", _output_dir)

# ============================================================================
# 数据库配置
# ============================================================================

NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "username": os.getenv("NEO4J_USERNAME", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", ""),
    "database": os.getenv("NEO4J_DATABASE", "neo4j"),
}

# ============================================================================
# API配置
# ============================================================================

API_CONFIG = {
    "zhipu_api_key": os.getenv("ZHIPU_API_KEY", ""),
    "dashscope_api_key": os.getenv("DASHSCOPE_API_KEY", ""),
    "aliyun_api_key": os.getenv("ALIYUN_API_KEY", ""),
    "zhipu_model": os.getenv("ZHIPU_MODEL", "glm-4-flash"),
    "aliyun_model": os.getenv("ALIYUN_MODEL", "deepseek-r1-0528"),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "4096")),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
    "api_retry_times": int(os.getenv("LLM_API_RETRY_TIMES", "3")),
    "api_retry_delay": float(os.getenv("LLM_API_RETRY_DELAY", "5")),
    "api_call_interval": float(os.getenv("LLM_API_CALL_INTERVAL", "2")),
}

# ============================================================================
# 文档处理配置
# ============================================================================

DOCUMENT_CONFIG = {
    "chunk_token_num": 700,  # 每个分块的最大token数（调大上下文粒度，建议600-800）
    "encoding_name": "cl100k_base",  # tiktoken编码方式
    "overlap_size": 150,  # 分块重叠大小（提高跨句关系抽取）
    "min_chunk_size": 100,  # 最小分块大小
    "max_chunk_size": 4000,  # 最大分块大小
    "batch_size": 10,  # 批处理大小
    "group_size": 12   # 每组规则合并条数，降低块内主题混杂
}

# ============================================================================
# 实体识别配置
# ============================================================================

ENTITY_EXTRACTION_CONFIG = {
    "entity_types": [
        "组织机构", "人员", "地点", "物资", "事件", "行动", "法规", "时间"
    ],
    "min_entity_length": 2,
    "max_entity_length": 50,
    "confidence_threshold": 0.7
}

# ============================================================================
# 关系抽取配置
# ============================================================================

RELATION_EXTRACTION_CONFIG = {
    "relation_types": [
        "负责", "管理", "指导", "协调", "监督", "承担", "组织", "领导",
        "参与", "支持", "配合", "报告", "处置", "响应", "启动", "终止",
        "包含", "属于", "位于", "使用", "需要", "要求", "规定", "制定"
    ],
    "confidence_threshold": 0.6,  # 关系置信度阈值
    "max_relations_per_text": 50  # 每段文本最大关系数量
}

# ============================================================================
# 桥接关系与语义蕴含验证配置
# ============================================================================

BRIDGE_SCHEMA_PREDICATES = [
    "负责", "主管", "协同", "管辖", "执行",
    "资源需求", "位于", "适用于", "适用范围", "依据",
    "处置", "响应", "采取", "需要", "使用", "相关", "涉及"
]

NLI_CONFIG = {
    "enabled": True,
    "model_name": "microsoft/deberta-v3-large-mnli",
    "threshold": 0.85,
    "device": -1,
    "cache_dir": os.getenv("NLI_CACHE_DIR", str(Path(_output_dir) / "hf_cache")),
    "local_dir": os.getenv("NLI_LOCAL_DIR", str(Path(_output_dir) / "models_nli" / "deberta-v3-large-mnli")),
}

# ============================================================================
# 日志配置
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": os.getenv("KG_LOG_FILE", str(Path(_log_dir) / "knowledge_graph.log")),
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# ============================================================================
# 路径配置
# ============================================================================

PATH_CONFIG = {
    "data_dir": os.getenv("KG_DATA_DIR", _default_data_dir),
    "output_dir": _output_dir,
    "log_dir": _log_dir,
}

# ============================================================================
# 性能优化配置
# ============================================================================

PERFORMANCE_CONFIG = {
    "max_workers": 4,  # 最大工作线程数
    "chunk_queue_size": 100,  # 分块队列大小
    "entity_queue_size": 200,  # 实体队列大小
    "relation_queue_size": 300,  # 关系队列大小
    "batch_process_size": 50,  # 批处理大小
    "cache_size": 1000  # 缓存大小
}

# ============================================================================
# 质量评估配置
# ============================================================================

QUALITY_CONFIG = {
    "enable_entity_linking": True,  # 启用实体链接
    "enable_relation_validation": True,  # 启用关系验证
    "enable_knowledge_consistency": True,  # 启用知识一致性检查
    "fragmentation_threshold": 0.8,  # 碎片化阈值
    "redundancy_threshold": 0.7  # 冗余度阈值
}
