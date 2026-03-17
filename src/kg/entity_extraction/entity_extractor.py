#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体识别模块
用于从应急条例文本中识别和抽取实体
"""

import re
import json
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time
from utils.kg_prompt import build_extraction_prompt, parse_json_response

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """实体类"""
    name: str
    entity_type: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    source_text: str = ""
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.entity_type,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'confidence': self.confidence,
            'source_text': self.source_text,
            'properties': self.properties
        }


class EntityExtractor:
    """实体抽取器类"""
    
    def __init__(self, llm_client=None):
        """初始化实体抽取器
        
        Args:
            llm_client: LLM客户端，用于调用大模型进行实体抽取
        """
        self.llm_client = llm_client
        self.allowed_types = {
            '组织机构', '人员', '地点', '物资', '事件', '行动', '法规', '时间'
        }
        self.type_map = {
            '设备': '物资',
            '预案': '法规',
            '职责': '行动'
        }
        
        # 预定义实体类型和模式
        self.entity_patterns = {
            '组织机构': [
                r'[\u4e00-\u9fa5]+(?:局|厅|部|委|办|中心|指挥部|领导小组|工作组|委员会|公司|企业|单位)',
                r'[\u4e00-\u9fa5]+(?:人民政府|政府|街道办|村委会|居委会)',
                r'国家[\u4e00-\u9fa5]+(?:局|委|部|办)',
                r'[\u4e00-\u9fa5]+省[\u4e00-\u9fa5]+(?:厅|局)',
                r'[\u4e00-\u9fa5]+市[\u4e00-\u9fa5]+(?:局|委|办)'
            ],
            '人员': [
                r'[\u4e00-\u9fa5]+(?:长|主任|书记|局长|厅长|部长|委员|指挥|领导|负责人|专家|人员|工作人员)',
                r'(?:总指挥|副总指挥|组长|副组长|成员|联络员|信息员)',
                r'[\u4e00-\u9fa5]+同志'
            ],
            '地点': [
                r'[\u4e00-\u9fa5]+(?:省|市|县|区|镇|乡|村|街道|社区)',
                r'[\u4e00-\u9fa5]+(?:山|河|湖|海|江|水库|大坝|桥梁|隧道)',
                r'[\u4e00-\u9fa5]+(?:现场|区域|地段|路段|水域|空域)',
                r'[A-Z][A-Z][A-Z][A-Z]\d+'  # 行政区划代码
            ],
            '设备': [
                r'[\u4e00-\u9fa5]+(?:设备|装置|仪器|仪表|系统|平台|车辆|船舶|飞机|直升机)',
                r'(?:通信|监测|监控|救援|应急|指挥|调度)[\u4e00-\u9fa5]+(?:设备|系统)',
                r'[\u4e00-\u9fa5]+(?:卫星|雷达|传感器|摄像头|无人机)'
            ],
            '物资': [
                r'[\u4e00-\u9fa5]+(?:物资|材料|药品|食品|水|燃料|器材|装备)',
                r'(?:应急|救援|救灾|医疗|生活)[\u4e00-\u9fa5]+(?:物资|用品)',
                r'[\u4e00-\u9fa5]+(?:储备|库存|仓库|基地)'
            ],
            '事件': [
                r'[\u4e00-\u9fa5]+(?:事故|灾害|事件|险情|预警|警报|紧急状态)',
                r'(?:地震|洪水|台风|暴雨|雪灾|火灾|爆炸|泄漏|坍塌|滑坡|泥石流)',
                r'[\u4e00-\u9fa5]+(?:级响应|级预警|级警报)',
                r'(?:特别重大|重大|较大|一般)[\u4e00-\u9fa5]+(?:事故|灾害|事件)'
            ],
            '行动': [
                r'[\u4e00-\u9fa5]+(?:救援|抢险|救灾|疏散|转移|安置|抢修|维护|保障)',
                r'(?:应急|抢险|救援|救灾)[\u4e00-\u9fa5]+(?:行动|工作|任务)',
                r'[\u4e00-\u9fa5]+(?:演练|演习|培训|宣传|教育)',
                r'[\u4e00-\u9fa5]+(?:调查|评估|分析|总结|报告)'
            ],
            '职责': [
                r'[\u4e00-\u9fa5]+(?:职责|责任|义务|任务|工作|职能)',
                r'(?:主要|重要|基本|法定)[\u4e00-\u9fa5]+(?:职责|责任)',
                r'[\u4e00-\u9fa5]+(?:工作|任务|职能)[\u4e00-\u9fa5]+(?:职责|责任)'
            ],
            '预案': [
                r'[\u4e00-\u9fa5]+(?:预案|方案|计划|规划|策略)',
                r'(?:应急|救援|抢险|救灾)[\u4e00-\u9fa5]+(?:预案|方案)',
                r'[\u4e00-\u9fa5]+(?:专项|部门|总体|综合)[\u4e00-\u9fa5]+(?:预案|方案)',
                r'[\u4e00-\u9fa5]+(?:修订|制定|编制|完善)[\u4e00-\u9fa5]+(?:预案|方案)'
            ],
            '法规': [
                r'《[\u4e00-\u9fa5\s\d]+》',  # 书名号内的内容
                r'[\u4e00-\u9fa5]+(?:法|条例|规定|办法|细则|规范|标准|制度)',
                r'(?:国家|地方|行业)[\u4e00-\u9fa5]+(?:标准|规范|制度)',
                r'[\u4e00-\u9fa5]+(?:法律|法规|规章|规范性文件)'
            ],
            '时间': [
                r'\d{4}年\d{1,2}月\d{1,2}日',
                r'\d{1,2}月\d{1,2}日',
                r'(?:立即|马上|及时|迅速|第一时间)',
                r'(?:小时|分钟|秒|天|周|月|年)内',
                r'\d+(?:小时|分钟|秒|天|周|月|年)'
            ]
        }
        
        # 停用词列表
        self.stop_words = {'的', '了', '和', '与', '或', '在', '是', '有', '对', '等', '及', '由', '为', '将', '应', '要', '可', '能', '会', '需', '须'}
        
        
        logger.info("实体抽取器初始化完成")

    def _set_llm_temperature(self, temperature: float):
        if self.llm_client is None:
            return
        try:
            setattr(self.llm_client, "temperature", temperature)
        except Exception:
            return

    def _canonicalize_type(self, etype: str, name: str = "") -> str:
        t = (etype or '').strip()
        if t in self.allowed_types:
            return t
        if t in self.type_map:
            return self.type_map[t]
        # 简单基于名称的兜底映射
        if '预案' in name or '方案' in name or '制度' in name or '条例' in name or '办法' in name:
            return '法规'
        if any(k in name for k in ['设备', '系统', '器材', '装备']):
            return '物资'
        return t
    
    def extract_by_patterns(self, text: str) -> List[Entity]:
        entities: List[Entity] = []
        for etype, patterns in self.entity_patterns.items():
            for pat in patterns:
                try:
                    for m in re.finditer(pat, text):
                        name = m.group(0)
                        start_pos = m.start()
                        end_pos = m.end()
                        source_text = text[max(0, start_pos - 10):min(len(text), end_pos + 10)]
                        canon_type = self._canonicalize_type(etype, name)
                        entities.append(
                            Entity(
                                name=name,
                                entity_type=canon_type,
                                start_pos=start_pos,
                                end_pos=end_pos,
                                confidence=0.6,
                                source_text=source_text,
                                properties={"description": "", "aliases": []}
                            )
                        )
                except Exception:
                    continue
        return self._merge_and_filter_entities(entities)
    
    def extract_by_llm(self, text: str) -> List[Entity]:
        """基于LLM抽取实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        if not self.llm_client:
            logger.warning("LLM客户端未配置，跳过LLM抽取")
            return []
        
        try:
            prompt = build_extraction_prompt(text, predicate_whitelist=None)
            try:
                original_temp = getattr(self.llm_client, "temperature")
            except Exception:
                original_temp = 0.3

            attempts = [
                {
                    "temperature": original_temp,
                    "system_message": "仅返回一个JSON对象，包含entities和relations两个键，不要输出解释或代码块。"
                },
                {
                    "temperature": 0.1,
                    "system_message": "只返回JSON对象，严格包含entities与relations键；实体字段必须完整，避免空数组。"
                },
                {
                    "temperature": 0.35 if original_temp is None else max(float(original_temp), 0.35),
                    "system_message": "只返回JSON对象，严格包含entities与relations键；若原文包含组织机构、人员、地点、物资、事件、行动、法规、时间等信息，请尽量抽取，避免空数组。"
                },
            ]

            entities: List[Entity] = []
            for attempt in attempts:
                self._set_llm_temperature(float(attempt["temperature"]))
                response = self.llm_client.invoke(prompt, system_message=attempt["system_message"])
                result_text = response.content if hasattr(response, 'content') else str(response)
                result_json = parse_json_response(result_text)

                current: List[Entity] = []
                for entity_data in result_json.get('entities', []):
                    name = entity_data.get('name', '').strip()
                    etype = entity_data.get('type', '').strip() or '未知'
                    etype = self._canonicalize_type(etype, name)
                    confidence = float(entity_data.get('confidence', 0.8))

                    start_pos = text.find(name) if name else -1
                    end_pos = (start_pos + len(name)) if start_pos >= 0 else -1
                    source_text = text[max(0, start_pos - 10):min(len(text), end_pos + 10)] if start_pos >= 0 else ''
                    aliases = entity_data.get('aliases') or []

                    current.append(Entity(
                        name=name,
                        entity_type=etype,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        confidence=confidence,
                        source_text=source_text,
                        properties={
                            "description": entity_data.get('description', ''),
                            "aliases": aliases
                        }
                    ))

                entities = current
                if len(entities) > 0:
                    break

            self._set_llm_temperature(original_temp if original_temp is not None else 0.3)
            logger.info(f"LLM抽取完成，共抽取 {len(entities)} 个实体")
            return entities

        except Exception as e:
            logger.error(f"LLM实体抽取失败: {str(e)}")
            return []
    
    def extract_entities(self, text: str, use_llm: bool = True) -> List[Entity]:
        """综合实体抽取（LLM与模式双通道融合）
        
        Args:
            text: 输入文本
            use_llm: 是否使用LLM抽取
            
        Returns:
            实体列表
        """
        logger.info(f"开始抽取文本中的实体，文本长度: {len(text)}")
        
        llm_entities = self.extract_by_llm(text) if use_llm else []
        pattern_entities = self.extract_by_patterns(text)
        combined: List[Entity] = []
        if llm_entities:
            combined.extend(llm_entities)
        if pattern_entities:
            combined.extend(pattern_entities)
        final_entities = self._merge_and_filter_entities(combined)
        
        logger.info(f"实体抽取完成，共抽取 {len(final_entities)} 个实体")
        return final_entities
    
    def _remove_duplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """按位置去重实体"""
        seen_positions = set()
        unique_entities = []
        
        for entity in entities:
            position_key = (entity.start_pos, entity.end_pos)
            if position_key not in seen_positions:
                seen_positions.add(position_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _merge_and_filter_entities(self, entities: List[Entity]) -> List[Entity]:
        """合并和过滤实体"""
        # 按位置排序
        entities.sort(key=lambda x: (x.start_pos, x.end_pos))
        
        # 合并重叠实体
        merged_entities = []
        for entity in entities:
            # 检查是否与已有实体重叠
            is_overlapped = False
            for merged in merged_entities:
                # 如果重叠，选择置信度更高的
                if (entity.start_pos < merged.end_pos and entity.end_pos > merged.start_pos):
                    if entity.confidence > merged.confidence:
                        merged_entities.remove(merged)
                        merged_entities.append(entity)
                    is_overlapped = True
                    break
            
            if not is_overlapped:
                merged_entities.append(entity)
        
        # 按置信度排序
        merged_entities.sort(key=lambda x: x.confidence, reverse=True)
        
        return merged_entities
    
    def batch_extract_entities(self, texts: List[str], use_llm: bool = True, 
                             max_workers: int = 4) -> List[List[Entity]]:
        """批量抽取实体
        
        Args:
            texts: 文本列表
            use_llm: 是否使用LLM抽取
            max_workers: 最大工作线程数
            
        Returns:
            实体列表的列表
        """
        logger.info(f"开始批量抽取 {len(texts)} 个文本的实体")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for text in texts:
                future = executor.submit(self.extract_entities, text, use_llm)
                futures.append(future)
            
            results = []
            for future in futures:
                try:
                    entities = future.result(timeout=30)  # 30秒超时
                    results.append(entities)
                except Exception as e:
                    logger.error(f"批量抽取实体失败: {str(e)}")
                    results.append([])
        
        logger.info(f"批量实体抽取完成")
        return results
