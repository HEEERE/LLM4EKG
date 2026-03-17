#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关系抽取模块
用于从应急条例文本中抽取实体间的关系
"""

import re
import json
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from utils.kg_prompt import build_extraction_prompt, parse_json_response

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Relation:
    """关系类"""
    subject: str
    predicate: str
    object: str
    subject_type: str
    object_type: str
    confidence: float = 1.0
    source_text: str = ""
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'subject_type': self.subject_type,
            'object_type': self.object_type,
            'confidence': self.confidence,
            'source_text': self.source_text,
            'properties': self.properties
        }


class RelationExtractor:
    """关系抽取器类"""
    
    def __init__(self, llm_client=None):
        """初始化关系抽取器
        
        Args:
            llm_client: LLM客户端，用于调用大模型进行关系抽取
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
        
        # 预定义关系模式和对应的谓词
        self.relation_patterns = [
            # 职责关系
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))负责([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '负责',
                'subject_type': '组织机构',
                'object_type': '行动'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))承担([\u4e00-\u9fa5、，,\s]+)职责',
                'relation_type': '承担职责',
                'subject_type': '组织机构',
                'object_type': '职责'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))承担([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '承担',
                'subject_type': '组织机构',
                'object_type': '行动'
            },
            # 管理关系
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))管理([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '管理',
                'subject_type': '组织机构',
                'object_type': '地点'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))指导([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '指导',
                'subject_type': '组织机构',
                'object_type': '组织机构'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))监督([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '监督',
                'subject_type': '组织机构',
                'object_type': '组织机构'
            },
            # 协调关系
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))协调([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '协调',
                'subject_type': '组织机构',
                'object_type': '组织机构'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))组织([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '组织',
                'subject_type': '组织机构',
                'object_type': '行动'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))领导([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '领导',
                'subject_type': '组织机构',
                'object_type': '行动'
            },
            # 参与关系
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))参与([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '参与',
                'subject_type': '组织机构',
                'object_type': '行动'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))配合([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '配合',
                'subject_type': '组织机构',
                'object_type': '组织机构'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))支持([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '支持',
                'subject_type': '组织机构',
                'object_type': '组织机构'
            },
            # 报告关系
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))报告([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '报告',
                'subject_type': '组织机构',
                'object_type': '组织机构'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))向([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))报告',
                'relation_type': '向...报告',
                'subject_type': '组织机构',
                'object_type': '组织机构'
            },
            # 处置关系
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))处置([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '处置',
                'subject_type': '组织机构',
                'object_type': '事件'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))响应([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '响应',
                'subject_type': '组织机构',
                'object_type': '事件'
            },
            # 启动关系
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))启动([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '启动',
                'subject_type': '组织机构',
                'object_type': '预案'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:预案|方案|计划))启动',
                'relation_type': '被启动',
                'subject_type': '预案',
                'object_type': '组织机构'
            },
            # 包含关系
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:预案|方案|计划))包含([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '包含',
                'subject_type': '预案',
                'object_type': '行动'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))包括([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '包括',
                'subject_type': '组织机构',
                'object_type': '组织机构'
            },
            # 属于关系
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))属于([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))',
                'relation_type': '属于',
                'subject_type': '组织机构',
                'object_type': '组织机构'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:事件|事故|灾害))属于([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '属于',
                'subject_type': '事件',
                'object_type': '事件'
            },
            # 使用关系
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))使用([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '使用',
                'subject_type': '组织机构',
                'object_type': '设备'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))采用([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '采用',
                'subject_type': '组织机构',
                'object_type': '设备'
            },
            # 需要关系
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))需要([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '需要',
                'subject_type': '组织机构',
                'object_type': '物资'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:行动|工作|任务))需要([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '需要',
                'subject_type': '行动',
                'object_type': '物资'
            },
            # 制定关系
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))制定([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '制定',
                'subject_type': '组织机构',
                'object_type': '预案'
            },
            {
                'pattern': r'([\u4e00-\u9fa5]+(?:机构|部门|单位|指挥部))编制([\u4e00-\u9fa5、，,\s]+)',
                'relation_type': '编制',
                'subject_type': '组织机构',
                'object_type': '预案'
            }
        ]
        
        logger.info("关系抽取器初始化完成")

    def _set_llm_temperature(self, temperature: float):
        if self.llm_client is None:
            return
        try:
            setattr(self.llm_client, "temperature", temperature)
        except Exception:
            return

    def _canonicalize_type(self, t: str) -> str:
        tt = (t or '').strip()
        if tt in self.allowed_types:
            return tt
        if tt in self.type_map:
            return self.type_map[tt]
        return tt
    
    def extract_by_patterns(self, text: str, entities: List[Dict[str, Any]] = None) -> List[Relation]:
        relations: List[Relation] = []
        try:
            for cfg in self.relation_patterns:
                pat = cfg.get('pattern')
                rel_type = cfg.get('relation_type')
                st = cfg.get('subject_type')
                ot = cfg.get('object_type')
                if not pat or not rel_type:
                    continue
                for m in re.finditer(pat, text):
                    s = (m.group(1) or "").strip()
                    o = ""
                    try:
                        o = (m.group(2) or "").strip()
                    except Exception:
                        o = ""
                    if not s or not o:
                        continue
                    subject_type = self._canonicalize_type(st)
                    object_type = self._canonicalize_type(ot)
                    start_pos = m.start()
                    end_pos = m.end()
                    source_text = text[max(0, start_pos - 10):min(len(text), end_pos + 10)]
                    relations.append(Relation(
                        subject=s,
                        predicate=rel_type,
                        object=o,
                        subject_type=subject_type,
                        object_type=object_type,
                        confidence=0.6,
                        source_text=source_text,
                        properties={"description": ""}
                    ))
        except Exception:
            pass
        return self._merge_and_filter_relations(relations)
    
    def extract_by_llm(self, text: str, entities: List[Dict[str, Any]] = None) -> List[Relation]:
        """基于LLM抽取关系
        
        Args:
            text: 输入文本
            entities: 已识别的实体列表（可选）
            
        Returns:
            关系列表
        """
        if not self.llm_client:
            logger.warning("LLM客户端未配置，跳过LLM抽取")
            return []
        
        try:
            try:
                original_temp = getattr(self.llm_client, "temperature")
            except Exception:
                original_temp = 0.3

            attempts = [
                {
                    "known_entities": entities,
                    "temperature": original_temp,
                    "system_message": "仅返回一个JSON对象，包含entities和relations两个键，不要输出解释或代码块。"
                },
                {
                    "known_entities": entities,
                    "temperature": 0.1,
                    "system_message": "只返回JSON对象，严格包含entities与relations键；关系字段必须完整，避免空数组。"
                },
                {
                    "known_entities": None,
                    "temperature": 0.35 if original_temp is None else max(float(original_temp), 0.35),
                    "system_message": "只返回JSON对象，严格包含entities与relations键；若原文存在任何职责、处置、响应、报送、通报、组织、制定、依据等信息，请尽量抽取关系，避免空数组。"
                },
            ]

            relations: List[Relation] = []
            for attempt in attempts:
                prompt = build_extraction_prompt(
                    text,
                    known_entities=attempt["known_entities"],
                    predicate_whitelist=None,
                )
                self._set_llm_temperature(float(attempt["temperature"]))
                response = self.llm_client.invoke(prompt, system_message=attempt["system_message"])
                result_text = response.content if hasattr(response, 'content') else str(response)
                result_json = parse_json_response(result_text)

                entity_type_map = {}
                if entities:
                    for e in entities:
                        name = e.get('name')
                        etype = e.get('type')
                        if name:
                            entity_type_map[name] = etype

                current: List[Relation] = []
                for relation_data in result_json.get('relations', []):
                    subject = relation_data.get('subject', '').strip()
                    predicate = relation_data.get('predicate', '').strip()
                    obj = relation_data.get('object', '').strip()
                    subject_type = relation_data.get('subject_type') or entity_type_map.get(subject, '未知')
                    object_type = relation_data.get('object_type') or entity_type_map.get(obj, '未知')
                    subject_type = self._canonicalize_type(subject_type)
                    object_type = self._canonicalize_type(object_type)
                    confidence = float(relation_data.get('confidence', 0.8))
                    source_text = relation_data.get('source_text', '')

                    current.append(Relation(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        subject_type=subject_type,
                        object_type=object_type,
                        confidence=confidence,
                        source_text=source_text,
                        properties={"description": relation_data.get('description', '')}
                    ))

                relations = current
                if len(relations) > 0:
                    break

            self._set_llm_temperature(original_temp if original_temp is not None else 0.3)
            logger.info(f"LLM抽取完成，共抽取 {len(relations)} 个关系")
            return relations

        except Exception as e:
            logger.error(f"LLM关系抽取失败: {str(e)}")
            return []
    
    def extract_relations(self, text: str, entities: List[Dict[str, Any]] = None, 
                         use_llm: bool = True) -> List[Relation]:
        """综合关系抽取
        
        Args:
            text: 输入文本
            entities: 已识别的实体列表（可选）
            use_llm: 是否使用LLM抽取
            
        Returns:
            关系列表
        """
        logger.info(f"开始抽取文本中的关系，文本长度: {len(text)}")
        
        llm_relations = self.extract_by_llm(text, entities) if use_llm else []
        pattern_relations = self.extract_by_patterns(text, entities)
        combined = []
        if llm_relations:
            combined.extend(llm_relations)
        if pattern_relations:
            combined.extend(pattern_relations)
        final_relations = self._merge_and_filter_relations(combined)
        
        logger.info(f"关系抽取完成，共抽取 {len(final_relations)} 个关系")
        return final_relations
    
    def _remove_duplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """去重关系"""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # 使用主体-谓词-客体作为唯一标识
            relation_key = (relation.subject, relation.predicate, relation.object)
            if relation_key not in seen:
                seen.add(relation_key)
                unique_relations.append(relation)
        
        return unique_relations
    
    def _merge_and_filter_relations(self, relations: List[Relation]) -> List[Relation]:
        """合并和过滤关系"""
        # 去重
        unique_relations = self._remove_duplicate_relations(relations)
        
        # 按置信度排序
        unique_relations.sort(key=lambda x: x.confidence, reverse=True)
        
        # 过滤置信度较低的关系
        filtered_relations = [
            relation for relation in unique_relations 
            if relation.confidence >= 0.5
        ]
        
        return filtered_relations
    
    def batch_extract_relations(self, texts: List[str], 
                              entities_list: List[List[Dict[str, Any]]] = None,
                              use_llm: bool = True, 
                              max_workers: int = 4) -> List[List[Relation]]:
        """批量抽取关系
        
        Args:
            texts: 文本列表
            entities_list: 实体列表的列表（可选）
            use_llm: 是否使用LLM抽取
            max_workers: 最大工作线程数
            
        Returns:
            关系列表的列表
        """
        logger.info(f"开始批量抽取 {len(texts)} 个文本的关系")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, text in enumerate(texts):
                entities = entities_list[i] if entities_list and i < len(entities_list) else None
                future = executor.submit(self.extract_relations, text, entities, use_llm)
                futures.append(future)
            
            results = []
            for future in futures:
                try:
                    relations = future.result(timeout=60)  # 60秒超时
                    results.append(relations)
                except Exception as e:
                    logger.error(f"批量抽取关系失败: {str(e)}")
                    results.append([])
        
        logger.info(f"批量关系抽取完成")
        return results
