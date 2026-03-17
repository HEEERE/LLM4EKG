#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应急条例知识图谱构建主程序
整合实体识别、关系抽取、图数据库等模块，构建完整的知识图谱
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from pathlib import Path
import math
import glob
import argparse

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入各模块
from config.config import (
    API_CONFIG, DOCUMENT_CONFIG, PATH_CONFIG, 
    LOGGING_CONFIG, PERFORMANCE_CONFIG, QUALITY_CONFIG,
    BRIDGE_SCHEMA_PREDICATES, NLI_CONFIG
)
from entity_extraction.entity_extractor import EntityExtractor, Entity
from relation_extraction.relation_extractor import RelationExtractor, Relation
from graph_db.graph_database import GraphDatabaseManager, KGEntity, KGRelation
from utils.utils import setup_logging, load_json_data, save_json_data, chunk_text, batch_process
from utils.llm_client import LLMClient, LLMClientFactory
from utils.kg_prompt import build_consolidation_prompt, parse_json_response


class EmergencyRegulationKGBuilder:
    """应急条例知识图谱构建器"""
    
    def __init__(self):
        """初始化构建器"""
        # 设置日志
        self.logger = setup_logging(
            log_file=os.path.join(PATH_CONFIG["log_dir"], "knowledge_graph.log"),
            level=LOGGING_CONFIG["level"],
            format_string=LOGGING_CONFIG["format"]
        )
        
        self.logger.info("开始初始化应急条例知识图谱构建器...")
        
        # 初始化LLM客户端（使用阿里云百炼deepseek-v3）
        self.llm_client = LLMClientFactory.create_client(
            provider="aliyun",
            api_key=API_CONFIG["aliyun_api_key"],
            model=API_CONFIG["aliyun_model"],
            max_tokens=API_CONFIG["max_tokens"],
            temperature=API_CONFIG["temperature"]
        )
        
        # 初始化各模块
        self.entity_extractor = EntityExtractor(llm_client=self.llm_client)
        self.relation_extractor = RelationExtractor(llm_client=self.llm_client)
        self.graph_manager = GraphDatabaseManager()
        
        # 创建数据库约束
        self.graph_manager.create_constraints()
        
        # 处理统计
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_entities": 0,
            "total_relations": 0,
            "processing_time": 0,
            "error_count": 0
        }
        
        self.logger.info("应急条例知识图谱构建器初始化完成")
        self.root_node_name = None
        self._nli_pipe = None
    
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            数据列表
        """
        self.logger.info(f"开始加载数据: {data_path}")
        
        try:
            records: List[Dict[str, Any]] = []
            if ("*" in data_path) or ("?" in data_path):
                paths = sorted(glob.glob(data_path))
                for p in paths:
                    try:
                        ds = load_json_data(p)
                        src = Path(p).stem
                        for item in ds:
                            sn = (item.get("rule_source") or src)
                            item["source_name"] = sn
                            records.append(item)
                    except Exception as e:
                        self.logger.warning(f"加载文件失败: {p} - {str(e)}")
            else:
                ds = load_json_data(data_path)
                src = Path(data_path).stem
                for item in ds:
                    sn = (item.get("rule_source") or src)
                    item["source_name"] = sn
                    records.append(item)
            self.stats["total_documents"] = len(records)
            self.logger.info(f"成功加载 {len(records)} 条数据")
            return records
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            raise
    
    def index_rule_embeddings(self, data_path: str, output_dir: str = None, dimensions: int = 512) -> str:
        out_dir = output_dir or os.path.join(PATH_CONFIG["output_dir"], "embeddings")
        os.makedirs(out_dir, exist_ok=True)
        data = self.load_data(data_path)
        texts = []
        metas = []
        for item in data:
            rt = (item.get("rule_text") or "").strip()
            if not rt:
                continue
            texts.append(rt)
            metas.append({
                "rule_id": item.get("rule_id"),
                "rule_source": item.get("rule_source"),
                "rule_tags": item.get("rule_tags", [])
            })
        zhipu_client = LLMClientFactory.create_client(
            provider="zhipu",
            api_key=API_CONFIG.get("zhipu_api_key", ""),
            model="embedding-3",
            timeout=30,
            max_retries=3
        )
        records = []
        offset = 0
        for batch in batch_process(texts, batch_size=64):
            vecs = zhipu_client.embed(batch, dimensions=dimensions)
            for i, v in enumerate(vecs):
                meta = metas[offset + i]
                records.append({
                    "rule_id": meta.get("rule_id"),
                    "rule_text": batch[i],
                    "embedding": v,
                    "model": "embedding-3",
                    "dim": len(v),
                    "norm_l2": math.sqrt(sum(x*x for x in v)) if v else 0.0,
                    "source": {
                        "rule_source": meta.get("rule_source"),
                        "rule_tags": meta.get("rule_tags")
                    }
                })
            offset += len(batch)
        out_path = os.path.join(out_dir, "rule_embeddings.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return out_path

    def _group_documents_by_rule_count(self, documents: List[Dict[str, Any]], group_size: int = 20) -> List[Dict[str, Any]]:
        """按规则条目分组，将每 group_size 条规则合并为一个处理单元（单块）。

        Args:
            documents: 原始规则文档列表，每项通常包含 rule_id, rule_text
            group_size: 分组大小，默认 20 条一组

        Returns:
            分组后的文档列表，每项包含合并后的文本与成员信息
        """
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for doc in documents:
            src = (doc.get("source_name") or doc.get("rule_source") or "未知")
            buckets.setdefault(src, []).append(doc)
        grouped: List[Dict[str, Any]] = []
        for src, docs in buckets.items():
            total = len(docs)
            for start in range(0, total, group_size):
                group = docs[start:start + group_size]
                members = []
                texts = []
                for idx, doc in enumerate(group):
                    rid = doc.get("rule_id", f"doc_{start + idx}")
                    members.append(rid)
                    rt = (doc.get("rule_text", "") or "").strip()
                    texts.append(f"【{rid}】{rt}")
                group_text = "\n".join(texts)
                grouped.append({
                    "rule_id": f"group_{src}_{start + 1}_{start + len(group)}",
                    "rule_text": group_text,
                    "is_group": True,
                    "group_members": members,
                    "source_name": src
                })
        return grouped
    
    def process_document(self, document: Dict[str, Any], doc_index: int) -> Dict[str, Any]:
        """处理单个文档
        
        Args:
            document: 文档数据
            doc_index: 文档索引
            
        Returns:
            处理结果
        """
        self.logger.info(f"开始处理文档 {doc_index + 1}")
        
        result = {
            "document_id": document.get("rule_id", f"doc_{doc_index}"),
            "document_text": document.get("rule_text", ""),
            "chunks": [],
            "entities": [],
            "relations": [],
            "processing_time": 0,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            # 1. 文本分块
            text = result["document_text"]
            if not text:
                raise ValueError("文档文本为空")
            
            # 分块规则：若为分组文档，则整组作为一个单块；否则使用默认按句子+重叠的分块
            if document.get("is_group"):
                chunks = [text]
            else:
                chunks = chunk_text(
                    text, 
                    chunk_size=DOCUMENT_CONFIG["chunk_token_num"],
                    overlap=DOCUMENT_CONFIG["overlap_size"]
                )
            result["chunks"] = chunks
            self.stats["total_chunks"] += len(chunks)
            
            self.logger.info(f"文档 {doc_index + 1} 分块完成，共 {len(chunks)} 个块")
            
            # 2. 实体抽取
            all_entities = []
            source_name = document.get("source_name") or document.get("rule_source") or self.root_node_name
            self.root_node_name = source_name
            for chunk_index, chunk in enumerate(chunks):
                entities = self.entity_extractor.extract_entities(
                    chunk, 
                    use_llm=True
                )
                
                # 添加块信息到实体
                for entity in entities:
                    entity.properties = entity.properties or {}
                    entity.properties["chunk_index"] = chunk_index
                    entity.properties["document_id"] = result["document_id"]
                    entity.properties["source_name"] = source_name
                
                all_entities.extend(entities)
            
            # 实体去重
            unique_entities = self._deduplicate_entities(all_entities)
            result["entities"] = [entity.to_dict() for entity in unique_entities]
            self.stats["total_entities"] += len(unique_entities)
            
            self.logger.info(f"文档 {doc_index + 1} 实体抽取完成，共 {len(unique_entities)} 个实体")
            
            # 3. 关系抽取
            all_relations = []
            for chunk_index, chunk in enumerate(chunks):
                # 获取当前块的实体
                chunk_entities = [
                    entity for entity in unique_entities 
                    if entity.properties.get("chunk_index") == chunk_index
                ]
                
                relations = self.relation_extractor.extract_relations(
                    chunk, 
                    entities=[entity.to_dict() for entity in chunk_entities],
                    use_llm=True
                )
                
                # 添加块信息到关系
                for relation in relations:
                    relation.properties = relation.properties or {}
                    relation.properties["chunk_index"] = chunk_index
                    relation.properties["document_id"] = result["document_id"]
                    relation.properties["source_name"] = source_name
                
                all_relations.extend(relations)
            
            # 关系去重（初始）
            unique_relations = self._deduplicate_relations(all_relations)

            # 3.5 文档级整合：别名合并 + 桥接边补充
            try:
                # 识别当前文档中的孤立实体（尚未参与任何关系）
                isolated_names = self._find_isolated_entity_names(unique_entities, unique_relations)
                consolidation_prompt = build_consolidation_prompt(
                    result["document_text"],
                    entities=[e.to_dict() for e in unique_entities],
                    relations=[r.to_dict() for r in unique_relations],
                    isolated_entities=isolated_names,
                    predicate_whitelist=BRIDGE_SCHEMA_PREDICATES,
                )
                cons_resp = self.llm_client.invoke(consolidation_prompt, system_message="你是知识图谱整合专家。")
                cons_text = cons_resp.content if hasattr(cons_resp, 'content') else str(cons_resp)
                cons_json = parse_json_response(cons_text)

                alias_map: Dict[str, str] = cons_json.get("alias_map", {}) or {}
                vector_alias_map = _build_alias_map_by_vectors(unique_entities, dimensions=512, threshold=0.86)
                alias_map.update(vector_alias_map)

                # 应用别名映射到实体
                unique_entities = self._apply_alias_map(unique_entities, alias_map)
                # 合并规范实体的别名与描述
                unique_entities = self._merge_canonical_entities(unique_entities, cons_json.get("canonical_entities", []) or [], source_name)
                # 将关系中的主体与客体按别名映射到规范名称
                unique_relations = self._remap_relations_by_alias(unique_relations, alias_map)

                # 桥接边加入
                bridge_relations = self._build_relations_from_json(cons_json.get("bridge_relations", []), result["document_id"], -1, source_name)
                unique_relations.extend(bridge_relations)
                unique_relations = self._deduplicate_relations(unique_relations)
            except Exception as e:
                self.logger.warning(f"文档级整合失败，继续流程：{str(e)}")

            

            # 更新结果统计
            result["entities"] = [entity.to_dict() for entity in unique_entities]
            result["relations"] = [relation.to_dict() for relation in unique_relations]
            self.stats["total_relations"] += len(unique_relations)

            self.logger.info(f"文档 {doc_index + 1} 关系抽取完成（含整合与统计边），共 {len(unique_relations)} 个关系")

            # 4. 存储到图数据库
            self._store_to_graph(unique_entities, unique_relations)
            try:
                _write_entity_embeddings(result["document_id"], unique_entities, dimensions=512)
            except Exception as e:
                self.logger.warning(f"实体向量写入失败: {str(e)}")
            
            # 计算处理时间
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            self.logger.info(f"文档 {doc_index + 1} 处理完成，耗时 {processing_time:.2f} 秒")
            
        except Exception as e:
            error_msg = f"处理文档 {doc_index + 1} 失败: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            result["error"] = str(e)
            self.stats["error_count"] += 1
        
        return result
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """实体去重
        
        Args:
            entities: 实体列表
            
        Returns:
            去重后的实体列表
        """
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # 使用名称和类型作为唯一标识
            entity_key = (entity.name, entity.entity_type)
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """关系去重
        
        Args:
            relations: 关系列表
            
        Returns:
            去重后的关系列表
        """
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # 使用主体-谓词-客体作为唯一标识
            relation_key = (relation.subject, relation.predicate, relation.object)
            if relation_key not in seen:
                seen.add(relation_key)
                unique_relations.append(relation)
        
        return unique_relations
    
    def _store_to_graph(self, entities: List[Entity], relations: List[Relation]):
        """存储到图数据库
        
        Args:
            entities: 实体列表
            relations: 关系列表
        """
        try:
            # 转换实体格式
            kg_entities = []
            for entity in entities:
                kg_entity = KGEntity(
                    name=entity.name,
                    entity_type=entity.entity_type,
                    properties=entity.properties or {}
                )
                kg_entities.append(kg_entity)
            
            # 批量创建实体
            if kg_entities:
                entity_count = self.graph_manager.batch_create_entities(kg_entities)
                self.logger.info(f"成功创建 {entity_count} 个实体到图数据库")
            
            # 转换关系格式
            kg_relations = []
            for relation in relations:
                kg_relation = KGRelation(
                    subject=relation.subject,
                    predicate=relation.predicate,
                    object=relation.object,
                    properties=relation.properties or {}
                )
                kg_relations.append(kg_relation)
            type_link_relations = []
            for entity in entities:
                type_link_relations.append(KGRelation(
                    subject=entity.name,
                    predicate="属于类型",
                    object=entity.entity_type,
                    properties={
                        "is_meta": True,
                        "subject_label": entity.entity_type,
                        "object_label": "实体类型",
                        "document_id": entity.properties.get("document_id"),
                        "chunk_index": entity.properties.get("chunk_index", -1),
                        "source_name": self.root_node_name
                    }
                ))
            kg_relations.extend(type_link_relations)
            
            # 批量创建关系
            if kg_relations:
                relation_count = self.graph_manager.batch_create_relations(kg_relations)
                self.logger.info(f"成功创建 {relation_count} 个关系到图数据库")
                
        except Exception as e:
            self.logger.error(f"存储到图数据库失败: {str(e)}")
            raise

    def _apply_alias_map(self, entities: List[Entity], alias_map: Dict[str, str]) -> List[Entity]:
        """应用别名映射到实体列表，并汇总别名到properties.aliases。
        别名映射为 {alias: canonical}；若别名实体存在，则重命名为规范名，并将别名并入规范实体的aliases。
        """
        if not alias_map:
            return entities

        # 规范名到实体的索引
        name_index: Dict[Tuple[str, str], Entity] = {}
        for e in entities:
            name_index[(e.name, e.entity_type)] = e

        for e in entities:
            canonical = alias_map.get(e.name)
            if canonical and canonical != e.name:
                # 重命名
                old_name = e.name
                e.name = canonical
                # 将别名合并到规范实体
                key = (e.name, e.entity_type)
                if key in name_index and name_index[key] is not e:
                    # 目标规范实体
                    target = name_index[key]
                    aliases = set(target.properties.get("aliases", []) or [])
                    aliases.add(old_name)
                    # 合并描述（保持已有描述为主）
                    if e.properties.get("description") and not target.properties.get("description"):
                        target.properties["description"] = e.properties.get("description")
                    target.properties["aliases"] = list(aliases)
                else:
                    # 当前实体自身成为规范实体，记录别名
                    aliases = set(e.properties.get("aliases", []) or [])
                    aliases.add(old_name)
                    e.properties["aliases"] = list(aliases)

        # 别名合并后再去重
        return self._deduplicate_entities(entities)

    def _get_nli_pipe(self):
        if not (NLI_CONFIG or {}).get("enabled", True):
            return None
        if self._nli_pipe is not None:
            return self._nli_pipe
        cache_dir = (NLI_CONFIG or {}).get("cache_dir")
        if cache_dir:
            try:
                os.makedirs(str(cache_dir), exist_ok=True)
                os.environ.setdefault("HF_HOME", str(cache_dir))
                os.environ.setdefault("HF_HUB_CACHE", str(cache_dir))
                os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))
            except Exception:
                pass
        try:
            from transformers import pipeline  # type: ignore
        except Exception:
            self._nli_pipe = None
            return None
        model_name = (NLI_CONFIG or {}).get("model_name") or "microsoft/deberta-v3-large-mnli"
        local_dir = (NLI_CONFIG or {}).get("local_dir")
        model_ref = str(local_dir) if local_dir and os.path.isdir(str(local_dir)) else model_name
        device = int((NLI_CONFIG or {}).get("device", -1))
        try:
            self._nli_pipe = pipeline(
                "text-classification",
                model=model_ref,
                tokenizer=model_ref,
                device=device,
                truncation=True,
            )
        except Exception:
            self._nli_pipe = None
        return self._nli_pipe

    def _entailment_score(self, premise: str, hypothesis: str) -> float:
        premise = (premise or "").strip()
        hypothesis = (hypothesis or "").strip()
        if not premise or not hypothesis:
            return 0.0
        nli = self._get_nli_pipe()
        if nli is None:
            return 0.0
        payload = {"text": premise, "text_pair": hypothesis}
        out = None
        try:
            out = nli(payload, top_k=None)
        except Exception:
            try:
                out = nli(payload, return_all_scores=True)
            except Exception:
                try:
                    out = nli(premise, hypothesis, top_k=None)
                except Exception:
                    try:
                        out = nli(premise, hypothesis)
                    except Exception:
                        out = None
        if out is None:
            return 0.0
        if isinstance(out, list) and len(out) == 1 and isinstance(out[0], list):
            out = out[0]
        if not isinstance(out, list):
            return 0.0
        id2label = {}
        try:
            id2label = dict(getattr(getattr(nli, "model", None), "config", None).id2label or {})
        except Exception:
            id2label = {}
        best = 0.0
        for item in out:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "")
            score = float(item.get("score") or 0.0)
            mapped = id2label.get(int(label.split("_")[-1])) if label.startswith("LABEL_") and label.split("_")[-1].isdigit() else None
            lab = (mapped or label).lower()
            if "entail" in lab:
                if score > best:
                    best = score
        return float(best)

    def _build_relations_from_json(self, rel_list: List[Dict[str, Any]], document_id: str, chunk_index: int, source_name: str) -> List[Relation]:
        """从JSON关系数组构建 Relation 对象，附加文档与分块属性，并按置信度与证据过滤桥接边。"""
        relations: List[Relation] = []
        schema = set(BRIDGE_SCHEMA_PREDICATES or [])
        entail_threshold = float((NLI_CONFIG or {}).get("threshold", 0.85))
        for rel in rel_list or []:
            try:
                subject = (rel.get("subject") or "").strip()
                predicate = (rel.get("predicate") or "").strip()
                obj = (rel.get("object") or "").strip()
                subject_type = rel.get("subject_type") or "未知"
                object_type = rel.get("object_type") or "未知"
                confidence = float(rel.get("confidence", 0.7))
                s_evidence = (rel.get("s_evidence") or rel.get("source_text") or "").strip()
                source_rule_ids = rel.get("source_rule_ids") or []
                is_llm_generated = bool(rel.get("is_llm_generated", True))

                if not subject or not predicate or not obj:
                    continue
                if confidence < 0.85:
                    continue
                if not s_evidence:
                    continue
                if predicate not in schema:
                    score = self._entailment_score(s_evidence, f"根据上述文本，{subject}{predicate}{obj}。")
                    if score < entail_threshold:
                        continue

                relation = Relation(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    subject_type=subject_type,
                    object_type=object_type,
                    confidence=confidence,
                    source_text=s_evidence,
                    properties={
                        "description": rel.get("description", ""),
                        "document_id": document_id,
                        "chunk_index": chunk_index,
                        "is_bridge": True,
                        "is_llm_generated": is_llm_generated,
                        "source_rule_ids": source_rule_ids,
                        "source_name": source_name,
                        "s_evidence": s_evidence,
                        "predicate_is_schema": (predicate in schema),
                        "entailment_threshold": entail_threshold,
                        "entailment_score": (None if predicate in schema else score),
                        "entailment_model": (None if predicate in schema else (NLI_CONFIG or {}).get("model_name")),
                    }
                )
                relations.append(relation)
            except Exception:
                continue
        return relations

    def _find_isolated_entity_names(self, entities: List[Entity], relations: List[Relation]) -> List[str]:
        involved: set = set()
        for r in relations:
            if r.subject:
                involved.add(r.subject)
            if r.object:
                involved.add(r.object)
        names = []
        for e in entities:
            if e.name not in involved:
                names.append(e.name)
        return names

    def _merge_canonical_entities(self, entities: List[Entity], canonical_entities: List[Dict[str, Any]], source_name: str) -> List[Entity]:
        index: Dict[Tuple[str, str], Entity] = {}
        for e in entities:
            index[(e.name, e.entity_type)] = e
        for c in canonical_entities or []:
            name = (c.get("name") or "").strip()
            etype = (c.get("type") or "").strip() or "未知"
            key = (name, etype)
            aliases = c.get("aliases") or []
            desc = c.get("description") or ""
            if key in index:
                target = index[key]
                a = set(target.properties.get("aliases", []) or [])
                for al in aliases:
                    if al:
                        a.add(al)
                if desc and not target.properties.get("description"):
                    target.properties["description"] = desc
                target.properties["aliases"] = list(a)
                if not target.properties.get("source_name"):
                    target.properties["source_name"] = source_name
            else:
                entities.append(Entity(name=name, entity_type=etype, start_pos=-1, end_pos=-1, confidence=0.8, source_text="", properties={"description": desc, "aliases": aliases, "source_name": source_name}))
        return self._deduplicate_entities(entities)

    def _remap_relations_by_alias(self, relations: List[Relation], alias_map: Dict[str, str]) -> List[Relation]:
        if not alias_map:
            return relations
        remapped: List[Relation] = []
        for r in relations:
            s = alias_map.get(r.subject, r.subject)
            o = alias_map.get(r.object, r.object)
            remapped.append(Relation(subject=s, predicate=r.predicate, object=o, subject_type=r.subject_type, object_type=r.object_type, confidence=r.confidence, source_text=r.source_text, properties=r.properties))
        return self._deduplicate_relations(remapped)

    def _ensure_meta_nodes(self, source_name: str):
        self.root_node_name = source_name
        types = [
            "组织机构",
            "人员",
            "地点",
            "物资",
            "事件",
            "行动",
            "法规",
            "时间",
        ]
        root_entity = KGEntity(name=source_name, entity_type="数据源", properties={})
        type_entities = [KGEntity(name=t, entity_type="实体类型", properties={"source_name": source_name}) for t in types]
        self.graph_manager.batch_create_entities([root_entity] + type_entities)
        rels = []
        for t in types:
            rels.append(KGRelation(subject=source_name, predicate="包含类型", object=t, properties={"object_label": "实体类型", "source_name": source_name}))
        self.graph_manager.batch_create_relations(rels)

    
    def build_knowledge_graph(self, data_path: str, output_dir: str = None) -> Dict[str, Any]:
        """构建知识图谱
        
        Args:
            data_path: 数据文件路径
            output_dir: 输出目录（可选）
            
        Returns:
            构建结果和统计信息
        """
        self.logger.info("开始构建应急条例知识图谱...")
        start_time = time.time()
        
        try:
            # 1. 加载数据
            raw_documents = self.load_data(data_path)
            documents = self._group_documents_by_rule_count(raw_documents, group_size=DOCUMENT_CONFIG.get("group_size", 12))
            # 覆盖统计中的文档数量为分组后的数量
            self.stats["total_documents"] = len(documents)
            try:
                srcs = sorted({d.get("source_name") or d.get("rule_source") for d in documents})
                for src in srcs:
                    if src:
                        self._ensure_meta_nodes(src)
            except Exception as e:
                self.logger.warning(f"创建元节点失败: {str(e)}")
            
            # 2. 处理文档
            results = []
            
            # 使用线程池进行并行处理
            with ThreadPoolExecutor(max_workers=PERFORMANCE_CONFIG["max_workers"]) as executor:
                futures = []
                
                for i, document in enumerate(documents):
                    future = executor.submit(self.process_document, document, i)
                    futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
            
            # 3. 保存结果
            if output_dir:
                self._save_results(results, output_dir)
            
            # 3.1 清理异常类型连接并回填实体到类型的连接边
            try:
                before_stats = self.graph_manager.get_type_link_stats()
                self.logger.info(f"类型连接状态(前): {before_stats}")
            except Exception as e:
                self.logger.warning(f"类型连接前统计失败: {str(e)}")
            try:
                cleanup_stats = self.graph_manager.cleanup_wrong_type_links()
                self.logger.info(f"已清理类型连接异常：loops={cleanup_stats.get('loops')}, subject_is_type={cleanup_stats.get('subject_is_type')}, target_not_type={cleanup_stats.get('target_not_type')}")
            except Exception as e:
                self.logger.warning(f"清理类型连接异常失败，继续后续流程: {str(e)}")
            
            # 3.2 回填实体到类型的连接边，确保所有实体均与对应类型节点连接
            try:
                linked_count = self.graph_manager.backfill_type_links()
                self.logger.info(f"实体到类型的连接边补建完成，影响关系数: {linked_count}")
            except Exception as e:
                self.logger.warning(f"补建实体类型连接失败，继续后续流程: {str(e)}")
            try:
                after_stats = self.graph_manager.get_type_link_stats()
                self.logger.info(f"类型连接状态(后): {after_stats}")
            except Exception as e:
                self.logger.warning(f"类型连接后统计失败: {str(e)}")
            # 3.3 删除仅连接实体类型的实体节点
            try:
                pruned = self.graph_manager.prune_type_only_entities()
                self.logger.info(f"删除仅连接实体类型的实体节点数量: {pruned}")
            except Exception as e:
                self.logger.warning(f"删除仅连接实体类型的实体节点失败: {str(e)}")
            
            # 4. 生成报告
            total_time = time.time() - start_time
            self.stats["processing_time"] = total_time
            
            report = self._generate_report(results)
            
            self.logger.info("应急条例知识图谱构建完成")
            self.logger.info(f"总耗时: {total_time:.2f} 秒")
            self.logger.info(f"处理文档: {self.stats['total_documents']}")
            self.logger.info(f"分块数量: {self.stats['total_chunks']}")
            self.logger.info(f"实体数量: {self.stats['total_entities']}")
            self.logger.info(f"关系数量: {self.stats['total_relations']}")
            self.logger.info(f"错误数量: {self.stats['error_count']}")
            
            return {
                "success": True,
                "results": results,
                "statistics": self.stats,
                "report": report
            }
            
        except Exception as e:
            error_msg = f"构建知识图谱失败: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            
            return {
                "success": False,
                "error": error_msg,
                "statistics": self.stats
            }
    
    def _save_results(self, results: List[Dict[str, Any]], output_dir: str):
        """保存处理结果
        
        Args:
            results: 处理结果列表
            output_dir: 输出目录
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for idx, result in enumerate(results):
                doc_id = result.get("document_id") or f"doc_{idx}"
                results_file = os.path.join(output_dir, f"processing_results_{doc_id}.json")
                save_json_data([result], results_file)

                entities = result.get("entities", []) or []
                entities_file = os.path.join(output_dir, f"extracted_entities_{doc_id}.json")
                save_json_data(entities, entities_file)

                relations = result.get("relations", []) or []
                relations_file = os.path.join(output_dir, f"extracted_relations_{doc_id}.json")
                save_json_data(relations, relations_file)
            
            self.logger.info(f"结果已保存到: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")
    
    def _generate_report(self, results: List[Dict[str, Any]]) -> str:
        """生成处理报告
        
        Args:
            results: 处理结果列表
            
        Returns:
            报告文本
        """
        success_count = sum(1 for r in results if r.get("error") is None)
        total_count = len(results)
        
        report = f"""
# 应急条例知识图谱构建报告

## 处理统计
- 总文档数: {total_count}
- 成功处理: {success_count}
- 失败处理: {total_count - success_count}
- 成功率: {(success_count/total_count*100):.1f}%

## 实体统计
- 总实体数: {self.stats['total_entities']}
- 实体类型分布:
"""
        
        # 统计实体类型
        entity_type_count = {}
        for result in results:
            for entity in result.get("entities", []):
                entity_type = entity.get("type", "未知")
                entity_type_count[entity_type] = entity_type_count.get(entity_type, 0) + 1
        
        for entity_type, count in sorted(entity_type_count.items()):
            report += f"  - {entity_type}: {count}\n"
        
        report += f"""
## 关系统计
- 总关系数: {self.stats['total_relations']}
- 关系类型分布:
"""
        
        # 统计关系类型
        relation_type_count = {}
        for result in results:
            for relation in result.get("relations", []):
                relation_type = relation.get("predicate", "未知")
                relation_type_count[relation_type] = relation_type_count.get(relation_type, 0) + 1
        
        for relation_type, count in sorted(relation_type_count.items()):
            report += f"  - {relation_type}: {count}\n"
        
        report += f"""
## 性能统计
- 总处理时间: {self.stats['processing_time']:.2f} 秒
- 平均每文档处理时间: {self.stats['processing_time']/total_count:.2f} 秒
- 平均每秒处理文档数: {total_count/self.stats['processing_time']:.2f} 个

## 图数据库统计
"""
        
        # 获取图数据库统计
        try:
            graph_stats = self.graph_manager.get_graph_statistics()
            report += f"- 总节点数: {graph_stats.get('total_nodes', 0)}\n"
            report += f"- 总关系数: {graph_stats.get('total_relationships', 0)}\n"
            
            if "nodes" in graph_stats:
                report += "- 节点类型分布:\n"
                for node_type, count in sorted(graph_stats["nodes"].items()):
                    report += f"  - {node_type}: {count}\n"
            
            if "relationships" in graph_stats:
                report += "- 关系类型分布:\n"
                for rel_type, count in sorted(graph_stats["relationships"].items()):
                    report += f"  - {rel_type}: {count}\n"
            # 类型连接完整性
            try:
                type_link_stats = self.graph_manager.get_type_link_stats()
                report += "- 类型连接完整性:\n"
                for t, s in type_link_stats.items():
                    report += f"  - {t}: total={s.get('total',0)}, linked={s.get('linked',0)}, missing={s.get('missing',0)}\n"
            except Exception as e:
                report += f"- 类型连接统计失败: {str(e)}\n"
                    
        except Exception as e:
            report += f"- 无法获取图数据库统计信息: {str(e)}\n"
        
        report += f"""
## 错误统计
- 错误数量: {self.stats['error_count']}

## 建议
1. 检查错误文档，分析失败原因
2. 根据需要调整实体和关系抽取参数
3. 验证图数据库中的数据质量
4. 考虑知识图谱的后续应用和分析
"""
        
        return report
    
    def query_knowledge_graph(self, entity_name: str = None, relation_type: str = None):
        """查询知识图谱
        
        Args:
            entity_name: 实体名称（可选）
            relation_type: 关系类型（可选）
            
        Returns:
            查询结果
        """
        try:
            if entity_name:
                # 查询特定实体
                entity_info = self.graph_manager.query_entity(entity_name)
                relations = self.graph_manager.query_relations(entity_name, relation_type)
                
                return {
                    "entity": entity_info,
                    "relations": relations
                }
            else:
                # 获取图数据库统计信息
                stats = self.graph_manager.get_graph_statistics()
                return {
                    "statistics": stats
                }
                
        except Exception as e:
            self.logger.error(f"查询知识图谱失败: {str(e)}")
            return {
                "error": str(e)
            }
    
    def clear_knowledge_graph(self):
        """清空知识图谱"""
        try:
            self.graph_manager.clear_database()
            self.logger.info("知识图谱已清空")
            return True
            
        except Exception as e:
            self.logger.error(f"清空知识图谱失败: {str(e)}")
            return False


def build_kg(data_path: str, output_dir: str = None) -> Dict[str, Any]:
    builder = EmergencyRegulationKGBuilder()
    res = builder.build_knowledge_graph(data_path, output_dir or PATH_CONFIG["output_dir"])
    builder.graph_manager.close()
    return res

def query(entity_name: str = None, relation_type: str = None) -> Dict[str, Any]:
    builder = EmergencyRegulationKGBuilder()
    res = builder.query_knowledge_graph(entity_name, relation_type)
    builder.graph_manager.close()
    return res

def index_embeddings(data_path: str, output_dir: str = None, dimensions: int = 512) -> str:
    builder = EmergencyRegulationKGBuilder()
    try:
        path = builder.index_rule_embeddings(data_path, output_dir or PATH_CONFIG["output_dir"], dimensions)
        return path
    finally:
        builder.graph_manager.close()
    
def _compute_l2(v: List[float]) -> float:
    return math.sqrt(sum(x*x for x in v)) if v else 0.0

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _embeddings_dir() -> str:
    return os.path.join(PATH_CONFIG["output_dir"], "embeddings")

def _save_jsonl(records: List[Dict[str, Any]], out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _entity_record(doc_id: str, name: str, etype: str, aliases: List[str], desc: str, v: List[float]) -> Dict[str, Any]:
    return {
        "name": name,
        "type": etype,
        "aliases": aliases or [],
        "description": desc or "",
        "embedding": v,
        "model": "embedding-3",
        "dim": len(v),
        "norm_l2": _compute_l2(v),
        "source": {
            "document_id": doc_id
        }
    }

def _entity_embeddings_path(doc_id: str) -> str:
    return os.path.join(_embeddings_dir(), f"{doc_id}_entity_embeddings.jsonl")

def _zhipu_client() -> LLMClient:
    return LLMClientFactory.create_client(
        provider="zhipu",
        api_key=API_CONFIG.get("zhipu_api_key", ""),
        model="embedding-3",
        timeout=30,
        max_retries=3
    )

def _embed_texts(texts: List[str], dimensions: int) -> List[List[float]]:
    c = _zhipu_client()
    return c.embed(texts, dimensions=dimensions)

def _entity_texts(entities: List[Entity]) -> List[str]:
    return [e.name for e in entities if (e.name or "").strip()]

def _entity_metas(entities: List[Entity]) -> List[Dict[str, Any]]:
    return [{"type": e.entity_type, "aliases": (e.properties or {}).get("aliases", []), "description": (e.properties or {}).get("description", "")} for e in entities if (e.name or "").strip()]

def _write_entity_embeddings(document_id: str, entities: List[Entity], dimensions: int = 512):
    names = _entity_texts(entities)
    metas = _entity_metas(entities)
    if not names:
        return
    recs: List[Dict[str, Any]] = []
    offset = 0
    for batch in batch_process(names, batch_size=64):
        vecs = _embed_texts(batch, dimensions)
        for i, v in enumerate(vecs):
            meta = metas[offset + i]
            recs.append(_entity_record(document_id, batch[i], meta["type"], meta["aliases"], meta["description"], v))
        offset += len(batch)
    _save_jsonl(recs, _entity_embeddings_path(document_id))
def _normalize_vec(v: List[float]) -> List[float]:
    n = _compute_l2(v)
    if n == 0.0:
        return v
    return [x / n for x in v]
def _build_alias_map_by_vectors(entities: List[Entity], dimensions: int = 512, threshold: float = 0.86) -> Dict[str, str]:
    data: List[Tuple[str, str]] = []
    for e in entities:
        n = (e.name or "").strip()
        if n:
            data.append((n, e.entity_type))
    if not data:
        return {}
    names = [n for n, t in data]
    types = [t for n, t in data]
    vecs = _embed_texts(names, dimensions)
    normed = [_normalize_vec(v) for v in vecs]
    idx_by_type: Dict[str, List[int]] = {}
    for i, t in enumerate(types):
        idx_by_type.setdefault(t, []).append(i)
    alias_map: Dict[str, str] = {}
    def _dot(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        return float(sum(x * y for x, y in zip(a, b)))

    def _ann_edges(local_vecs: List[List[float]], k: int) -> Tuple[List[List[int]], List[List[float]]]:
        if not local_vecs:
            return [], []
        k = max(1, min(k, len(local_vecs)))
        try:
            import numpy as np  # type: ignore
            import faiss  # type: ignore
            X = np.asarray(local_vecs, dtype=np.float32)
            if X.ndim != 2 or X.shape[0] == 0:
                raise ValueError("bad vectors")
            idx = faiss.IndexFlatIP(int(X.shape[1]))
            idx.add(X)
            D, I = idx.search(X, k)
            neigh = I.tolist()
            sims = D.tolist()
            return neigh, sims
        except Exception:
            neigh: List[List[int]] = []
            sims: List[List[float]] = []
            for i, vi in enumerate(local_vecs):
                scored: List[Tuple[int, float]] = []
                for j, vj in enumerate(local_vecs):
                    scored.append((j, _dot(vi, vj)))
                scored.sort(key=lambda x: x[1], reverse=True)
                top = scored[:k]
                neigh.append([j for j, _ in top])
                sims.append([s for _, s in top])
            return neigh, sims

    def _connected_components(nodes: List[int], adj: Dict[int, List[int]]) -> List[List[int]]:
        seen: set[int] = set()
        comps: List[List[int]] = []
        for s in nodes:
            if s in seen:
                continue
            stack = [s]
            seen.add(s)
            comp: List[int] = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj.get(u, []):
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(comp)
        return comps

    lambda_sim = float(threshold)
    for t, idxs in idx_by_type.items():
        if not idxs:
            continue
        local_to_global = list(idxs)
        local_vecs = [normed[i] for i in local_to_global]
        m = len(local_to_global)
        k = min(64, m)
        neigh, sims = _ann_edges(local_vecs, k)
        adj: Dict[int, List[int]] = {i: [] for i in local_to_global}
        deg: Dict[int, int] = {i: 0 for i in local_to_global}
        for li, (nbrs, ss) in enumerate(zip(neigh, sims)):
            gi = local_to_global[li]
            for lj, sim in zip(nbrs, ss):
                if lj < 0 or lj >= m:
                    continue
                gj = local_to_global[lj]
                if gi == gj:
                    continue
                if names[gi] == names[gj]:
                    continue
                if float(sim) > lambda_sim:
                    adj[gi].append(gj)
                    adj[gj].append(gi)
        for gi in local_to_global:
            if not adj.get(gi):
                continue
            uniq = list(set(adj[gi]))
            adj[gi] = uniq
            deg[gi] = len(uniq)
        comps = _connected_components(local_to_global, adj)
        for comp in comps:
            if len(comp) <= 1:
                continue
            anchor = max(comp, key=lambda i: (deg.get(i, 0), -len(names[i]), names[i]))
            kept: List[int] = []
            va = normed[anchor]
            for j in comp:
                if j == anchor:
                    kept.append(j)
                    continue
                if _dot(normed[j], va) > lambda_sim:
                    kept.append(j)
            if len(kept) <= 1:
                continue
            canonical = names[anchor]
            for j in kept:
                if names[j] != canonical:
                    alias_map[names[j]] = canonical
    return alias_map

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default=os.path.join(PATH_CONFIG["data_dir"], "*.json"),
    )
    parser.add_argument(
        "--output_dir",
        default=PATH_CONFIG["output_dir"],
    )
    parser.add_argument(
        "--neo4j_summary",
        action="store_true",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    builder = EmergencyRegulationKGBuilder()

    if args.neo4j_summary:
        summary = builder.graph_manager.get_extended_graph_summary(top_k=args.topk)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"neo4j_summary_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(out_path)
        builder.graph_manager.close()
        raise SystemExit(0)

    result = builder.build_knowledge_graph(args.data_path, args.output_dir)

    if result["success"]:
        print("知识图谱构建成功！")
        print(f"处理文档: {result['statistics']['total_documents']}")
        print(f"实体数量: {result['statistics']['total_entities']}")
        print(f"关系数量: {result['statistics']['total_relations']}")

        results = result.get("results") or []
        if results:
            first_id = str(results[0].get("document_id") or "doc_0")
            last_id = str(results[-1].get("document_id") or f"doc_{len(results)-1}")
            report_name = f"knowledge_graph_report_{first_id}_{last_id}.txt"
        else:
            report_name = "knowledge_graph_report.txt"
        report_file = os.path.join(args.output_dir, report_name)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(result["report"])
        print(f"详细报告已保存到: {report_file}")
    else:
        print(f"知识图谱构建失败: {result['error']}")

    print("\n查询示例：")
    query_result = builder.query_knowledge_graph("应急管理部")
    entity_info = query_result.get("entity")
    relations_info = query_result.get("relations", [])
    if entity_info:
        print(f"查询到实体: {entity_info['name']}")
        print(f"实体类型: {entity_info['types']}")
        print(f"相关关系数量: {len(relations_info)}")

    builder.graph_manager.close()
