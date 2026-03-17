#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图数据库操作模块
用于与Neo4j图数据库进行交互，存储和查询知识图谱
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
import sys
import os

# 添加配置路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import NEO4J_CONFIG

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _cypher_ident(name: str) -> str:
    n = (name or "").strip()
    n = n.replace("`", "``")
    return f"`{n}`"


@dataclass
class KGEntity:
    """知识图谱实体"""
    name: str
    entity_type: str
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class KGRelation:
    """知识图谱关系"""
    subject: str
    predicate: str
    object: str
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class GraphDatabaseManager:
    """图数据库管理器"""
    
    def __init__(self, uri: str = None, username: str = None, password: str = None, 
                 database: str = None):
        """初始化图数据库连接
        
        Args:
            uri: Neo4j数据库URI
            username: 用户名
            password: 密码
            database: 数据库名称
        """
        self.uri = uri or NEO4J_CONFIG["uri"]
        self.username = username or NEO4J_CONFIG["username"]
        self.password = password or NEO4J_CONFIG["password"]
        self.database = database or NEO4J_CONFIG["database"]
        
        # 初始化驱动和会话
        self.driver = None
        self.session = None
        self.langchain_graph = None
        
        self._connect()
        logger.info("图数据库管理器初始化完成")
    
    def _connect(self):
        """建立数据库连接"""
        try:
            # 创建驱动
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            
            # 测试连接
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1")
                if result.single():
                    logger.info("Neo4j数据库连接成功")
            
            # 初始化LangChain图
            self.langchain_graph = Neo4jGraph(
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database
            )
            
        except Exception as e:
            logger.error(f"Neo4j数据库连接失败: {str(e)}")
            raise
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j数据库连接已关闭")
    
    def create_constraints(self):
        """创建数据库约束"""
        try:
            with self.driver.session(database=self.database) as session:
                legacy = [
                    "DROP CONSTRAINT ON (n:组织机构) ASSERT n.name IS UNIQUE",
                    "DROP CONSTRAINT ON (n:人员) ASSERT n.name IS UNIQUE",
                    "DROP CONSTRAINT ON (n:地点) ASSERT n.name IS UNIQUE",
                    "DROP CONSTRAINT ON (n:物资) ASSERT n.name IS UNIQUE",
                    "DROP CONSTRAINT ON (n:事件) ASSERT n.name IS UNIQUE",
                    "DROP CONSTRAINT ON (n:行动) ASSERT n.name IS UNIQUE",
                    "DROP CONSTRAINT ON (n:法规) ASSERT n.name IS UNIQUE",
                    "DROP CONSTRAINT ON (n:时间) ASSERT n.name IS UNIQUE",
                    "DROP CONSTRAINT ON (n:实体类型) ASSERT n.name IS UNIQUE",
                ]
                for stmt in legacy:
                    try:
                        session.run(stmt)
                    except Exception as e:
                        pass
        except Exception as e:
            pass
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:数据源) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:组织机构) REQUIRE (n.name, n.source_name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:人员) REQUIRE (n.name, n.source_name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:地点) REQUIRE (n.name, n.source_name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:物资) REQUIRE (n.name, n.source_name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:事件) REQUIRE (n.name, n.source_name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:行动) REQUIRE (n.name, n.source_name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:法规) REQUIRE (n.name, n.source_name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:时间) REQUIRE (n.name, n.source_name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:实体类型) REQUIRE (n.name, n.source_name) IS UNIQUE",
        ]
        
        try:
            with self.driver.session(database=self.database) as session:
                for constraint in constraints:
                    try:
                        session.run(constraint)
                    except Exception as e:
                        logger.warning(f"创建约束失败（可能已存在）: {str(e)}")
            
            logger.info("数据库约束创建完成")
            
        except Exception as e:
            logger.error(f"创建数据库约束失败: {str(e)}")
    
    def create_entity(self, entity: KGEntity) -> bool:
        """创建实体节点
        
        Args:
            entity: 实体对象
            
        Returns:
            是否成功创建
        """
        try:
            with self.driver.session(database=self.database) as session:
                params = {"name": entity.name}
                if entity.properties:
                    params.update(entity.properties)
                et = entity.entity_type
                if et == "数据源":
                    query = """
                    MERGE (n:数据源 {name: $name})
                    SET n.created_at = coalesce(n.created_at, datetime())
                    """
                elif et == "实体类型":
                    query = """
                    MERGE (n:实体类型 {name: $name, source_name: $source_name})
                    SET n.created_at = coalesce(n.created_at, datetime())
                    """
                else:
                    query = f"""
                    MERGE (n:{et} {{name: $name, source_name: $source_name}})
                    SET n.created_at = coalesce(n.created_at, datetime())
                    """
                if entity.properties:
                    for key in entity.properties.keys():
                        query += f"SET n.{key} = ${key}\n"
                result = session.run(query, **params)
                if result.single():
                    logger.debug(f"实体创建成功: {entity.name} ({entity.entity_type})")
                    return True
                
        except Exception as e:
            logger.error(f"创建实体失败: {entity.name} - {str(e)}")
            return False
        
        return False
    
    def create_relation(self, relation: KGRelation) -> bool:
        """创建关系
        
        Args:
            relation: 关系对象
            
        Returns:
            是否成功创建
        """
        try:
            with self.driver.session(database=self.database) as session:
                rel_type = relation.predicate.replace(' ', '_').replace('...', '_')
                subject_label = relation.properties.get('subject_label')
                object_label = relation.properties.get('object_label')
                params = {"subject_name": relation.subject, "object_name": relation.object}
                if relation.properties:
                    params.update(relation.properties)
                if rel_type == '属于类型' and object_label == '实体类型' and subject_label:
                    if "source_name" in params:
                        query = f"""
                        MATCH (s:{subject_label} {{name: $subject_name, source_name: $source_name}})
                        MATCH (o:实体类型 {{name: $object_name, source_name: $source_name}})
                        MERGE (s)-[r:{rel_type}]->(o)
                        SET r.created_at = coalesce(r.created_at, datetime())
                        """
                    else:
                        query = f"""
                        MATCH (s:{subject_label} {{name: $subject_name}})
                        MATCH (o:实体类型 {{name: $object_name}})
                        MERGE (s)-[r:{rel_type}]->(o)
                        SET r.created_at = coalesce(r.created_at, datetime())
                        """
                else:
                    if rel_type == '包含类型' and object_label == '实体类型':
                        query = f"""
                        MATCH (s:数据源 {{name: $subject_name}})
                        MATCH (o:实体类型 {{name: $object_name, source_name: $source_name}})
                        MERGE (s)-[r:{rel_type}]->(o)
                        SET r.created_at = coalesce(r.created_at, datetime())
                        """
                    else:
                        if "source_name" in params:
                            query = f"""
                            MATCH (s {{name: $subject_name, source_name: $source_name}})
                            MATCH (o {{name: $object_name, source_name: $source_name}})
                            MERGE (s)-[r:{rel_type}]->(o)
                            SET r.created_at = coalesce(r.created_at, datetime())
                            """
                        else:
                            query = f"""
                            MATCH (s {{name: $subject_name}})
                            MATCH (o {{name: $object_name}})
                            MERGE (s)-[r:{rel_type}]->(o)
                            SET r.created_at = coalesce(r.created_at, datetime())
                            """
                if relation.properties:
                    for key in relation.properties.keys():
                        query += f"SET r.{key} = ${key}\n"
                result = session.run(query, **params)
                if result.single():
                    logger.debug(f"关系创建成功: {relation.subject} -> {relation.predicate} -> {relation.object}")
                    return True
                
        except Exception as e:
            logger.error(f"创建关系失败: {relation.subject} -> {relation.predicate} -> {relation.object} - {str(e)}")
            return False
        
        return False
    
    def batch_create_entities(self, entities: List[KGEntity]) -> int:
        """批量创建实体
        
        Args:
            entities: 实体列表
            
        Returns:
            成功创建的实体数量
        """
        success_count = 0
        
        try:
            with self.driver.session(database=self.database) as session:
                # 使用事务批量创建
                with session.begin_transaction() as tx:
                    for entity in entities:
                        try:
                            # 构建属性参数
                            params = {"name": entity.name}
                            if entity.properties:
                                params.update(entity.properties)
                            
                            # 创建节点
                            et = entity.entity_type
                            if et == "数据源":
                                query = """
                                MERGE (n:数据源 {name: $name})
                                SET n.created_at = coalesce(n.created_at, datetime())
                                """
                            elif et == "实体类型":
                                query = """
                                MERGE (n:实体类型 {name: $name, source_name: $source_name})
                                SET n.created_at = coalesce(n.created_at, datetime())
                                """
                            else:
                                query = f"""
                                MERGE (n:{et} {{name: $name, source_name: $source_name}})
                                SET n.created_at = coalesce(n.created_at, datetime())
                                """
                            
                            # 添加其他属性
                            if entity.properties:
                                for key in entity.properties.keys():
                                    query += f"SET n.{key} = ${key}\n"
                            
                            tx.run(query, **params)
                            success_count += 1
                            
                        except Exception as e:
                            logger.warning(f"批量创建实体失败: {entity.name} - {str(e)}")
                    
                    # 提交事务
                    tx.commit()
                    
        except Exception as e:
            logger.error(f"批量创建实体失败: {str(e)}")
        
        logger.info(f"批量实体创建完成，成功 {success_count}/{len(entities)}")
        return success_count
    
    def batch_create_relations(self, relations: List[KGRelation]) -> int:
        """批量创建关系
        
        Args:
            relations: 关系列表
            
        Returns:
            成功创建的关系数量
        """
        success_count = 0
        
        try:
            with self.driver.session(database=self.database) as session:
                tx = session.begin_transaction()
                try:
                    for relation in relations:
                        try:
                            params = {
                                "subject_name": relation.subject,
                                "object_name": relation.object
                            }
                            if relation.properties:
                                params.update(relation.properties)
                            
                            relation_type_raw = relation.predicate.replace(' ', '_').replace('...', '_')
                            relation_type = _cypher_ident(relation_type_raw)
                            subject_label_raw = relation.properties.get('subject_label') if relation.properties else None
                            object_label = relation.properties.get('object_label') if relation.properties else None
                            
                            if relation_type_raw == '属于类型' and object_label == '实体类型' and subject_label_raw:
                                s_label = _cypher_ident(subject_label_raw)
                                o_label = _cypher_ident("实体类型")
                                if "source_name" in params:
                                    query = f"""
                                    MATCH (s:{s_label} {{name: $subject_name, source_name: $source_name}})
                                    MATCH (o:{o_label} {{name: $object_name, source_name: $source_name}})
                                    MERGE (s)-[r:{relation_type}]->(o)
                                    SET r.created_at = coalesce(r.created_at, datetime())
                                    """
                                else:
                                    query = f"""
                                    MATCH (s:{s_label} {{name: $subject_name}})
                                    MATCH (o:{o_label} {{name: $object_name}})
                                    MERGE (s)-[r:{relation_type}]->(o)
                                    SET r.created_at = coalesce(r.created_at, datetime())
                                    """
                            else:
                                if relation_type_raw == '包含类型' and object_label == '实体类型':
                                    s_label = _cypher_ident("数据源")
                                    o_label = _cypher_ident("实体类型")
                                    query = f"""
                                    MATCH (s:{s_label} {{name: $subject_name}})
                                    MATCH (o:{o_label} {{name: $object_name, source_name: $source_name}})
                                    MERGE (s)-[r:{relation_type}]->(o)
                                    SET r.created_at = coalesce(r.created_at, datetime())
                                    """
                                else:
                                    if "source_name" in params:
                                        query = f"""
                                        MATCH (s {{name: $subject_name, source_name: $source_name}})
                                        MATCH (o {{name: $object_name, source_name: $source_name}})
                                        MERGE (s)-[r:{relation_type}]->(o)
                                        SET r.created_at = coalesce(r.created_at, datetime())
                                        """
                                    else:
                                        query = f"""
                                        MATCH (s {{name: $subject_name}})
                                        MATCH (o {{name: $object_name}})
                                        MERGE (s)-[r:{relation_type}]->(o)
                                        SET r.created_at = coalesce(r.created_at, datetime())
                                        """
                            
                            if relation.properties:
                                for key in relation.properties.keys():
                                    key_ident = _cypher_ident(key)
                                    query += f"SET r.{key_ident} = ${key}\n"
                            
                            tx.run(query, **params)
                            success_count += 1
                            
                        except Exception as e:
                            logger.warning(f"批量创建关系失败: {relation.subject} -> {relation.predicate} -> {relation.object} - {str(e)}")
                            try:
                                tx.rollback()
                            except Exception:
                                pass
                            try:
                                tx.close()
                            except Exception:
                                pass
                            tx = session.begin_transaction()
                    
                    try:
                        tx.commit()
                    except Exception:
                        try:
                            tx.rollback()
                        except Exception:
                            pass
                finally:
                    try:
                        tx.close()
                    except Exception:
                        pass
                    
        except Exception as e:
            logger.error(f"批量创建关系失败: {str(e)}")
        
        logger.info(f"批量关系创建完成，成功 {success_count}/{len(relations)}")
        return success_count
    
    def query_entity(self, name: str) -> Optional[Dict[str, Any]]:
        """查询实体
        
        Args:
            name: 实体名称
            
        Returns:
            实体信息字典
        """
        try:
            with self.driver.session(database=self.database) as session:
                query = """
                MATCH (n {name: $name})
                RETURN n, labels(n) as labels
                """
                
                result = session.run(query, name=name)
                record = result.single()
                
                if record:
                    node = record["n"]
                    labels = record["labels"]
                    
                    entity_info = {
                        "name": node["name"],
                        "types": labels,
                        "properties": dict(node)
                    }
                    
                    return entity_info
                    
        except Exception as e:
            logger.error(f"查询实体失败: {name} - {str(e)}")
        
        return None
    
    def query_relations(self, entity_name: str, relation_type: str = None, 
                       direction: str = "both") -> List[Dict[str, Any]]:
        """查询实体的关系
        
        Args:
            entity_name: 实体名称
            relation_type: 关系类型（可选）
            direction: 查询方向（"in", "out", "both"）
            
        Returns:
            关系列表
        """
        relations = []
        
        try:
            with self.driver.session(database=self.database) as session:
                # 构建查询
                if direction == "out":
                    query = """
                    MATCH (s {name: $entity_name})-[r]->(o)
                    """
                elif direction == "in":
                    query = """
                    MATCH (s)-[r]->(o {name: $entity_name})
                    """
                else:  # both
                    query = """
                    MATCH (s)-[r]->(o)
                    WHERE s.name = $entity_name OR o.name = $entity_name
                    """
                
                # 添加关系类型过滤
                if relation_type:
                    relation_type_clean = relation_type.replace(' ', '_').replace('...', '_')
                    if direction == "out":
                        query += f" AND type(r) = '{relation_type_clean}'"
                    elif direction == "in":
                        query += f" AND type(r) = '{relation_type_clean}'"
                    else:
                        query += f" AND type(r) = '{relation_type_clean}'"
                
                query += """
                RETURN s.name as subject, type(r) as predicate, o.name as object, 
                       labels(s) as subject_types, labels(o) as object_types, 
                       properties(r) as relation_properties
                """
                
                result = session.run(query, entity_name=entity_name)
                
                for record in result:
                    relation_info = {
                        "subject": record["subject"],
                        "predicate": record["predicate"].replace('_', ' '),
                        "object": record["object"],
                        "subject_types": record["subject_types"],
                        "object_types": record["object_types"],
                        "properties": record["relation_properties"]
                    }
                    relations.append(relation_info)
                    
        except Exception as e:
            logger.error(f"查询关系失败: {entity_name} - {str(e)}")
        
        return relations
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """获取图数据库统计信息
        
        Returns:
            统计信息字典
        """
        stats = {}
        
        try:
            with self.driver.session(database=self.database) as session:
                # 节点统计
                node_query = """
                MATCH (n)
                RETURN labels(n) as types, count(n) as count
                """
                
                node_result = session.run(node_query)
                node_stats = {}
                for record in node_result:
                    types = record["types"]
                    count = record["count"]
                    if types:
                        node_stats[types[0]] = count
                
                stats["nodes"] = node_stats
                stats["total_nodes"] = sum(node_stats.values())
                
                # 关系统计
                rel_query = """
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
                """
                
                rel_result = session.run(rel_query)
                rel_stats = {}
                for record in rel_result:
                    rel_type = record["type"]
                    count = record["count"]
                    rel_stats[rel_type.replace('_', ' ')] = count
                
                stats["relationships"] = rel_stats
                stats["total_relationships"] = sum(rel_stats.values())
                
        except Exception as e:
            logger.error(f"获取图数据库统计信息失败: {str(e)}")
        
        return stats

    def get_extended_graph_summary(self, top_k: int = 20) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "totals": {},
            "nodes_by_label": [],
            "relationships_by_type": [],
            "nodes_by_source_name": [],
            "relationships_by_source_name": [],
            "nodes_by_source_and_label": [],
            "meta": {},
            "bridge": {},
            "property_completeness": {},
            "top_degree_nodes": [],
        }

        def _records_to_list(result, mapper):
            out = []
            for r in result:
                try:
                    out.append(mapper(r))
                except Exception:
                    continue
            return out

        try:
            with self.driver.session(database=self.database) as session:
                totals = session.run(
                    """
                    MATCH (n)
                    WITH count(n) AS nodes
                    MATCH ()-[r]->()
                    RETURN nodes, count(r) AS relationships
                    """
                ).single()
                if totals:
                    summary["totals"] = {
                        "nodes": int(totals.get("nodes", 0)),
                        "relationships": int(totals.get("relationships", 0)),
                    }

                nodes_by_label = session.run(
                    """
                    MATCH (n)
                    UNWIND labels(n) AS label
                    RETURN label, count(*) AS count
                    ORDER BY count DESC
                    """
                )
                summary["nodes_by_label"] = _records_to_list(
                    nodes_by_label,
                    lambda r: {"label": r["label"], "count": int(r["count"])},
                )

                rels_by_type = session.run(
                    """
                    MATCH ()-[r]->()
                    RETURN type(r) AS type, count(*) AS count
                    ORDER BY count DESC
                    """
                )
                summary["relationships_by_type"] = _records_to_list(
                    rels_by_type,
                    lambda r: {"type": r["type"], "count": int(r["count"])},
                )

                nodes_by_source = session.run(
                    """
                    MATCH (n)
                    WHERE n.source_name IS NOT NULL
                    RETURN n.source_name AS source_name, count(*) AS count
                    ORDER BY count DESC
                    """
                )
                summary["nodes_by_source_name"] = _records_to_list(
                    nodes_by_source,
                    lambda r: {"source_name": r["source_name"], "count": int(r["count"])},
                )

                rels_by_source = session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE r.source_name IS NOT NULL
                    RETURN r.source_name AS source_name, count(*) AS count
                    ORDER BY count DESC
                    """
                )
                summary["relationships_by_source_name"] = _records_to_list(
                    rels_by_source,
                    lambda r: {"source_name": r["source_name"], "count": int(r["count"])},
                )

                nodes_by_source_and_label = session.run(
                    """
                    MATCH (n)
                    WHERE n.source_name IS NOT NULL
                    UNWIND labels(n) AS label
                    RETURN n.source_name AS source_name, label, count(*) AS count
                    ORDER BY source_name, count DESC
                    """
                )
                summary["nodes_by_source_and_label"] = _records_to_list(
                    nodes_by_source_and_label,
                    lambda r: {
                        "source_name": r["source_name"],
                        "label": r["label"],
                        "count": int(r["count"]),
                    },
                )

                meta_counts = {}
                for rel_type in ["属于类型", "包含类型"]:
                    try:
                        rec = session.run(
                            f"MATCH ()-[r:{_cypher_ident(rel_type)}]->() RETURN count(r) AS c"
                        ).single()
                        meta_counts[rel_type] = int(rec.get("c", 0)) if rec else 0
                    except Exception:
                        meta_counts[rel_type] = 0

                for flag in ["is_bridge", "is_meta", "stat_edge", "is_llm_generated"]:
                    try:
                        rec = session.run(
                            f"""
                            MATCH ()-[r]->()
                            WHERE coalesce(r.{_cypher_ident(flag)}, false) = true
                            RETURN count(r) AS c
                            """
                        ).single()
                        meta_counts[flag] = int(rec.get("c", 0)) if rec else 0
                    except Exception:
                        meta_counts[flag] = 0
                summary["meta"] = meta_counts

                try:
                    bridge_total = session.run(
                        f"""
                        MATCH ()-[r]->()
                        WHERE coalesce(r.{_cypher_ident('is_bridge')}, false) = true
                        RETURN count(r) AS c
                        """
                    ).single()
                    bridge_total_n = int(bridge_total.get("c", 0)) if bridge_total else 0
                except Exception:
                    bridge_total_n = 0

                try:
                    paragraph_bridge = session.run(
                        f"""
                        MATCH ()-[r]->()
                        WHERE coalesce(r.{_cypher_ident('is_bridge')}, false) = true
                        AND (r.{_cypher_ident('chunk_index')} = -1 OR toString(r.{_cypher_ident('chunk_index')}) = '-1')
                        RETURN count(r) AS c
                        """
                    ).single()
                    paragraph_bridge_n = int(paragraph_bridge.get("c", 0)) if paragraph_bridge else 0
                except Exception:
                    paragraph_bridge_n = 0

                try:
                    bridge_llm = session.run(
                        f"""
                        MATCH ()-[r]->()
                        WHERE coalesce(r.{_cypher_ident('is_bridge')}, false) = true
                        AND coalesce(r.{_cypher_ident('is_llm_generated')}, false) = true
                        RETURN count(r) AS c
                        """
                    ).single()
                    bridge_llm_n = int(bridge_llm.get("c", 0)) if bridge_llm else 0
                except Exception:
                    bridge_llm_n = 0

                try:
                    non_bridge_llm = session.run(
                        f"""
                        MATCH ()-[r]->()
                        WHERE (coalesce(r.{_cypher_ident('is_bridge')}, false) = false OR r.{_cypher_ident('is_bridge')} IS NULL)
                        AND coalesce(r.{_cypher_ident('is_llm_generated')}, false) = true
                        RETURN count(r) AS c
                        """
                    ).single()
                    non_bridge_llm_n = int(non_bridge_llm.get("c", 0)) if non_bridge_llm else 0
                except Exception:
                    non_bridge_llm_n = 0

                bridge_by_source = []
                try:
                    res = session.run(
                        f"""
                        MATCH ()-[r]->()
                        WHERE coalesce(r.{_cypher_ident('is_bridge')}, false) = true
                        AND r.{_cypher_ident('source_name')} IS NOT NULL
                        RETURN r.{_cypher_ident('source_name')} AS source_name, count(*) AS count
                        ORDER BY count DESC
                        """
                    )
                    bridge_by_source = _records_to_list(
                        res, lambda r: {"source_name": r["source_name"], "count": int(r["count"])}
                    )
                except Exception:
                    bridge_by_source = []

                bridge_by_type = []
                try:
                    res = session.run(
                        f"""
                        MATCH ()-[r]->()
                        WHERE coalesce(r.{_cypher_ident('is_bridge')}, false) = true
                        RETURN type(r) AS type, count(*) AS count
                        ORDER BY count DESC
                        """
                    )
                    bridge_by_type = _records_to_list(
                        res, lambda r: {"type": r["type"], "count": int(r["count"])}
                    )
                except Exception:
                    bridge_by_type = []

                summary["bridge"] = {
                    "total": bridge_total_n,
                    "paragraph": paragraph_bridge_n,
                    "llm_generated": bridge_llm_n,
                    "by_source_name": bridge_by_source,
                    "by_relationship_type": bridge_by_type,
                }
                summary["llm_generated"] = {
                    "bridge": bridge_llm_n,
                    "non_bridge": non_bridge_llm_n,
                    "total": int(meta_counts.get("is_llm_generated", 0)),
                }

                main_labels = ["组织机构", "人员", "地点", "物资", "事件", "行动", "法规", "时间"]
                prop_stats = {}
                for lbl in main_labels:
                    try:
                        rec = session.run(
                            f"""
                            MATCH (n:{_cypher_ident(lbl)})
                            RETURN
                              count(n) AS total,
                              sum(CASE WHEN n.description IS NULL OR trim(toString(n.description)) = '' THEN 1 ELSE 0 END) AS missing_description,
                              sum(CASE WHEN n.aliases IS NULL OR size(n.aliases) = 0 THEN 1 ELSE 0 END) AS missing_aliases,
                              sum(CASE WHEN n.document_id IS NULL OR trim(toString(n.document_id)) = '' THEN 1 ELSE 0 END) AS missing_document_id,
                              sum(CASE WHEN n.chunk_index IS NULL THEN 1 ELSE 0 END) AS missing_chunk_index
                            """
                        ).single()
                        if rec:
                            prop_stats[lbl] = {
                                "total": int(rec.get("total", 0)),
                                "missing_description": int(rec.get("missing_description", 0)),
                                "missing_aliases": int(rec.get("missing_aliases", 0)),
                                "missing_document_id": int(rec.get("missing_document_id", 0)),
                                "missing_chunk_index": int(rec.get("missing_chunk_index", 0)),
                            }
                    except Exception:
                        continue
                summary["property_completeness"] = prop_stats

                try:
                    top_degree = session.run(
                        """
                        MATCH (n)
                        WHERE NOT n:实体类型 AND NOT n:数据源
                        OPTIONAL MATCH (n)-[r]-()
                        WITH n, count(r) AS degree
                        RETURN n.name AS name, labels(n) AS labels, n.source_name AS source_name, degree
                        ORDER BY degree DESC
                        LIMIT $top_k
                        """,
                        top_k=int(top_k),
                    )
                    summary["top_degree_nodes"] = _records_to_list(
                        top_degree,
                        lambda r: {
                            "name": r.get("name"),
                            "labels": r.get("labels"),
                            "source_name": r.get("source_name"),
                            "degree": int(r.get("degree", 0)),
                        },
                    )
                except Exception:
                    summary["top_degree_nodes"] = []
        except Exception as e:
            logger.error(f"获取扩展图谱统计失败: {str(e)}")

        return summary
    
    def clear_database(self):
        """清空数据库"""
        try:
            with self.driver.session(database=self.database) as session:
                # 删除所有节点和关系
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("数据库已清空")
                
        except Exception as e:
            logger.error(f"清空数据库失败: {str(e)}")
    
    def backfill_type_links(self) -> int:
        """为所有现有实体节点补建到对应实体类型元节点的连接边（属于类型）。
        返回补建的关系数量的估计值。
        """
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
        legacy_map = {
            "设备": "物资",
            "预案": "法规",
            "职责": "行动",
            "标准": "法规",
            "程序": "行动",
            "措施": "行动",
            "要求": "法规",
            "条件": "法规",
        }
        total = 0
        try:
            with self.driver.session(database=self.database) as session:
                for t in types:
                    try:
                        query = f"""
                        MATCH (root:数据源)
                        WITH root
                        MERGE (typeNode:实体类型 {{name: '{t}', source_name: root.name}})
                        WITH root, typeNode
                        MATCH (n:{t} {{source_name: root.name}})
                        MERGE (n)-[r:属于类型]->(typeNode)
                        SET r.is_meta = true, r.created_at = coalesce(r.created_at, datetime())
                        RETURN count(r) as cnt
                        """
                        result = session.run(query)
                        record = result.single()
                        if record and record["cnt"] is not None:
                            total += int(record["cnt"])
                    except Exception as e:
                        logger.warning(f"类型 '{t}' 边补建失败: {str(e)}")
                # 处理历史标签映射
                for legacy, canon in legacy_map.items():
                    try:
                        query = f"""
                        MATCH (root:数据源)
                        WITH root
                        MERGE (typeNode:实体类型 {{name: '{canon}', source_name: root.name}})
                        WITH root, typeNode
                        MATCH (n:{legacy} {{source_name: root.name}})
                        MERGE (n)-[r:属于类型]->(typeNode)
                        SET r.is_meta = true, r.created_at = coalesce(r.created_at, datetime())
                        RETURN count(r) as cnt
                        """
                        result = session.run(query)
                        record = result.single()
                        if record and record["cnt"] is not None:
                            total += int(record["cnt"])
                    except Exception as e:
                        logger.warning(f"映射 '{legacy}'→'{canon}' 类型边补建失败: {str(e)}")
                # 兜底：为任何未连类型、且首标签属于八类的节点补建类型边
                try:
                    fallback_q = """
                    MATCH (n)
                    WHERE NOT n:实体类型 AND NOT n:数据源 AND NOT (n)-[:属于类型]->(:实体类型)
                    WITH n, labels(n) AS lbs
                    WITH n, CASE WHEN size(lbs)>0 THEN lbs[0] ELSE '' END AS lbl
                    WHERE lbl IN ['组织机构','人员','地点','物资','事件','行动','法规','时间']
                    MERGE (typeNode:实体类型 {name: lbl, source_name: n.source_name})
                    MERGE (n)-[r:属于类型]->(typeNode)
                    SET r.is_meta = true, r.created_at = coalesce(r.created_at, datetime())
                    RETURN count(r) as cnt
                    """
                    res = session.run(fallback_q)
                    rec = res.single()
                    if rec and rec["cnt"] is not None:
                        total += int(rec["cnt"])
                except Exception as e:
                    logger.warning(f"兜底类型边补建失败: {str(e)}")
            logger.info("实体到类型的连接边补建完成")
        except Exception as e:
            logger.error(f"补建类型连接边失败: {str(e)}")
        return total

    def get_type_link_stats(self) -> Dict[str, Dict[str, int]]:
        """统计每类实体的类型连接情况：已连接数量与未连接数量。"""
        types = ["组织机构","人员","地点","物资","事件","行动","法规","时间"]
        stats: Dict[str, Dict[str, int]] = {}
        try:
            with self.driver.session(database=self.database) as session:
                for t in types:
                    q_total = f"MATCH (n:{t}) RETURN count(n) as c"
                    q_linked = f"MATCH (n:{t})-[:属于类型]->(:实体类型 {{name:'{t}'}}) RETURN count(n) as c"
                    q_missing = f"MATCH (n:{t}) WHERE NOT (n)-[:属于类型]->(:实体类型 {{name:'{t}'}}) RETURN count(n) as c"
                    total = int((session.run(q_total).single() or {}).get("c", 0))
                    linked = int((session.run(q_linked).single() or {}).get("c", 0))
                    missing = int((session.run(q_missing).single() or {}).get("c", 0))
                    stats[t] = {"total": total, "linked": linked, "missing": missing}
        except Exception as e:
            logger.error(f"类型连接统计失败: {str(e)}")
        return stats

    def cleanup_wrong_type_links(self) -> Dict[str, int]:
        """清理不合法的类型连接：
        - 删除 (:实体类型)-[:属于类型]->(:实体类型) 的环和链
        - 删除 主体为 :实体类型 的属于类型边
        - 删除 目标非 :实体类型 的属于类型边
        返回删除计数
        """
        stats = {"loops": 0, "subject_is_type": 0, "target_not_type": 0}
        try:
            with self.driver.session(database=self.database) as session:
                res1 = session.run("MATCH (t:实体类型)-[r:属于类型]->(t) DELETE r RETURN count(r) as c")
                rec1 = res1.single()
                stats["loops"] = int(rec1["c"]) if rec1 and "c" in rec1.keys() else 0
                res2 = session.run("MATCH (s:实体类型)-[r:属于类型]->() DELETE r RETURN count(r) as c")
                rec2 = res2.single()
                stats["subject_is_type"] = int(rec2["c"]) if rec2 and "c" in rec2.keys() else 0
                res3 = session.run("MATCH ()-[r:属于类型]->(o) WHERE NOT o:实体类型 DELETE r RETURN count(r) as c")
                rec3 = res3.single()
                stats["target_not_type"] = int(rec3["c"]) if rec3 and "c" in rec3.keys() else 0
        except Exception as e:
            logger.error(f"清理不合法类型连接失败: {str(e)}")
        return stats

    def prune_type_only_entities(self) -> int:
        """删除仅与实体类型元节点通过“属于类型”关系相连、而不与其他实体存在任何关系的实体节点。
        返回删除的节点数量。
        """
        deleted = 0
        try:
            with self.driver.session(database=self.database) as session:
                q = (
                    """
                    MATCH (n)
                    WHERE NOT n:实体类型 AND NOT n:数据源
                    AND NOT EXISTS {
                      MATCH (n)-[r]-()
                      WHERE NOT ( type(r) = '属于类型' AND ( startNode(r):实体类型 OR endNode(r):实体类型 ) )
                    }
                    WITH n
                    DETACH DELETE n
                    RETURN count(n) AS cnt
                    """
                )
                res = session.run(q)
                rec = res.single()
                if rec and rec["cnt"] is not None:
                    deleted = int(rec["cnt"])
                logger.info(f"已删除仅连接实体类型的实体节点: {deleted}")
        except Exception as e:
            logger.error(f"删除仅连接实体类型的实体节点失败: {str(e)}")
        return deleted
    
    def create_graph_from_documents(self, graph_documents: List[GraphDocument]) -> int:
        """从GraphDocument创建图
        
        Args:
            graph_documents: GraphDocument列表
            
        Returns:
            成功创建的文档数量
        """
        success_count = 0
        
        try:
            for graph_doc in graph_documents:
                # 使用LangChain的add_graph_documents方法
                self.langchain_graph.add_graph_documents([graph_doc])
                success_count += 1
                
        except Exception as e:
            logger.error(f"从GraphDocument创建图失败: {str(e)}")
        
        logger.info(f"从GraphDocument创建图完成，成功 {success_count}/{len(graph_documents)}")
        return success_count
