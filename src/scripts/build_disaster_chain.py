#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
灾害链本体树构建与动态路由配置脚本
实现基于领域特异性(Domain Specificity)的冲突感知路由机制
"""

import os
import sys
import logging

# 将 src 目录加入 sys.path 以便导入 kg.* 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg.graph_db.graph_database import GraphDatabaseManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DisasterChainBuilder")

class DisasterChainBuilder:
    def __init__(self):
        logger.info("初始化灾害链本体树构建器...")
        self.db_manager = GraphDatabaseManager()
        
    def build_ontology_tree(self):
        """第一步：创建预案元数据节点并建立拓扑层级结构"""
        logger.info("开始构建预案拓扑层级 (L0 -> L3)...")
        
        create_nodes_cypher = """
        // L0 绝对基准
        MERGE (root:PlanMeta {name: '国家突发事件总体应急预案'}) SET root.level = 'L0'
        
        // L1 综合大类与共性保障
        MERGE (n_relief:PlanMeta {name: '国家自然灾害救助应急预案'}) SET n_relief.level = 'L1'
        MERGE (e_env:PlanMeta {name: '国家突发环境事件应急预案'}) SET e_env.level = 'L1'
        MERGE (s_comm:PlanMeta {name: '国家通信保障应急预案'}) SET s_comm.level = 'L1'
        MERGE (p_food:PlanMeta {name: '国家食品安全事故应急预案'}) SET p_food.level = 'L1'
        MERGE (e_grain:PlanMeta {name: '国家粮食应急预案'}) SET e_grain.level = 'L1'
        MERGE (e_debt:PlanMeta {name: '地方政府性债务风险应急处置预案'}) SET e_debt.level = 'L1'
        
        // L2 具体灾种与专项预案
        MERGE (n_quake:PlanMeta {name: '国家地震应急预案'}) SET n_quake.level = 'L2'
        MERGE (n_flood:PlanMeta {name: '国家防汛抗旱应急预案'}) SET n_flood.level = 'L2'
        MERGE (n_fire:PlanMeta {name: '国家森林草原火灾应急预案'}) SET n_fire.level = 'L2'
        MERGE (n_marine:PlanMeta {name: '海洋灾害应急预案'}) SET n_marine.level = 'L2'
        MERGE (e_oil:PlanMeta {name: '国家重大海上溢油应急处置预案'}) SET e_oil.level = 'L2'
        MERGE (a_power:PlanMeta {name: '国家大面积停电事件应急预案'}) SET a_power.level = 'L2'
        MERGE (a_transit:PlanMeta {name: '国家城市轨道交通运营突发事件应急预案'}) SET a_transit.level = 'L2'
        MERGE (v_elder:PlanMeta {name: '养老机构突发事件应急预案'}) SET v_elder.level = 'L2'
        
        // L3 细分衍生灾害
        MERGE (n_redtide:PlanMeta {name: '赤潮灾害应急预案'}) SET n_redtide.level = 'L3'
        """
        
        create_edges_cypher = """
        // 建立 L1 -> L0 的关系
        MATCH (root:PlanMeta {level: 'L0'}), (l1:PlanMeta {level: 'L1'})
        MERGE (l1)-[:SUB_PLAN_OF]->(root);
        """
        
        create_edges_cypher_2 = """
        // 建立 L2 -> L1 (自然灾害分支)
        MATCH (n_relief:PlanMeta {name: '国家自然灾害救助应急预案'})
        MATCH (l2_nat:PlanMeta) WHERE l2_nat.name IN ['国家地震应急预案', '国家防汛抗旱应急预案', '国家森林草原火灾应急预案', '海洋灾害应急预案']
        MERGE (l2_nat)-[:SUB_PLAN_OF]->(n_relief);
        
        // 建立 L2 -> L1 (环境事件分支)
        MATCH (e_env:PlanMeta {name: '国家突发环境事件应急预案'})
        MATCH (e_oil:PlanMeta {name: '国家重大海上溢油应急处置预案'})
        MERGE (e_oil)-[:SUB_PLAN_OF]->(e_env);
        
        // 将其他独立 L2 直接挂载到 L0
        MATCH (root:PlanMeta {level: 'L0'})
        MATCH (l2_other:PlanMeta) WHERE l2_other.name IN ['国家大面积停电事件应急预案', '国家城市轨道交通运营突发事件应急预案', '养老机构突发事件应急预案']
        MERGE (l2_other)-[:SUB_PLAN_OF]->(root);
        
        // 建立 L3 -> L2 (海洋衍生灾害分支)
        MATCH (n_marine:PlanMeta {name: '海洋灾害应急预案'}), (n_redtide:PlanMeta {name: '赤潮灾害应急预案'})
        MERGE (n_redtide)-[:SUB_PLAN_OF]->(n_marine);
        """
        
        with self.db_manager.driver.session(database=self.db_manager.database) as session:
            session.run(create_nodes_cypher)
            session.run(create_edges_cypher)
            session.run(create_edges_cypher_2)
        logger.info("预案拓扑层级构建完成！")

    def calculate_specificity(self):
        """第二步：动态计算领域特异性权重(Specificity)"""
        logger.info("开始计算领域特异性权重 (Specificity)...")
        
        calc_cypher = """
        // 获取全局最大深度作为分母
        MATCH p=(m:PlanMeta)-[:SUB_PLAN_OF*]->(root:PlanMeta {level: 'L0'})
        WITH max(length(p)) AS max_depth
        
        // 计算每个节点的深度并赋予 specificity 属性
        MATCH (node:PlanMeta)
        OPTIONAL MATCH p=(node)-[:SUB_PLAN_OF*]->(root:PlanMeta {level: 'L0'})
        WITH node, max_depth, 
             CASE WHEN node.level = 'L0' THEN 0 ELSE length(p) END AS depth
        SET node.depth = depth,
            node.specificity = toFloat(depth) / toFloat(max_depth)
        RETURN node.name as name, node.level as level, node.specificity as specificity
        ORDER BY node.specificity DESC
        """
        
        with self.db_manager.driver.session(database=self.db_manager.database) as session:
            result = session.run(calc_cypher)
            logger.info("各预案特异性得分如下：")
            for record in result:
                logger.info(f" - [{record['level']}] {record['name']}: {record['specificity']:.2f}")
                
    def anchor_entities(self):
        """第三步：将图谱中已有的物理实体与预案元节点进行锚定连接"""
        logger.info("开始将图谱物理实体锚定至 PlanMeta 节点...")
        
        anchor_cypher = """
        MATCH (e)
        WHERE NOT e:PlanMeta AND e.source_name IS NOT NULL
        MATCH (m:PlanMeta)
        // 使用 CONTAINS 进行宽泛匹配，因为 source_name 可能包含文件后缀
        WHERE e.source_name CONTAINS m.name 
        MERGE (e)-[:BELONGS_TO_SOURCE]->(m)
        """
        
        with self.db_manager.driver.session(database=self.db_manager.database) as session:
            result = session.run(anchor_cypher)
            summary = result.consume()
            logger.info(f"成功创建 {summary.counters.relationships_created} 条 BELONGS_TO_SOURCE 锚定关系。")

    def run(self):
        """执行完整流程"""
        try:
            self.build_ontology_tree()
            self.calculate_specificity()
            self.anchor_entities()
            logger.info("灾害链动态路由基础结构构建完毕！")
        finally:
            self.db_manager.close()

if __name__ == "__main__":
    builder = DisasterChainBuilder()
    builder.run()
