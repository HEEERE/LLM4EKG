#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import logging
from typing import Optional, Dict, Any

# 允许作为模块运行：把 src 加入 sys.path，便于导入 kg.*
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from kg.pipeline.main import EmergencyRegulationKGBuilder  # type: ignore
from kg.config.config import PATH_CONFIG  # type: ignore
from scripts.build_disaster_chain import DisasterChainBuilder  # type: ignore

logger = logging.getLogger("EKGPipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_pipeline(
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    extract_rules: bool = False,
    input_docs_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    统一流水线：
    1) （可选）DOCX -> 规则JSON
    2) 规则JSON -> 构建知识图谱（Neo4j）
    3) 构建灾害链本体树并锚定实体
    """
    res: Dict[str, Any] = {"success": False}

    # Step 1: （可选）规则抽取
    if extract_rules:
        try:
            # 尝试以模块方式调用（若 extract_rules_from_docx.py 保持不变，这里可能会失败）
            from scripts.extract_rules_from_docx import RuleExtractor  # type: ignore
            if not input_docs_dir or not os.path.isdir(input_docs_dir):
                logger.warning("未提供有效的 input_docs_dir，跳过 DOCX 规则抽取")
            else:
                logger.info(f"开始 DOCX 规则抽取：{input_docs_dir}")
                extractor = RuleExtractor()
                # 简单批处理：逐个 .docx 文件处理
                import glob
                files = sorted(glob.glob(os.path.join(input_docs_dir, "*.docx")))
                for f in files:
                    try:
                        extractor.extract_from_docx(f)
                    except Exception as e:
                        logger.warning(f"文档抽取失败：{f} - {e}")
                logger.info("DOCX 规则抽取完成")
        except Exception as e:
            logger.warning(f"DOCX 抽取模块不可用或导入失败，跳过 Step1：{e}")

    # Step 2: 知识图谱构建
    builder = EmergencyRegulationKGBuilder()
    try:
        logger.info("开始构建知识图谱（规则JSON -> KG -> Neo4j）")
        result = builder.build_knowledge_graph(
            data_path=data_path or os.path.join(PATH_CONFIG["data_dir"], "*.json"),
            output_dir=output_dir or PATH_CONFIG["output_dir"],
        )
        res["kg_result"] = result
        if not result.get("success", False):
            logger.warning("知识图谱构建返回非成功状态")
        else:
            logger.info("知识图谱构建完成")
    finally:
        # 不在此处关闭连接，交由灾害链构建复用；灾害链脚本内部会关闭
        pass

    # Step 3: 灾害链构建与锚定
    try:
        logger.info("开始构建灾害链本体树并锚定实体")
        DisasterChainBuilder().run()
        logger.info("灾害链构建与锚定完成")
    except Exception as e:
        logger.warning(f"灾害链构建失败：{e}")

    res["success"] = True
    return res


if __name__ == "__main__":
    # 命令行运行：使用默认配置中的 data_dir/output_dir
    out = run_pipeline()
    print(out.get("success", False))
