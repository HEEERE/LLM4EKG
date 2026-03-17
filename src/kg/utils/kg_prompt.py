#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱抽取提示词与解析工具
提供统一的LLM提示词构建与响应解析方法，便于实体与关系抽取完全依赖LLM。
"""

import re
import json
import ast
from typing import List, Dict, Any, Optional


def build_extraction_prompt(
    text: str,
    known_entities: Optional[List[Dict[str, Any]]] = None,
    predicate_whitelist: Optional[List[str]] = None,
) -> str:
    """
    构建实体与关系统一抽取提示词。
    
    Args:
        text: 输入的条例文本（一个分块或整段）
        known_entities: 已知实体（可选），用于引导关系抽取填写类型
    
    Returns:
        提示词字符串
    """
    entity_hint = ""
    if known_entities:
        entity_hint = "已知实体（供参考）：\n" + "\n".join(
            [f"- {e.get('name', '')} ({e.get('type', '未知')})" for e in known_entities]
        )

    return f"""
你是一个专业的应急管理领域知识图谱构建专家。请从给定的应急条例规则文本中提取实体和关系，构建完善的知识图谱。

【核心任务】
从每条规则中识别并提取所有重要的实体和它们之间的关系，确保知识图谱的完整性和准确性。

【实体类型定义（仅从以下集合选择，严格限定为 8 类）】
1. 组织机构：国家防总、流域防总、各级防汛抗旱指挥机构、人民政府、水利部、应急部等
2. 人员：群众、责任人、负责人、专家、信号发送员等
3. 地点：行政区划与工程设施（堤防、水库、涵闸、泵站、蓄滞洪区、受淹区域等）
4. 物资：设备与物资统一归此（防汛通信专网、监测系统、测报站网、救援器材、装备、生活物资等）
5. 事件：洪涝、干旱、台风、风暴潮、堰塞湖、山洪等灾害与突发事件
6. 行动：处置措施与具体行动（人员转移、分洪、启用蓄滞洪区、抢排、抢修、停工停学、预泄预排等）
7. 法规：规范性文件统一归此（法律、条例、办法、制度、应急预案、方案等）
8. 时间：时点与时限（“每日9时”“4小时内”“8小时内”等）

规范说明：
- 工程设施与地理位置统一归“地点”。
- 设备与物资统一归“物资”。
- 各类预案/方案与管理办法统一归“法规”。
- 响应等级不作为独立实体类型，请作为“事件/行动”的描述或属性体现。
 - 职责/责任/义务相关内容归入“行动”或作为关系描述，不单列为实体类型。
 - 实体类型仅限上述 8 类，不得新增第 9 类；若出现“设备/预案/职责”等，请分别归并为“物资/法规/行动”。

【关系抽取要求（与原文一致）】
- 关系谓词尽量直接使用原文中的动词或动词短语（如：负责、组织、协调、报告、发布、调度、启动、实施、采取、转移、抢修等）。
- 仅抽取原文中显式表达的主语-谓词-宾语，不推断或创造原文未出现的关系。
- 对关系给出简短描述（如有需要），并尽量提供 `source_text` 作为证据片段。

【质量控制要求】
1. 实体去重：相同含义的实体使用统一名称
2. 关系验证：确保每个关系的源实体和目标实体都已提取
3. 描述完整：为每个实体和关系提供清晰的描述
4. 来源标记：标记实体和关系的来源规则

{entity_hint}

请分析以下应急条例规则文本：

{text}

【返回JSON格式，严格遵守】
请只输出一个JSON对象，包含：
- entities: 实体列表（每项对象必须包含以下键）
  - name: 非空字符串，使用原文中的规范名称
  - type: 仅限 8 类之一（组织机构/人员/地点/物资/事件/行动/法规/时间）
  - description: 简要说明，字符串
  - aliases: 非空数组（至少1项），包含常见简称/全称/同义写法
  - confidence: 必填，数值，范围 [0,1]
  - source_text: 可选，摘录原文中出现该实体的证据片段
- relations: 关系列表（每项对象必须包含以下键）
  - subject, predicate, object: 分别为主语/谓词/宾语，均为非空字符串
  - subject_type, object_type: 仅限 8 类之一
  - description: 简要说明，字符串
  - confidence: 必填，数值，范围 [0,1]
  - source_text: 可选，摘录原文证据片段
仅返回该JSON对象，不要输出代码块或解释。若确实没有实体或关系，请返回空数组，不要返回空字符串或 null。

示例：
```json
{{
  "entities": [
    {{"name": "应急管理部", "type": "组织机构", "description": "国家应急管理主管部门", "aliases": ["应急管理部门"], "confidence": 0.95, "source_text": "应急管理部负责全国应急管理工作"}}
  ],
  "relations": [
    {{
      "subject": "应急管理部",
      "predicate": "负责",
      "object": "全国应急管理工作",
      "subject_type": "组织机构",
      "object_type": "行动",
      "description": "负责全国范围内应急管理相关工作",
      "confidence": 0.9,
      "source_text": "应急管理部负责全国应急管理工作"
    }}
  ]
}}
```
"""


def build_consolidation_prompt(
    document_text: str,
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    predicate_whitelist: Optional[List[str]] = None,
    isolated_entities: Optional[List[str]] = None,
) -> str:
    """
    构建文档级整合提示词：
    - 进行别名合并与指代消解，输出 alias_map 与 canonical_entities
    - 基于整体语义补充高价值桥接边（仅使用谓词白名单）
    
    Returns:
        提示词字符串
    """
    predicates = predicate_whitelist or [
        "负责", "主管", "协同", "管辖", "执行",
        "资源需求", "位于", "适用于", "适用范围", "依据",
        "处置", "响应", "采取", "需要", "使用", "相关", "涉及"
    ]

    entity_lines = "\n".join([
        f"- {e.get('name','')} ({e.get('type','未知')})" for e in entities
    ])
    relation_lines = "\n".join([
        f"- {r.get('subject','')} -{r.get('predicate','')}> {r.get('object','')}" for r in relations
    ])

    isolated_lines = "\n".join([f"- {name}" for name in (isolated_entities or [])])

    return f"""
你是一名知识图谱整合与推理专家。请对整篇文档的实体与关系进行整合：

【任务】
1. 别名合并与指代消解：
   - 识别同义/简称/全称（如“应急管理部门/应急管理部”），统一为规范名称。
   - 解析指代（如“其”、“该部门”等），补全真实实体名。
   - 输出 alias_map: {{"别名": "规范名"}}，同时给出 canonical_entities（含别名数组，确保每个规范实体至少包含1个别名）。
   - 实体类型仅限 8 类：组织机构、人员、地点、物资、事件、行动、法规、时间。不要新增其他类型；必要时将“设备/预案/职责”分别归并到“物资/法规/行动”。
2. 桥接边补充：
   - 模式谓词集合 P_schema（优先使用）：{predicates}
   - 允许生成不在 P_schema 中的开放词汇谓词 p_open，但必须能被原文的字面证据句蕴含。
   - 每条桥接边必须从原文逐字抽取一条字面支持句 s_evidence（Semantic Entailment Span），不要改写或概括。
   - 为每条桥接边给出数值置信度 confidence（0–1）；并表明该关系由LLM直接生成（is_llm_generated=true）。
   - 证据锚点必须提供：source_rule_ids（来源条目ID数组，可选）与 s_evidence（二者至少其一，但强制提供 s_evidence）。

【当前孤立实体（未与其他实体建立关系）】
{isolated_lines}
请优先为上述孤立实体补充与其他实体的非“属于类型”关系，每个孤立实体至少补充1条可证实的桥接边。

【已抽取的实体】
{entity_lines}

【已抽取的关系】
{relation_lines}

【返回JSON格式，严格遵守】
请只输出一个JSON对象，包含：
- alias_map: 对象（键为别名，值为规范名称）
- canonical_entities: 实体列表（每项：name, type, description, aliases[数组]）
- bridge_relations: 关系列表（每项同关系输出格式：subject, predicate, object, subject_type, object_type, description, confidence, source_rule_ids[数组，可选], s_evidence[必填], is_llm_generated[布尔]）

示例：
```json
{{
  "alias_map": {{"应急管理部门": "应急管理部"}},
  "canonical_entities": [
    {{"name": "应急管理部", "type": "组织机构", "description": "国家应急管理主管部门", "aliases": ["应急管理部门"]}}
  ],
  "bridge_relations": [
    {{
      "subject": "应急管理部",
      "predicate": "管辖",
      "object": "全国应急管理工作",
      "subject_type": "组织机构",
      "object_type": "行动",
      "description": "对应急管理工作的法定管辖职责",
      "confidence": 0.92,
      "source_rule_ids": ["29", "48"],
      "s_evidence": "应急管理部负责全国应急管理工作并承担管辖职责。",
      "is_llm_generated": true
    }}
  ]
}}
```
"""


def parse_json_response(result_text: str) -> Dict[str, Any]:
    """
    解析LLM返回的文本中的JSON对象（支持代码块包裹）。
    
    Args:
        result_text: LLM返回的文本
    
    Returns:
        解析后的字典（若失败返回空结构）
    """
    default: Dict[str, Any] = {
        "entities": [],
        "relations": [],
        "alias_map": {},
        "canonical_entities": [],
        "bridge_relations": [],
    }

    if not result_text or not isinstance(result_text, str):
        return dict(default)

    text = result_text.strip().lstrip("\ufeff")
    if not text:
        return dict(default)

    def _iter_balanced_sections(t: str, open_ch: str, close_ch: str):
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(t):
            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue
            if ch == open_ch:
                if depth == 0:
                    start = i
                depth += 1
                continue
            if ch == close_ch:
                if depth <= 0:
                    continue
                depth -= 1
                if depth == 0 and start >= 0:
                    yield t[start : i + 1]
                    start = -1

    def _extract_code_blocks(t: str) -> List[str]:
        blocks: List[str] = []
        for m in re.finditer(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", t):
            inner = (m.group(1) or "").strip()
            if inner:
                blocks.append(inner)
        return blocks

    def _iter_brace_objects(t: str):
        yield from _iter_balanced_sections(t, "{", "}")

    def _repair_json(s: str) -> str:
        ss = s.strip()
        if not ss:
            return ss
        ss = ss.replace("\u00a0", " ")
        ss = ss.replace("\u200b", "")
        ss = re.sub(r'(?P<prefix>[\s,\{\[]+)[\u4e00-\u9fff]+\s*(?=")', r"\g<prefix>", ss)
        ss = re.sub(r'"[\u4e00-\u9fff]+([A-Za-z_][A-Za-z0-9_]*)"\s*:', r'"\1":', ss)
        ss = re.sub(r'"confidence"\s*:\s*[^\d\-\+]*?(\d+(?:\.\d+)?)', r'"confidence": \1', ss)
        ss = re.sub(r'(\s*:\s*)(?:NaN|nan)(\s*[,}\]])', r"\1null\2", ss)
        ss = re.sub(r'(\s*:\s*)(?:Infinity|inf)(\s*[,}\]])', r"\1null\2", ss)
        ss = re.sub(r",\s*([}\]])", r"\1", ss)
        return ss

    def _select_best(obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(obj, dict):
            if "entities" in obj and "relations" in obj:
                return obj
            if "alias_map" in obj or "canonical_entities" in obj or "bridge_relations" in obj:
                return obj
            for v in obj.values():
                if isinstance(v, dict):
                    picked = _select_best(v)
                    if picked is not None:
                        return picked
            return None

        if isinstance(obj, list) and obj:
            if all(isinstance(x, dict) and ("name" in x or "type" in x) for x in obj):
                return {"entities": obj, "relations": []}
            if all(isinstance(x, dict) and ("subject" in x or "predicate" in x or "object" in x) for x in obj):
                return {"entities": [], "relations": obj}

        return None

    def _try_parse_obj(s: str) -> Optional[Dict[str, Any]]:
        ss = _repair_json(s)
        if not ss:
            return None
        try:
            obj = json.loads(ss)
            picked = _select_best(obj)
            if picked is not None:
                return picked
        except Exception:
            pass
        try:
            obj = ast.literal_eval(ss)
            picked = _select_best(obj)
            if picked is not None:
                return picked
        except Exception:
            return None
        return None

    def _try_parse_any(s: str) -> Any:
        ss = _repair_json(s)
        if not ss:
            return None
        try:
            return json.loads(ss)
        except Exception:
            pass
        try:
            return ast.literal_eval(ss)
        except Exception:
            return None

    def _collect_dicts_after_field(t: str, field_name: str, required_keys: List[str]) -> List[Dict[str, Any]]:
        idx = t.find(f'"{field_name}"')
        if idx < 0:
            idx = t.find(f"'{field_name}'")
        if idx < 0:
            return []
        tail = t[idx:]
        out: List[Dict[str, Any]] = []
        for obj_str in _iter_balanced_sections(tail, "{", "}"):
            parsed = _try_parse_any(obj_str)
            if isinstance(parsed, dict) and all(k in parsed for k in required_keys):
                out.append(parsed)
        return out

    def _extract_field_section(t: str, field_name: str, open_ch: str, close_ch: str) -> Optional[str]:
        idx = t.find(f'"{field_name}"')
        if idx < 0:
            idx = t.find(f"'{field_name}'")
        if idx < 0:
            return None
        after = t.find(open_ch, idx)
        if after < 0:
            return None
        for sec in _iter_balanced_sections(t[after:], open_ch, close_ch):
            return sec
        return None

    def _salvage_by_fields(t: str) -> Optional[Dict[str, Any]]:
        picked: Dict[str, Any] = {}

        entities_sec = _extract_field_section(t, "entities", "[", "]")
        if entities_sec:
            ents = _try_parse_obj(entities_sec)
            if ents and "entities" in ents:
                picked["entities"] = ents.get("entities", [])
            else:
                try:
                    picked["entities"] = json.loads(_repair_json(entities_sec))
                except Exception:
                    try:
                        picked["entities"] = ast.literal_eval(_repair_json(entities_sec))
                    except Exception:
                        pass
        else:
            ents = _collect_dicts_after_field(t, "entities", ["name", "type"])
            if ents:
                picked["entities"] = ents

        relations_sec = _extract_field_section(t, "relations", "[", "]")
        if relations_sec:
            rels = _try_parse_obj(relations_sec)
            if rels and "relations" in rels:
                picked["relations"] = rels.get("relations", [])
            else:
                try:
                    picked["relations"] = json.loads(_repair_json(relations_sec))
                except Exception:
                    try:
                        picked["relations"] = ast.literal_eval(_repair_json(relations_sec))
                    except Exception:
                        pass
        else:
            rels = _collect_dicts_after_field(t, "relations", ["subject", "predicate", "object"])
            if rels:
                picked["relations"] = rels

        alias_sec = _extract_field_section(t, "alias_map", "{", "}")
        if alias_sec:
            amap = _try_parse_obj(alias_sec)
            if amap and isinstance(amap.get("alias_map"), dict):
                picked["alias_map"] = amap.get("alias_map", {})
            else:
                try:
                    parsed = json.loads(_repair_json(alias_sec))
                    if isinstance(parsed, dict):
                        picked["alias_map"] = parsed
                except Exception:
                    try:
                        parsed = ast.literal_eval(_repair_json(alias_sec))
                        if isinstance(parsed, dict):
                            picked["alias_map"] = parsed
                    except Exception:
                        pass

        canon_sec = _extract_field_section(t, "canonical_entities", "[", "]")
        if canon_sec:
            try:
                parsed = json.loads(_repair_json(canon_sec))
                if isinstance(parsed, list):
                    picked["canonical_entities"] = parsed
            except Exception:
                try:
                    parsed = ast.literal_eval(_repair_json(canon_sec))
                    if isinstance(parsed, list):
                        picked["canonical_entities"] = parsed
                except Exception:
                    pass

        bridge_sec = _extract_field_section(t, "bridge_relations", "[", "]")
        if bridge_sec:
            try:
                parsed = json.loads(_repair_json(bridge_sec))
                if isinstance(parsed, list):
                    picked["bridge_relations"] = parsed
            except Exception:
                try:
                    parsed = ast.literal_eval(_repair_json(bridge_sec))
                    if isinstance(parsed, list):
                        picked["bridge_relations"] = parsed
                except Exception:
                    pass

        if picked:
            return picked
        return None

    candidate_texts = _extract_code_blocks(text) + [text]
    for candidate in candidate_texts:
        salvaged = _salvage_by_fields(candidate)
        if salvaged is not None:
            return {**default, **salvaged}

        direct = _try_parse_obj(candidate)
        if direct is not None:
            return {**default, **direct}

        for obj_str in _iter_brace_objects(candidate):
            parsed = _try_parse_obj(obj_str)
            if parsed is not None:
                return {**default, **parsed}

    return dict(default)
