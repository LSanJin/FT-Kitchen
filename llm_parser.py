# llm_parser.py
# -*- coding: utf-8 -*-
"""
菜谱 -> JSON -> 调度文本

优化点：
1. LLM 只输出用到的设备（省 70%+ token）
2. 本地自动补全 10 台设备
3. 默认 qwen-turbo，失败自动降级
4. 硬超时 + 缓存
"""

import os
import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from openai import OpenAI
from json_utils import to_json_safe, _json_default

import threading
import random

# ============================================================
# 限流保护配置
# ============================================================
_LLM_SEMAPHORE = threading.Semaphore(5)   # 最多 5 个并发请求
_MAX_RETRIES = 4                          # 最多重试 4 次
_BASE_WAIT = 2.0                          # 首次等待 2 秒
_MAX_WAIT = 20.0                          # 单次最多等 30 秒


def _is_rate_limit_error(e):
    """判断异常是否是限流"""
    err_str = str(e).lower()
    indicators = [
        "rate limit", "ratelimit", "429",
        "too many requests", "quota",
        "throttl", "requests rate",
    ]
    return any(k in err_str for k in indicators)


def _is_server_error(e):
    """判断异常是否是服务端临时故障"""
    err_str = str(e).lower()
    indicators = [
        "500", "502", "503", "504",
        "internal server error", "service unavailable",
        "bad gateway", "gateway timeout",
    ]
    return any(k in err_str for k in indicators)


def _should_retry(e):
    """判断是否值得重试"""
    return _is_rate_limit_error(e) or _is_server_error(e)


# ============================================================
# 客户端
# ============================================================
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY",
                      ""),#接入大模型API密钥；并修改MODEL_NAME；
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=30.0,
    max_retries=0,
)

MODEL_NAME = "qwen-turbo"      
CACHE_DIR = "cache"

_USE_INNER_EXECUTOR = True
_executor = ThreadPoolExecutor(max_workers=8)

# ============================================================
# 设备默认表（本地补全用）
# 每台设备新增 mode_detail 字段，表示细粒度模式/档位
# ============================================================
ALL_MACHINES = {
    0: {"note": "烟机",       "capacity": 1, "rate": 0.02,
        "temp_c": 0,   "heat_level": 0, "mode": "smoke",
        "mode_detail": "弱挡"},
    1: {"note": "灶具",       "capacity": 2, "rate": 0.10,
        "temp_c": 0,   "heat_level": 3, "mode": "cook",
        "mode_detail": "中火"},
    2: {"note": "蒸箱",       "capacity": 1, "rate": 0.08,
        "temp_c": 100, "heat_level": 2, "mode": "steam",
        "mode_detail": "普通蒸"},
    3: {"note": "烤箱",       "capacity": 1, "rate": 0.08,
        "temp_c": 180, "heat_level": 2, "mode": "bake",
        "mode_detail": "上下火"},
    4: {"note": "冰箱",       "capacity": 1, "rate": 0.01,
        "temp_c": 4,   "heat_level": 0, "mode": "chill",
        "mode_detail": "冷藏"},
    5: {"note": "洗碗机",     "capacity": 1, "rate": 0.03,
        "temp_c": 0,   "heat_level": 0, "mode": "wash",
        "mode_detail": "日常洗"},
    6: {"note": "净咖一体机", "capacity": 1, "rate": 0.01,
        "temp_c": 25,  "heat_level": 0, "mode": "water",
        "mode_detail": "25℃"},
    7: {"note": "人工",       "capacity": 1, "rate": 0.50,
        "temp_c": 0,   "heat_level": 0, "mode": "human",
        "mode_detail": "手动操作"},

8: {
        "note": "准备工序",
        "capacity": 999,
        "rate": 0.0,
        "temp_c": 0,
        "heat_level": 0,
        "mode": "wait",
        "mode_detail": "none",
    },
    9: {
        "note": "其他设备",
        "capacity": 1,
        "rate": 0.05,
        "temp_c": 0,
        "heat_level": 0,
        "mode": "other",
        "mode_detail": "none",
    },
}
K = 10

# ============================================================
# 模式编码表（把中文模式名转成整数，便于在调度文本中传递）
# ============================================================
MODE_DETAIL_CODES = {
    "none": 0,
    # 烤箱
    "上下火": 1, "全开烤": 2, "鼓风烤": 3, "顶部烤": 4,
    "加湿烤": 5, "环风烤": 6, "脱脂烤": 7, "空气炸": 8,
    "蔬果干": 9, "发酵": 10, "常规烘焙": 1,
    # 蒸箱
    "普通蒸": 11, "鲜嫩蒸": 12, "脱脂蒸": 13, "多层蒸": 14,
    # 冰箱
    "冷藏": 15, "冷冻": 16, "解冻": 17,
    # 灶具
    "小火": 18, "中火": 19, "大火": 20, "猛火": 21,
    # 烟机
    "弱挡": 22, "强档": 23,
    # 洗碗机
    "日常洗": 24, "母婴洗": 25, "超净洗": 26,
    # 净咖一体机
    "25℃": 27, "45℃": 28, "60℃": 29, "85℃": 30, "98℃": 31,
    # 人工
    "手动操作": 32,
}
MODE_DETAIL_NAMES = {v: k for k, v in MODE_DETAIL_CODES.items()}


def _mode_code(name):
    """把中文模式名转成整数编码，未识别返回 0"""
    if not name:
        return 0
    return MODE_DETAIL_CODES.get(str(name).strip(), 0)


# ============================================================
# 设备批量容量表（同批次可合并的工序数）
# ============================================================
BATCH_CAPACITY = {
    2: 3,   # 蒸箱
    3: 3,   # 烤箱
    4: 3,   # 冰箱
}
DEFAULT_BATCH_CAPACITY = 1


SYSTEM_PROMPT = """你是菜谱结构化解析器。把自然语言菜谱转成严格 JSON。

## 输出格式
{
  "recipe_id": "string",
  "name": "string",
  "ingredients": [
    {"name": "鸡翅", "quantity": 3, "unit": "个", "type": "荤菜"}
  ],
  "steps": [
    {"id": 0, "name": "预热烤箱", "duration_min": 5,
     "needs_human": false,
     "resources": [{"machine_id": 3, "duration": 5,
                    "temp_c": 180, "heat_level": 2,
                    "mode_detail": "上下火"}],
     "predecessors": []}
  ]
}

## 设备id（只输出用到的）
0烟机 1灶具(2眼) 2蒸箱 3烤箱 4冰箱 5洗碗机 6净咖 7人工(不写)
8等待(被动等待，不占人工，容量无限，只用于拆分出的“等待”节点)

## ingredients（必填）
每项含 {name, quantity, unit, type}：
- name去量词；quantity数值；unit如个/克/毫升/根/片/茶匙
- type ∈ {荤菜,素菜,调味品}
  荤菜:肉/鱼/虾/蛋/禽 | 素菜:蔬菜/水果/豆/米面 | 调味品:油盐糖酱醋酒香料葱姜蒜
- "适量"→quantity=0
例: "鸡翅3个"→{name:"鸡翅",quantity:3,unit:"个",type:"荤菜"}

## 每工序独立带参数
同设备同菜可多模式，每工序填 temp_c/heat_level/mode_detail。
例: 烤箱同菜先"发酵35℃"后"上下火180℃"

## mode_detail 枚举（严格选）
烤箱3:上下火/全开烤/鼓风烤/顶部烤/加湿烤/环风烤/脱脂烤/空气炸/蔬果干/发酵
蒸箱2:普通蒸/鲜嫩蒸/脱脂蒸/多层蒸
冰箱4:冷藏/冷冻/解冻
灶具1:小火/中火/大火/猛火
烟机0:弱挡/强档
洗碗机5:日常洗/母婴洗/超净洗
净咖6:25℃/45℃/60℃/85℃/98℃
不占设备:none
等待设备8:none

## 预热规则

烤箱/蒸箱用前必须加独立"预热"步骤，是后续烹饪的前置。
⚠️ 预热步骤 needs_human=false（设备自动预热，不占人工）。

### 时长推断（分钟）

烤箱：
- 原文给出 → 用原文
- <180℃ → 4
- 180~200℃ → 6
- >200℃ → 8
- 未提温度 → 5

蒸箱：
- 原文给出 → 用原文
- 鲜嫩蒸 → 3
- 普通蒸/多层蒸 → 4
- 脱脂蒸 → 6
- 未提模式 → 4

### 格式

烤箱：
  name = "预热烤箱"
  needs_human = false
  resources = [{"machine_id": 3, "duration": <推断值>,
                "temp_c": <烹饪温度>, "heat_level": 2,
                "mode_detail": <烹饪模式>}]

蒸箱：
  name = "预热蒸箱"
  needs_human = false
  resources = [{"machine_id": 2, "duration": <推断值>,
                "temp_c": <烹饪温度>, "heat_level": 2,
                "mode_detail": <烹饪模式>}]

⚠️ 预热的 temp_c / mode_detail 必须与后续烹饪步骤一致。

## 被动等待类工序拆分规则（强制）

### 需要拆分的场景
当工序属于“被动等待”，且**不占用任何真实设备**（resources 为空）时，必须拆分为两个节点：
    浸泡、解冻（室温）、静置、醒发、发酵（室温）、腌制等待、冷却、沥干、退冰、松弛、浸味

### 不需要拆分的场景
如果等待过程**使用了真实设备**（如冰箱解冻、烤箱发酵、冷藏腌制），则保留为单个节点：
    - needs_human: false
    - resources 指向该设备，duration 为原文或常识时长
    - 不拆分

### 拆分格式（仅当不占用真实设备时）

  节点 A（人工准备）：
    - id:         新分配的连续 id
    - name:       "<原工序名>_准备"     例如 "浸泡糯米_准备"
    - duration_min: 3（固定，除非原文明确给出准备动作时长）
    - needs_human: true
    - resources:  []                    不占任何设备
    - predecessors: 与原工序相同的前置

  节点 B（等待）：
    - id:         新分配的连续 id（紧跟在 A 后面）
    - name:       "<原工序名>_等待"     例如 "浸泡糯米_等待"
    - duration_min: 原文给出的等待时长，例如 240
    - needs_human: false                ★ 关键：绝对不能占人工
    - resources:  [{"machine_id": 8, "duration": <等待时长>,
                    "temp_c": 0, "heat_level": 0,
                    "mode_detail": "none"}]
    - predecessors: [A 的 id]            即 A 完成后 B 才开始

  后续工序依赖：
    原来依赖“浸泡糯米”的下游节点，predecessors 里的 id 必须改为节点 B 的 id，
    不能再引用节点 A，也不能引用已被拆掉的旧 id。

### 完整示例

假设原步骤 id=5 是“浸泡糯米 4 小时”，下游 id=7 是“蒸糯米”。
原依赖：5 -> 7

正确输出（片段）：
  {"id": 5, "name": "浸泡糯米_准备", "duration_min": 3,
   "needs_human": true,
   "resources": [],
   "predecessors": []},

  {"id": 6, "name": "浸泡糯米_等待", "duration_min": 240,
   "needs_human": false,
   "resources": [{"machine_id": 8, "duration": 240,
                  "temp_c": 0, "heat_level": 0,
                  "mode_detail": "none"}],
   "predecessors": [5]},

  {"id": 7, "name": "蒸糯米", "duration_min": 30,
   "needs_human": false,
   "resources": [{"machine_id": 2, "duration": 30,
                  "temp_c": 100, "heat_level": 2,
                  "mode_detail": "普通蒸"}],
   "predecessors": [6]}     ★ 注意：是 [6]，不是 [5]

错误输出（不要这样）：
  {"id": 5, "name": "浸泡糯米", "duration_min": 240,
   "needs_human": true,     ← 错！会让厨师被占 240 分钟
   "resources": []}

### 注意
1. 拆分后节点总数会增加，请同步更新所有 id（从 0 连续）和 E。
2. 等待节点必须使用 machine_id=100，不要用蒸箱、烤箱、灶具等真实设备。
3. 如果原文没有明确等待时长，默认按常识给（浸泡 240、解冻 30、静置 15、发酵 60），
   并在 name 后加“_等待”。
4. 只有真正不占设备的被动等待才拆；“炒 5 分钟”“蒸 10 分钟”这种主动操作不要拆。
5. 使用设备的被动等待（如冰箱解冻、烤箱发酵）不拆分，直接一个节点，
   needs_human=false，resources 指向对应设备。

## needs_human
true:切/炒/煎/爆香/打蛋/搅拌/装盘/摆盘/刷油/调味/取水
     /所有“被动等待类工序_准备”节点
false:无人阶段的蒸烤炖/预热
     /所有“被动等待类工序_等待”节点
     /使用设备的被动等待（如冰箱解冻、烤箱发酵）
     /纯设备自动运行

## 时间(分钟)
切1 炒煎3 爆香1 焯水2 腌冷10
蒸8 烤15 炖30 发酵30 预热3 取水1
被动等待准备固定1，等待时长按原文或常识
原文给时间必须采用

## 温度
烤箱60-230 蒸箱30-120 冰箱冷藏4 其他0 原文给必须采用
等待设备8温度填0

## 依赖
id从0连续；predecessors是前置id数组；无因果不连边；必须无环
拆分出的“_准备”和“_等待”必须串联：准备 -> 等待 -> 原下游

只输出 JSON。"""

USER_PROMPT_TEMPLATE = """菜谱ID：{recipe_id}
名称：{name}
食材清单：{ingredients}
烹饪步骤：{steps}"""


# ============================================================
# 缓存
# ============================================================
def _cache_path(recipe_id: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    h = hashlib.md5(recipe_id.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{h}.json")

def human_attention_policy(step, setup_minutes=2, auto_min_minutes=6):
    """返回 (人工占用分钟数, 是否整个节点持续占人工)。

    仅用于每个节点内的人工资源，不改变蒸箱、烤箱、灶具的设备运行时长。
    可逐菜在 JSON 步骤中手工指定：
      human_mode: 'continuous' | 'setup' | 'none'
      human_attention_minutes: 非负数
    未指定时：炒煎炸等持续人工；较长煮蒸烤炖为开工时短暂人工；等待预热不占人工。
    """
    name = str(step.get('name', '') or '').replace(' ', '')
    resources = list(step.get('resources') or [])
    equipment = [r for r in resources if int(r.get('machine_id', -1)) != 7]
    duration = max(
        0,
        int(round(float(step.get('duration_min', 0) or 0))),
        *(int(round(float(r.get('duration', 0) or 0))) for r in resources),
    )
    if duration <= 0:
        return 0, False

    manual = bool(step.get('needs_human', True))
    override = step.get('human_attention_minutes', step.get('human_duration_min'))
    mode = str(step.get('human_mode', '') or '').strip().lower()

    if override is not None:
        minutes = max(0, min(duration, int(round(float(override)))))
        return minutes, bool(minutes == duration and mode != 'setup')

    if mode in ('none', 'auto', 'unattended'):
        return 0, False
    if mode in ('continuous', 'full', 'manual'):
        return duration, True
    if mode in ('setup', 'startup'):
        minutes = max(0, min(duration, int(setup_minutes)))
        return minutes, bool(minutes == duration)

    # 被动等待与设备预热不需要整段人工。
    if ('预热' in name or name.endswith('_等待') or any(
        k in name for k in ('发酵', '醒发', '泡发', '浸泡', '冷藏',
                            '冷冻', '静置', '自然冷却', '自然晾凉')
    ) and not name.endswith('_准备')):
        return 0, False

    # 含有持续手动动作的工序必须由人看管直至完成；优先于“煮/烤”等词。
    continuous_words = (
        '炒', '煸', '爆香', '翻', '煎', '炸', '搅拌', '搅匀', '打发',
        '揉', '擀', '切', '剁', '捏', '裹', '刷', '涂抹', '烙',
    )
    if any(k in name for k in continuous_words):
        return duration, True

    # 设备持续工作：开工时需短暂装料/启动，随后人工可以离开。
    # 仅当原步骤确实需要人工且存在真实烹饪设备时自动应用。
    # 时间太短的焯水或煮沸，仍按全程人工处理。
    automatic_words = ('煮', '蒸', '烤', '炖', '焖', '煲', '熬', '焯', '开始烹饪')
    cooking_machine = any(int(r.get('machine_id', -1)) in (1, 2, 3, 6)
                          for r in equipment)
    if any(k in name for k in automatic_words) and cooking_machine:
        if not manual:
            return 0, False
        if duration < int(auto_min_minutes):
            return duration, True
        minutes = max(0, min(duration, int(setup_minutes)))
        return minutes, bool(minutes == duration)

    # 非烹饪设备或普通食材准备：沿用原来的 needs_human。
    return (duration, True) if manual else (0, False)

def filter_precompleted_steps(data: dict, min_duration_minutes: float = 30.0) -> dict:
    """
    在 JSON -> Instance 之前物理删除调度前已完成的长时间预处理。

    识别：浸泡/泡发、腌制、发酵/醒发、冷藏/冷冻、静置/凝固等，
    默认持续 >= 30 分钟；显式 pre_completed=True 不受时长限制。

    删除节点后沿已删除节点向上追溯，重建保留节点之间的必要依赖。
    保留原始 step id，由 json_to_schedule_text 在转换时重新编号。
    不改写原始 LLM 缓存文件，且重复调用幂等。
    """
    import copy
    import re

    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise ValueError("菜谱 JSON 缺少 steps 数组")

    # 标记避免在返回结果上再次处理。
    if data.get("_pre_completed_filter_version") == 1:
        return copy.deepcopy(data)

    result = copy.deepcopy(data)
    raw_steps = result["steps"]
    id_to_step = {str(step["id"]): step for step in raw_steps}
    if len(id_to_step) != len(raw_steps):
        raise ValueError("菜谱 steps 中有重复 id")

    # "等待" 单独出现时，只按与预处理相关的设备工况识别；
    # 不会将正式蒸、烤、煮的较长工序删除。
    long_prep = re.compile(
        r"浸泡|泡发|泡软|腌制|腌渍|腌料|入味|"
        r"发酵|醒发|饧面|冷藏|冷冻|冰镇|冷却|凝固|定型|静置|"
        r"自然晾干|晾干|风干|解冻"
    )
    active_action = re.compile(
        r"烘烤|烤制|烤熟|蒸制|蒸熟|蒸煮|炖煮|熬煮|翻炒|"
        r"煎制|油炸|搅拌|搓揉|切块|切片|脱模|装盘|出锅|"
        r"取出|倒入|加入|混合|洗净|打发|包制"
    )
    setup_pattern = re.compile(
        r"浸泡|泡发|泡软|腌制|腌渍|发酵|醒发|冷藏|冷冻|静置"
    )

    def duration_of(step):
        value = step.get("duration_min", step.get("duration", 0))
        try:
            return max(0.0, float(value))
        except (ValueError, TypeError):
            # 当步骤自身未提供 duration_min 时，兼容机器资源持续时间。
            durations = []
            for r in step.get("resources", []):
                try:
                    durations.append(float(r.get("duration", 0)))
                except (ValueError, TypeError):
                    pass
            return max(durations, default=0.0)

    # 不能排除正式烹饪：即使设备模式里含"发酵/冷藏"，
    # 只在工序本身是长时间等待/预处理时才适用。
    removed = set()
    reasons = {}
    for step in raw_steps:
        sid = str(step["id"])
        name = str(step.get("name", "")).replace(" ", "")
        duration = duration_of(step)
        explicit = (
            step.get("pre_completed") is True
            or step.get("preparation_type") == "pre_completed"
        )
        forbid = (
            step.get("pre_completed") is False
            or step.get("preparation_type") in ("active_operation", "passive_wait")
        )
        if forbid and not explicit:
            continue
        resources = step.get("resources", []) or []
        device_modes = " ".join(
            str(r.get("mode_detail", "")) for r in resources
        )
        mode_prep = (
            bool(re.search(r"发酵|冷藏|冷冻|解冻", device_modes))
            and bool(re.search(r"等待|保温|发酵|冷藏|冷冻|解冻|静置", name))
        )
        keyword_prep = bool(long_prep.search(name))
        # 例如“冷藏后切块”“蒸熟后冷却装盘”是组合动作，
        # 不应仅按一个关键词删除整步。
        is_composite = bool(active_action.search(name))
        if explicit or (
            duration >= min_duration_minutes
            and (keyword_prep or mode_prep)
            and not is_composite
            and "预热" not in name
        ):
            removed.add(sid)
            reasons[sid] = "explicit" if explicit else "long_preparation"

    # 同一等待工序的短时启动操作也视为在调度前完成：
    # 浸泡糯米_准备(3分钟) -> 浸泡糯米_等待(240分钟)。
    # 仅删除具有唯一用途且 <=10分钟的 xxx_准备 节点。
    children = {str(step["id"]): set() for step in raw_steps}
    for step in raw_steps:
        for pred in step.get("predecessors", []):
            pid = str(pred)
            if pid in children:
                children[pid].add(str(step["id"]))
    changed = True
    while changed:
        changed = False
        for step in raw_steps:
            sid = str(step["id"])
            if sid in removed:
                continue
            name = str(step.get("name", "")).replace(" ", "")
            successors = children[sid]
            if (
                name.endswith("_准备")
                and setup_pattern.search(name)
                and duration_of(step) <= 10.0
                and successors
                and successors.issubset(removed)
            ):
                removed.add(sid)
                reasons[sid] = "setup_for_precompleted"
                changed = True

    if not removed:
        result["_pre_completed_filter_version"] = 1
        result["pre_completed_steps"] = []
        return result

    if len(removed) == len(raw_steps):
        raise ValueError(
            f"《{data.get('name', '菜谱')}》所有步骤都被认定为提前准备；"
            "请至少保留一个实际制作或出锅步骤"
        )

    # 将依赖链穿过已删除的节点。只保持真正存在的祖先；
    # 保留原 step id（不自行重编号，转换器负责重编号）。
    def kept_ancestors(step_id, visiting=None):
        if visiting is None:
            visiting = set()
        if step_id in visiting:
            raise ValueError("菜谱预处理依赖图存在循环")
        if step_id not in id_to_step:
            raise ValueError(f"菜谱依赖中引用了未知节点 {step_id}")
        if step_id not in removed:
            return {step_id}
        ancestors = set()
        for pid in id_to_step[step_id].get("predecessors", []):
            ancestors |= kept_ancestors(str(pid), visiting | {step_id})
        return ancestors

    kept_steps = []
    for step in raw_steps:
        sid = str(step["id"])
        if sid in removed:
            continue
        new_pred_ids = set()
        for pid in step.get("predecessors", []):
            new_pred_ids |= kept_ancestors(str(pid), {sid})
        new_pred_ids.discard(sid)
        # 遵循原始拓扑/步骤出现顺序，以防止变动导致输出不稳定。
        step["predecessors"] = [
            p["id"] for p in raw_steps
            if str(p["id"]) in new_pred_ids
        ]
        kept_steps.append(step)

    result["steps"] = kept_steps
    result["pre_completed_steps"] = [
        {
            "original_id": step["id"],
            "name": str(step.get("name", "")),
            "duration_min": duration_of(step),
            "reason": reasons[str(step["id"])],
        }
        for step in raw_steps if str(step["id"]) in removed
    ]
    result["_pre_completed_filter_version"] = 1
    return result

def _load_cache(recipe_id: str):
    p = _cache_path(recipe_id)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _save_cache(recipe_id: str, data: dict):
    p = _cache_path(recipe_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(data), f, ensure_ascii=False, indent=2)



def _is_rate_limit_error(e):
    """判断异常是否是限流"""
    err_str = str(e).lower()
    indicators = [
        "rate limit", "ratelimit", "429",
        "too many requests", "quota",
        "throttl", "requests rate",
    ]
    return any(k in err_str for k in indicators)


def _is_server_error(e):
    """判断异常是否是服务端临时故障（5xx）"""
    err_str = str(e).lower()
    indicators = [
        "500", "502", "503", "504",
        "internal server error", "service unavailable",
        "bad gateway", "gateway timeout",
    ]
    return any(k in err_str for k in indicators)


def _should_retry(e):
    """判断是否值得重试"""
    return _is_rate_limit_error(e) or _is_server_error(e)
# ============================================================
# 调用 LLM
# ============================================================
def _call_llm_sync(model_name: str, system: str, user: str,
                   timeout_s: float = 25.0) -> str:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=16000,
        stream=False,
        timeout=timeout_s,
    )
    return completion.choices[0].message.content


def remove_recipe_preheat_steps(data: dict) -> dict:
    """把菜谱级蒸箱/烤箱预热步骤移出调度 DAG，改由设备热状态求解器生成。

    前提：optimize_lexicographic_schedule() 已具备设备级自动预热和
    preheat_records，且会约束预热占用真实设备。删除的不是预热物理过程，
    只是不再让同一个热操作同时由菜谱节点与设备状态求解器负责。

    保留未删除节点的原始 step.id，通过已移除节点追溯真正依赖的祖先。
    不修改原始缓存/输入；可重复调用。
    """
    import copy
    import re

    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise ValueError("菜谱 JSON 必须包含 steps 列表")
    result = copy.deepcopy(data)
    steps = result["steps"]
    id_to_step = {str(s["id"]): s for s in steps}
    if len(id_to_step) != len(steps):
        raise ValueError("菜谱步骤 ID 重复，不能安全移除预热节点")

    def is_machine_preheat(s):
        name = re.sub(r"\s+", "", str(s.get("name", "")))
        if "预热" not in name:
            return False
        # 混合“预热并放入/开始烘烤”是复合操作，不可直接移除。
        if re.search(r"放入|加入|开始烹饪|开始蒸|开始烤|烤制|蒸制|烘烤|装入", name):
            return False
        resources = s.get("resources") or []
        device_ids = set()
        for r in resources:
            try:
                device_ids.add(int(r.get("machine_id")))
            except (TypeError, ValueError, AttributeError):
                pass
        return bool(device_ids & {2, 3}) or (
            not device_ids and bool(re.search(r"蒸箱|烤箱", name))
        )

    removed = {str(s["id"]) for s in steps if is_machine_preheat(s)}
    if not removed:
        result.setdefault("device_managed_preheats", [])
        return result
    if len(removed) == len(steps):
        raise ValueError("菜谱只含预热步骤，没有实际烹饪工序")

    def ancestors(step_id, visiting):
        if step_id in visiting:
            raise ValueError("预热节点前驱存在依赖循环")
        step = id_to_step.get(step_id)
        if step is None:
            raise ValueError(f"引用未知菜谱步骤：{step_id}")
        if step_id not in removed:
            return {step_id}
        found = set()
        for p in step.get("predecessors", ()):
            found.update(ancestors(str(p), visiting | {step_id}))
        return found

    new_steps = []
    for step in steps:
        sid = str(step["id"])
        if sid in removed:
            continue
        required = set()
        for p in step.get("predecessors", ()):
            required.update(ancestors(str(p), {sid}))
        required.discard(sid)
        step["predecessors"] = [
            original["id"] for original in steps
            if str(original["id"]) in required
        ]
        new_steps.append(step)

    result["steps"] = new_steps
    previous = list(result.get("device_managed_preheats") or [])
    existing_ids = {str(p.get("original_id")) for p in previous}
    previous.extend({
        "original_id": s["id"], "name": str(s.get("name", "")),
        "duration_min": float(s.get("duration_min", s.get("duration", 0)) or 0),
        "handled_by": "CP-SAT设备级预热",
    } for s in steps if str(s["id"]) in removed and str(s["id"]) not in existing_ids)
    result["device_managed_preheats"] = previous
    return result


def recipe_to_schedule_json(recipe_id: str, name: str,
                            ingredients: str, steps: str,
                            verbose: bool = True,
                            use_cache: bool = True) -> dict:
    """获取原始菜谱，并在返回给调度器前排除已提前完成的长时间准备工序。"""
    import json
    import time
    import random

    # 缓存保留原始步骤，以便日后修改过滤规则时无需重新请求LLM。
    if use_cache:
        cached = _load_cache(recipe_id)
        if cached is not None:
            filtered = ensure_required_final_steam(
                remove_recipe_preheat_steps(filter_precompleted_steps(cached))
            )
            if verbose:
                print("  → 命中缓存（跳过 LLM）")
            if filtered["pre_completed_steps"]:
                excluded = "、".join(
                    f"{s['name']}({s['duration_min']:g}min)"
                    for s in filtered["pre_completed_steps"]
                )
                print(f"  [调度前已完成] {name}: {excluded}")
            return filtered

    user_prompt = USER_PROMPT_TEMPLATE.format(
        recipe_id=recipe_id, name=name,
        ingredients=ingredients, steps=steps,
    )
    candidate_models = [MODEL_NAME, "qwen-turbo", "qwen-plus"]
    hard_timeout = 25.0
    last_err = None

    for attempt, current_model in enumerate(candidate_models, 1):
        print(f"  → 第 {attempt}/{len(candidate_models)} 次，"
              f"模型={current_model}...")
        t0 = time.time()
        content = None
        call_err = None
        for retry in range(_MAX_RETRIES + 1):
            with _LLM_SEMAPHORE:
                future = _executor.submit(
                    _call_llm_sync, current_model,
                    SYSTEM_PROMPT, user_prompt, hard_timeout,
                )
                try:
                    content = future.result(timeout=hard_timeout + 2)
                    call_err = None
                    break
                except FutureTimeout:
                    call_err = TimeoutError("hard timeout")
                    if retry < _MAX_RETRIES:
                        wait = min(_BASE_WAIT * (2 ** retry), _MAX_WAIT)
                        wait += random.uniform(0, 1)
                        print(f"  ⚠️ 硬超时，{wait:.1f}s 后重试 ({retry + 1}/{_MAX_RETRIES})")
                    continue
                except Exception as e:
                    call_err = e
                    if _should_retry(e) and retry < _MAX_RETRIES:
                        wait = min(_BASE_WAIT * (2 ** retry), _MAX_WAIT)
                        wait += random.uniform(0, 1)
                        time.sleep(wait)
                        continue
                    break
        if content is None:
            last_err = call_err
            print(f"  ✗ 第 {attempt}/{len(candidate_models)} 次失败：{call_err}")
            continue
        if verbose:
            print(f"  ✓ 第 {attempt} 次成功，耗时 {time.time() - t0:.2f}s")
        payload = content.strip()
        if payload.startswith("```"):
            payload = payload.split("\n", 1)[1] if "\n" in payload else payload
            if payload.endswith("```"):
                payload = payload[:-3]
            payload = payload.strip()
            if payload.startswith("json"):
                payload = payload[4:].lstrip()
        try:
            original = json.loads(payload)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        # 只保存原始结果，不覆盖缓存为过滤版。
        if use_cache:
            _save_cache(recipe_id, original)
        filtered = ensure_required_final_steam(
            remove_recipe_preheat_steps(filter_precompleted_steps(original))
        )
        if filtered["pre_completed_steps"]:
            excluded = "、".join(
                f"{s['name']}({s['duration_min']:g}min)"
                for s in filtered["pre_completed_steps"]
            )
            print(f"  [调度前已完成] {name}: {excluded}")
        return filtered
    raise RuntimeError(f"LLM 调用全部失败，最后错误: {last_err}")

# ============================================================
# JSON -> 调度文本（本地补全设备）
# ============================================================
def json_to_schedule_text(data: dict) -> str:
    """
    把 LLM 的 JSON 转成调度文本。

    新格式（含节点参数行）：
        第1行：      N E K
        第2行：      K cap_0 ... cap_{K-1}
        接下来 E 行： from to
        接下来 N 行： p needs_human (machine_id duration)*p
        ★ 接下来 N 行：(temp_c heat_level mode_code duration)*p
        最后 K 行：  设备参数行
    """
    from ft_equipment_adapter import normalize_recipe_equipment

    data = normalize_recipe_equipment(data)
    # ---- 1. 补全设备 ----
    llm_machines = {m["id"]: m for m in data.get("machines", [])}
    machines = []
    for mid in range(10):                      # 8 台：含人工
        base = dict(ALL_MACHINES[mid])
        base["id"] = mid
        llm_m = llm_machines.get(mid, {})
        for k in ("temp_c", "heat_level", "mode", "mode_detail"):
            if k in llm_m:
                base[k] = llm_m[k]
        machines.append(base)

    K = 10

    # ---- 2. 步骤重编号 ----
    steps = data["steps"]
    N = len(steps)
    id_map = {s["id"]: i for i, s in enumerate(steps)}

    # ---- 3. 依赖边 ----
    edges = []
    for i, s in enumerate(steps):
        for p in s.get("predecessors", []):
            if p in id_map:
                edges.append((id_map[p], i))
    edges.sort()
    E = len(edges)

    # ---- 4. 每个节点分开计算人工时长与设备时长 ----
    step_resources = []
    machine_duration = {m["id"]: 0 for m in machines}
    for step in steps:
        original = [dict(r) for r in step.get("resources", ())]
        # 删除原本可能与设备运行一样长的人工资源记录，重新设置。
        equipment = [r for r in original if int(r.get("machine_id", -1)) != 7]
        human_minutes, continuous = human_attention_policy(step)
        if human_minutes > 0:
            equipment.append({
                "machine_id": 7,
                "duration": human_minutes,
                "temp_c": 0,
                "heat_level": 0,
                "mode_detail": "手动操作",
            })
        # needs_human=True 在旧 CP-SAT 中表示全节点持续人工，
        # 短暂启动人工改由设备7的人工资源操作区间承担。
        needs_human = bool(continuous and human_minutes > 0)
        step_resources.append({
            "resources": equipment,
            "needs_human": needs_human,
        })
        for r in equipment:
            mid = int(r["machine_id"])
            machine_duration[mid] = machine_duration.get(mid, 0) + int(r["duration"])

    # ---- 5. 生成文本 ----
    lines = []
    lines.append(f"{N} {E} {K}")
    lines.append(str(K) + " " + " ".join(str(m["capacity"]) for m in machines))
    for frm, to in edges:
        lines.append(f"{frm} {to}")

    # ---- 节点行：p needs_human (machine_id duration)*p ----
    for sr in step_resources:
        resources = sr["resources"]
        needs_human = sr["needs_human"]
        row = [str(len(resources)), "1" if needs_human else "0"]
        for r in resources:
            row.append(str(r["machine_id"]))
            row.append(str(int(r["duration"])))
        lines.append(" ".join(row))

    # ---- ★ 节点参数行：(temp_c heat_level mode_code duration)*p ----
    for sr in step_resources:
        resources = sr["resources"]
        if not resources:
            lines.append("0 0 0 0")
        else:
            param_row = []
            for r in resources:
                temp = int(r.get("temp_c", 0))
                heat = int(r.get("heat_level", 0))
                mode_code = _mode_code(r.get("mode_detail", "none"))
                dur = int(r["duration"])
                param_row.extend([
                    str(temp), str(heat),
                    str(mode_code), str(dur),
                ])
            lines.append(" ".join(param_row))

    # ---- 设备参数行 ----
    for m in machines:
        note = (str(m.get("note", "none")).strip()
                .replace(" ", "_") or "none")
        mode_detail = (str(m.get("mode_detail", "none")).strip()
                       .replace(" ", "_") or "none")
        dur = machine_duration.get(m["id"], 0)
        line = (f"{m['id']} {m['capacity']} {m['rate']} "
                f"{m.get('temp_c', 0)} {m.get('heat_level', 0)} "
                f"{m.get('mode', 'none')} {dur} {note} {mode_detail}")
        lines.append(line)

    return "\n".join(lines)


def ensure_required_final_steam(data: dict, default_steam_minutes: int = 30) -> dict:
    """修复“轻松一锅蒸”缺失的正式蒸制，禁止将备料完成误判为菜品完成。

    调用时机：原始菜谱完成长期预处理过滤、菜谱级预热移除之后，
    JSON -> 调度文本之前。仅对确定需要整锅蒸制的该菜生效。

    30 分钟取自该菜谱此前的解析缓存；原始 CSV 没给明确时长，
    如工艺另有要求，应通过 default_steam_minutes 调整。
    """
    from copy import deepcopy

    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise ValueError("菜谱 JSON 必须有 steps 列表")

    if not (
        str(data.get("recipe_id", "")) == "66715324bfbee338853895c7"
        or str(data.get("name", "")).strip() == "轻松一锅蒸"
    ):
        return data

    result = deepcopy(data)
    steps = result["steps"]
    minutes = int(default_steam_minutes)
    if minutes <= 0:
        raise ValueError("轻松一锅蒸正式蒸制时长必须大于0")

    # 真正的蒸制 = 占用物理蒸箱 machine_id=2，不能只按名称包含“蒸”判断。
    def has_steam_resource(step):
        return any(
            isinstance(r, dict)
            and str(r.get("machine_id")) == "2"
            and float(r.get("duration", 0) or 0) > 0
            and "预热" not in str(step.get("name", ""))
            for r in (step.get("resources") or [])
        )

    if any(has_steam_resource(step) for step in steps):
        return result

    steam_resource = {
        "machine_id": 2,
        "duration": minutes,
        "temp_c": 100,
        "heat_level": 2,
        "mode_detail": "普通蒸",
    }

    # 情况 A：保留了“开始烹饪”等步骤，但 LLM 错把它解析为纯人工操作。
    import re
    explicit_cook = [
        step for step in steps
        if re.search(r"开始烹饪|正式蒸制|开始蒸制|^蒸制$", str(step.get("name", "")))
        and "预热" not in str(step.get("name", ""))
    ]
    if len(explicit_cook) > 1:
        raise ValueError("轻松一锅蒸存在多个疑似正式蒸制步骤，需人工核查")
    if len(explicit_cook) == 1:
        target = explicit_cook[0]
        target["duration_min"] = minutes
        target["needs_human"] = False
        target["resources"] = [steam_resource]
        print(f"[工艺补全] 轻松一锅蒸：修复“{target['name']}”的蒸箱占用 {minutes} 分钟")
        return result

    # 情况 B：整道菜停在最后的备料/装盘步骤，实际蒸制节点完全缺失。
    if not steps:
        raise ValueError("轻松一锅蒸没有任何备料工序，不能自动推断蒸制前驱")
    prep_candidates = [
        step for step in steps
        if any(key in str(step.get("name", ""))
               for key in ("蒸烤盘", "蒸烤架", "摆盘", "装盘", "放置鳙鱼头", "放入蒸箱"))
    ]
    if not prep_candidates:
        raise ValueError(
            "轻松一锅蒸缺少蒸制工序，也无法确定最后装盘/装炉步骤；"
            "请核查原始缓存，不可直接判定制作完成"
        )
    # 优先采用最后一个备料终点（按步骤出现顺序），不替代已存在的备料。
    predecessor = prep_candidates[-1]["id"]
    raw_ids = {str(s.get("id")) for s in steps}
    if all(str(s["id"]).lstrip("-").isdigit() for s in steps):
        new_id = max(int(s["id"]) for s in steps) + 1
    else:
        new_id = "ft_auto_final_steam"
        if str(new_id) in raw_ids:
            raise ValueError("补全蒸制节点 ID 已存在")

    steps.append({
        "id": new_id,
        "name": "开始烹饪（整锅蒸制）",
        "duration_min": minutes,
        "needs_human": False,
        "resources": [steam_resource],
        "predecessors": [predecessor],
    })
    print(f"[工艺补全] 轻松一锅蒸：{predecessor} -> 新增整锅蒸制 {minutes} 分钟（蒸箱100℃）")
    return result

# ============================================================
# 兼容旧接口
# ============================================================
def recipe_to_schedule_text(recipe_id: str, name: str,
                            ingredients: str, steps: str,
                            stream: bool = False,
                            verbose: bool = True) -> str:
    """保留旧签名，内部走 JSON + 本地补全。"""
    data = recipe_to_schedule_json(
        recipe_id, name, ingredients, steps, verbose=verbose
    )
    return json_to_schedule_text(data)