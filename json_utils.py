# json_utils.py
# -*- coding: utf-8 -*-
"""
JSON 序列化工具，供 llm_parser / schedule_exporter / run 共用。
不依赖任何项目内其他模块，避免循环导入。
"""

import dataclasses


def to_json_safe(obj):
    """
    递归把任意 Python 对象转成 JSON 可序列化格式。
    处理：dict / list / tuple / set / dataclass / __dict__ / 其他
    """
    # dict：key 转 str，value 递归
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]

    # set / frozenset
    if isinstance(obj, (set, frozenset)):
        return sorted(
            (to_json_safe(x) for x in obj),
            key=lambda x: str(x),
        )

    # dataclass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_json_safe(dataclasses.asdict(obj))

    # 有 __dict__ 的普通对象
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return to_json_safe(vars(obj))

    # 基本类型
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # 兜底
    return str(obj)


def _json_default(obj):
    """json.dump 的 default 参数：只处理单层"""
    if isinstance(obj, set):
        return sorted(obj, key=str)
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)