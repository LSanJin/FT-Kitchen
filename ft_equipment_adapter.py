
# -*- coding: utf-8 -*-
"""FT项目统一设备映射：8=准备工序，9=其他设备"""

from copy import deepcopy


def classify_equipment(name):
    """根据设备名称返回统一设备ID。"""
    text = str(name or "").replace(" ", "").replace("_", "").strip()

    if not text or text.lower() in ("none", "null", "无", "无需"):
        return None

    if text.startswith(("准备工序", "等待", "被动等待", "扩展设备8")):
        return 8

    if text.startswith(("其他设备", "其它设备", "扩展设备9")):
        return 9

    other_keywords = (
        "智能电磁炉", "电磁炉", "料理机", "搅拌机",
        "破壁机", "微波炉", "空气炸锅", "电饭煲",
        "电压力锅", "压力锅", "电炖锅", "电饼铛",
        "榨汁机", "养生壶", "厨师机", "绞肉机",
        "面包机", "多功能锅",
    )

    # 必须优先于标准灶具判断
    if any(k in text for k in other_keywords):
        return 9

    standard = {
        0: ("烟机", "油烟机"),
        1: ("灶具", "燃气灶"),
        2: ("蒸箱", "蒸柜"),
        3: ("烤箱",),
        4: ("冰箱", "冷藏柜", "冷冻柜"),
        5: ("洗碗机",),
        6: ("净咖一体机", "净咖"),
        7: ("人工", "手动操作", "厨师"),
    }

    for mid, aliases in standard.items():
        if text.startswith(aliases):
            return mid

    if text.startswith("设备") and text[2:].isdigit():
        mid = int(text[2:])
        return mid if 0 <= mid <= 9 else 9

    # 其余非空名称归类为其他设备
    return 9


def normalize_recipe_equipment(data):
    """
    统一菜谱设备ID，不改变工序时长和DAG依赖。
    保留原设备名称，便于后续展示详细信息。
    """
    result = deepcopy(data)

    definitions = {}
    for machine in result.get("machines", []):
        try:
            mid = int(machine.get("id", machine.get("machine_id")))
            definitions[mid] = machine
        except (ValueError, TypeError):
            continue

    other_keywords = (
        "电磁炉", "料理机", "搅拌机", "破壁机",
        "微波炉", "空气炸锅", "电饭煲", "电压力锅",
        "电饼铛", "榨汁机", "厨师机", "绞肉机",
    )

    for step in result.get("steps", []):
        step_name = str(step.get("name", ""))
        hint = next(
            (k for k in other_keywords if k in step_name),
            None,
        )

        original_names = list(
            step.get("other_equipment_names") or []
        )

        for resource in step.get("resources", []):
            raw_id = resource.get(
                "machine_id", resource.get("id")
            )

            try:
                old_id = int(raw_id)
            except (TypeError, ValueError):
                old_id = None

            machine = definitions.get(old_id) or {}

            explicit_name = next(
                (
                    resource[k]
                    for k in (
                        "original_machine_name",
                        "machine_name",
                        "equipment_name",
                        "device_name",
                        "machine_note",
                        "device",
                    )
                    if isinstance(resource.get(k), str)
                    and resource[k].strip()
                ),
                None,
            )

            declared_name = (
                machine.get("note")
                or machine.get("name")
                or machine.get("machine_name")
            )

            explicit_id = (
                classify_equipment(explicit_name)
                if explicit_name else None
            )

            declared_id = (
                classify_equipment(declared_name)
                if declared_name else None
            )

            if hint and old_id not in (7, 100):
                new_id = 9
            elif explicit_id is not None:
                new_id = explicit_id
            elif declared_id is not None:
                new_id = declared_id
            elif old_id in (8, 100):
                new_id = 8
            elif old_id is not None and 0 <= old_id <= 7:
                new_id = old_id
            elif old_id is not None and old_id >= 9:
                new_id = 9
            else:
                raise ValueError(
                    f"工序 {step_name} 的设备无法识别：{resource}"
                )

            if new_id == 9:
                original = str(
                    explicit_name or hint or declared_name
                    or f"原设备ID{old_id}"
                )

                resource["original_machine_name"] = original

                if original not in original_names:
                    original_names.append(original)

            resource["machine_id"] = new_id

        if original_names:
            step["other_equipment_names"] = original_names

    # 标准设备仍使用原配置
    normalized = {}

    for old_id, machine in definitions.items():
        label = (
            machine.get("note")
            or machine.get("name")
            or machine.get("machine_name")
        )

        target_id = (
            classify_equipment(label)
            if label else old_id
        )

        if target_id is not None and 0 <= target_id <= 7:
            machine_copy = deepcopy(machine)
            machine_copy["id"] = target_id
            normalized[target_id] = machine_copy

    normalized[8] = {"id": 8, "note": "准备工序"}
    normalized[9] = {"id": 9, "note": "其他设备"}

    result["machines"] = list(normalized.values())

    return result


def machine_merge_key(machine):
    """多菜合并时，生成统一物理设备名称。"""
    name = str(
        getattr(machine, "note", None) or machine
    ).strip()

    mode = str(
        getattr(machine, "mode", "") or ""
    ).lower()

    if mode == "other" or classify_equipment(name) == 9:
        return "其他设备"

    if mode == "wait" or classify_equipment(name) == 8:
        return "准备工序"

    return name


def machine_display_name(machine, mid=None, compact=False):
    """统一返回甘特图和网页中的设备显示名称。"""
    name = str(
        getattr(machine, "note", None)
        or (f"设备{mid}" if mid is not None else "设备")
    ).strip()

    mode = str(
        getattr(machine, "mode", "") or ""
    ).lower()

    if mid == 9 or mode == "other":
        return "其他设备"

    if mid == 8 or mode == "wait":
        return "准备工序"

    if classify_equipment(name) == 9:
        return "其他设备"

    if classify_equipment(name) == 8:
        return "准备工序"

    return name.split()[0] if compact else name
