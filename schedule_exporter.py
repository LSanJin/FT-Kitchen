# schedule_exporter.py
# -*- coding: utf-8 -*-
"""
调度结果 -> 结构化 JSON

导出内容:
    - meta       : makespan / 下界 / gap / 加速比 / 是否接近最优
    - recipes    : 菜谱清单
    - machines   : 设备清单 + 利用率
    - schedule   : 逐节点的开始/结束/工序明细
    - by_machine : 按设备分组的工序列表
    - by_recipe  : 按菜谱分组的节点列表
"""

import json
import os
from datetime import datetime
from collections import defaultdict
from json_utils import to_json_safe, _json_default
from ft_equipment_adapter import machine_display_name

def _machine_utilization(multi, res):
    """每台设备利用率 = 忙碌时间 / (容量 × makespan)"""
    makespan = res.makespan if res.makespan > 0 else 1.0
    busy = defaultdict(float)
    for rec in res.op_records:
        nid, mid, bid, s, e = rec[0], rec[1], rec[2], rec[3], rec[4]
        busy[mid] += (e - s)
    util = {}
    for k in range(multi.K):
        cap = max(1, multi.capacities[k])
        total_cap = cap * makespan
        util[k] = round(busy[k] / total_cap, 4) if total_cap > 0 else 0.0
    return util


# ============================================================
# 生成中文解析文档
# ============================================================
def build_chinese_doc(multi, res, lb=None, serial_makespan=None,
                      notifications=None):
    """
    生成人类可读的中文解析文档，供 JSON 输出使用。

    返回: dict，包含 4 个部分
        - 概览: 整体调度结果的中文描述
        - 菜谱说明: 每道菜的步骤明细
        - 设备说明: 每台设备的占用情况
        - 提醒清单: 关键节点提醒（可选）
    """
    makespan = float(res.makespan)

    # ---------- 1. 概览 ----------
    overview_lines = []
    overview_lines.append(
        f"本次调度共涉及 {len(multi.recipe_names)} 道菜谱，"
        f"总步骤 {multi.N} 步，依赖 {multi.E} 条，"
        f"设备 {multi.K} 台。"
    )
    overview_lines.append(
        f"所有菜品的总完成时间（makespan）为 {makespan:.1f} 分钟。"
    )
    if serial_makespan and serial_makespan > 0:
        saved = serial_makespan - makespan
        speedup = serial_makespan / max(1e-9, makespan)
        overview_lines.append(
            f"若逐道菜单独制作，串行总耗时为 {serial_makespan:.1f} 分钟，"
            f"并行后节省 {saved:.1f} 分钟，"
            f"节省比例 {saved / serial_makespan * 100:.1f}%，"
            f"加速比 {speedup:.2f} 倍。"
        )
    if lb and lb.get("lb", 0) > 0:
        gap = (makespan - lb["lb"]) / lb["lb"] * 100
        overview_lines.append(
            f"理论下界为 {lb['lb']:.1f} 分钟"
            f"（关键路径 {lb['lb_cp']:.1f} 分钟，"
            f"机器负荷 {lb['lb_machine']:.1f} 分钟），"
            f"当前方案与下界的差距为 {gap:.1f}%。"
        )

    # ---------- 2. 菜谱说明 ----------
    # 每道菜的步骤，按开始时间排序
    recipe_doc = []
    for rid in range(len(multi.recipe_names)):
        recipe_name = multi.recipe_names[rid]
        recipe_id = (multi.recipe_ids[rid]
                     if rid < len(multi.recipe_ids) else "")

        # 收集该菜所有节点
        nodes_in_recipe = [
            nid for nid in range(multi.N)
            if multi.recipe_of_node[nid] == rid
        ]
        nodes_in_recipe.sort(key=lambda x: res.node_start[x])

        steps_doc = []
        for seq, nid in enumerate(nodes_in_recipe):
            s = res.node_start[nid]
            e = res.node_end[nid]
            # 步骤名
            raw = multi.step_names[nid].split("#")[-1]
            step_name = raw if raw else f"步骤{seq+1}"
            # 该步骤占用的设备
            ops = [
                (mid, bid, op_s, op_e)
                for (nid_, mid, bid, op_s, op_e) in res.op_records
                if nid_ == nid
            ]
            devices_desc = []
            for mid, bid, _, _ in ops:
                if mid == 7:
                    devices_desc.append("人工操作")
                else:
                    dev_name = machine_display_name(
                        multi.machines[mid],
                        mid=mid
                    )
                    devices_desc.append(dev_name)

            step_entry = {
                "序号": seq + 1,
                "步骤名": step_name,
                "开始时间": round(s, 1),
                "结束时间": round(e, 1),
                "耗时": round(e - s, 1),
                "占用资源": devices_desc if devices_desc else ["无"],
                "是否需要人工": bool(
                    multi.needs_human[nid]
                    if hasattr(multi, "needs_human") and multi.needs_human
                    else False
                ),
            }
            steps_doc.append(step_entry)

        recipe_doc.append({
            "菜谱名": recipe_name,
            "菜谱ID": recipe_id,
            "步骤数": len(nodes_in_recipe),
            "开始时间": round(min(
                (res.node_start[n] for n in nodes_in_recipe),
                default=0.0
            ), 1),
            "完成时间": round(max(
                (res.node_end[n] for n in nodes_in_recipe),
                default=0.0
            ), 1),
            "步骤明细": steps_doc,
        })

    # ---------- 3. 设备说明 ----------
    machine_doc = []
    for k in range(multi.K):
        m = multi.machines[k]
        name = m.note or f"设备{k}"
        # 该设备上所有工序
        ops = [
            (nid, bid, s, e)
            for (nid, mid, bid, s, e) in res.op_records
            if mid == k
        ]
        if not ops:
            machine_doc.append({
                "设备名": name,
                "设备编号": k,
                "容量": multi.capacities[k],
                "总占用时长": 0.0,
                "利用率": "0%",
                "占用记录": [],
            })
            continue

        ops.sort(key=lambda x: x[2])
        total_busy = sum(e - s for _, _, s, e in ops)
        cap = max(1, multi.capacities[k])
        util = total_busy / (cap * makespan) if makespan > 0 else 0

        # 占用记录
        records = []
        for nid, bid, s, e in ops:
            rid = multi.recipe_of_node[nid]
            recipe_name = multi.recipe_names[rid]
            raw = multi.step_names[nid].split("#")[-1]
            records.append({
                "菜谱": recipe_name,
                "步骤": raw or f"N{nid}",
                "实例": bid,
                "开始": round(s, 1),
                "结束": round(e, 1),
                "时长": round(e - s, 1),
            })

        machine_doc.append({
            "设备名": name,
            "设备编号": k,
            "容量": multi.capacities[k],
            "总占用时长": round(total_busy, 1),
            "利用率": f"{util * 100:.1f}%",
            "占用记录": records,
        })

    # ---------- 4. 提醒清单 ----------
    reminder_doc = []
    if notifications:
        for r in notifications:
            reminder_doc.append({
                "级别": r.get("level", "info"),
                "触发时刻": round(r.get("time", 0), 1),
                "菜谱": r.get("recipe", ""),
                "步骤": r.get("step", ""),
                "消息": r.get("message", ""),
            })

    # ---------- 5. 组织成完整文档 ----------
    doc = {
        "概览": "\n".join(overview_lines),
        "菜谱说明": recipe_doc,
        "设备说明": machine_doc,
    }
    if reminder_doc:
        doc["提醒清单"] = reminder_doc

    return doc

def export_schedule_json(
    multi,
    res,
    save_path=None,
    lb=None,
    serial_makespan=None,
    mode="multi",
    now=None,
    notifications=None,
):
    """
    把调度结果导出为结构化 JSON。

    参数:
        multi             : MultiInstance 或 Instance
        res               : ScheduleResult 或 MultiScheduleResult
        save_path         : 保存路径；None 则只返回 dict
        lb                : 下界 dict（来自 compute_lower_bound）
        serial_makespan   : 串行总耗时（用于算加速比）
        mode              : "single" / "multi" / "reschedule"
        now               : 重排时刻（仅 reschedule 模式）
        notifications     : 通知列表（仅 reschedule 模式）
    """
    makespan = float(res.makespan)

    # ---------- 判断单菜 / 多菜 ----------
    is_multi = (hasattr(multi, "recipe_of_node")
                and hasattr(multi, "recipe_names")
                and multi.recipe_names)

    # ---------- 设备 ----------
    util = _machine_utilization(multi, res)
    machines_out = []
    for k in range(multi.K):
        m = multi.machines[k]
        machines_out.append({
            "id": k,
            "name": m.note or f"machine_{k}",
            "capacity": multi.capacities[k],
            "rate": m.rate,
            "temp_c": getattr(m, "temp_c", 0),
            "heat_level": getattr(m, "heat_level", 0),
            "mode": getattr(m, "mode", "none"),
            "utilization": util[k],
        })

    # ---------- 菜谱 ----------
    recipes_out = []
    if is_multi:
        for i, name in enumerate(multi.recipe_names):
            rid = multi.recipe_ids[i] if i < len(multi.recipe_ids) else ""
            recipes_out.append({"index": i, "id": rid, "name": name})
    else:
        recipes_out.append({"index": 0, "id": "", "name": "single_recipe"})

    # ---------- 逐节点排程 ----------
    node_ops = defaultdict(list)
    for rec in res.op_records:
        # 兼容 5 元组和 8 元组
        nid = rec[0]
        mid = rec[1]
        bid = rec[2]
        s = rec[3]
        e = rec[4]
        temp = rec[5] if len(rec) >= 8 else 0
        heat = rec[6] if len(rec) >= 8 else 0
        mode = rec[7] if len(rec) >= 8 else 0

        node_ops[nid].append({
            "machine_id": mid,
            "machine_name": multi.machines[mid].note or f"machine_{mid}",
            "batch_id": bid,
            "start": round(s, 3),
            "end": round(e, 3),
            "duration": round(e - s, 3),
            "temp_c": temp,
            "heat_level": heat,
            "mode_code": mode,
        })

    schedule_out = []
    for nid in range(multi.N):
        ops = sorted(node_ops.get(nid, []), key=lambda x: x["start"])
        if not ops:
            continue

        node_start = min(o["start"] for o in ops)
        node_end = max(o["end"] for o in ops)

        # 菜谱归属
        if is_multi:
            rid_idx = multi.recipe_of_node[nid]
            recipe_id = multi.recipe_ids[rid_idx]
            recipe_name = multi.recipe_names[rid_idx]
            step_idx = sum(
                1 for j in range(nid)
                if multi.recipe_of_node[j] == rid_idx
            )
        else:
            recipe_id = ""
            recipe_name = "single_recipe"
            step_idx = nid

        # 前置 / 后继
        preds = list(multi.predecessors[nid]) if hasattr(multi, "predecessors") else []
        succs = list(multi.successors[nid]) if hasattr(multi, "successors") else []

        schedule_out.append({
            "node_id": nid,
            "recipe_id": recipe_id,
            "recipe_name": recipe_name,
            "step_index": step_idx,
            "start": round(node_start, 3),
            "end": round(node_end, 3),
            "duration": round(node_end - node_start, 3),
            "predecessors": preds,
            "successors": succs,
            "operations": ops,
        })

    # ---------- 按设备分组 ----------
    by_machine = defaultdict(list)
    for entry in schedule_out:
        for op in entry["operations"]:
            by_machine[str(op["machine_id"])].append({
                "node_id": entry["node_id"],
                "recipe_name": entry["recipe_name"],
                "step_index": entry["step_index"],
                "batch_id": op["batch_id"],
                "start": op["start"],
                "end": op["end"],
                "duration": op["duration"],
            })
    for k in by_machine:
        by_machine[k].sort(key=lambda x: x["start"])

    # ---------- 按菜谱分组 ----------
    by_recipe = defaultdict(list)
    for entry in schedule_out:
        by_recipe[entry["recipe_name"]].append({
            "node_id": entry["node_id"],
            "step_index": entry["step_index"],
            "start": entry["start"],
            "end": entry["end"],
            "duration": entry["duration"],
        })
    for name in by_recipe:
        by_recipe[name].sort(key=lambda x: x["step_index"])

    # ---------- 元信息 ----------
    meta = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "makespan": round(makespan, 3),
        "N": multi.N,
        "E": getattr(multi, "E", 0),
        "K": multi.K,
        "num_recipes": len(recipes_out),
        "num_operations": len(res.op_records),
    }
    if now is not None:
        meta["now"] = now
    if lb:
        meta["lb_cp"] = round(lb.get("lb_cp", 0), 3)
        meta["lb_machine"] = round(lb.get("lb_machine", 0), 3)
        meta["lb"] = round(lb.get("lb", 0), 3)
        if lb.get("lb", 0) > 0:
            meta["gap_pct"] = round(
                (makespan - lb["lb"]) / lb["lb"] * 100, 2
            )
            # 是否接近最优：gap < 5%
            meta["is_near_optimal"] = meta["gap_pct"] < 5.0
    if serial_makespan:
        meta["serial_makespan"] = round(serial_makespan, 3)
        meta["speedup"] = round(serial_makespan / max(1e-9, makespan), 3)

    # ---------- 组装 ----------
    out = {
        "meta": meta,
        "recipes": recipes_out,
        "machines": machines_out,
        "schedule": schedule_out,
        "by_machine": dict(by_machine),
        "by_recipe": dict(by_recipe),
    }
    # 自动预热不是菜谱节点，单独导出，不能伪造 node_id。
    out["device_preheat"] = [
        {**dict(p), "machine_name": multi.machines[int(p["machine_id"])].note}
        for p in getattr(res, "preheat_records", ()) or ()
    ]
    for p in out["device_preheat"]:
        key = str(p["machine_id"])
        out["by_machine"].setdefault(key, []).append({
            "node_id": None, "recipe_name": "设备热管理",
            "step_index": None, "batch_id": p["batch_id"],
            "start": p["start"], "end": p["end"],
            "duration": round(p["end"] - p["start"], 3),
            "step": "设备自动预热",
        })
        out["by_machine"][key].sort(key=lambda x: x["start"])
    out["meta"]["num_auto_preheat"] = len(out["device_preheat"])
    if notifications is not None:
        out["notifications"] = notifications


    # ---------- 保存 ----------
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(to_json_safe(out), f, ensure_ascii=False, indent=2)
        print(f"  ✓ 排程 JSON 已保存: {save_path}")

    return out



# ============================================================
# 单独导出中文解析文档
# ============================================================
def export_chinese_doc_json(
    multi,
    res,
    save_path=None,
    lb=None,
    serial_makespan=None,
    notifications=None,
    mode="multi",
):
    """
    生成中文解析文档并单独保存为 JSON 文件。

    与 export_schedule_json 的区别：
        - 内容以中文说明为主，不是调度数据结构
        - 面向人类阅读或报告使用
        - 单独存一个文件，不嵌入 schedule.json

    参数:
        multi             : MultiInstance 或 Instance
        res               : ScheduleResult 或 MultiScheduleResult
        save_path         : 保存路径；None 则只返回 dict
        lb                : 下界 dict
        serial_makespan   : 串行总耗时
        notifications     : 提醒列表（可选）
        mode              : "single" / "multi" / "reschedule"
    """
    makespan = float(res.makespan)
    is_multi = (hasattr(multi, "recipe_of_node")
                and hasattr(multi, "recipe_names")
                and multi.recipe_names)

    # ---------- 1. 文档头 ----------
    meta_doc = {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "模式": mode,
        "菜谱数": len(multi.recipe_names) if is_multi else 1,
        "步骤总数": multi.N,
        "依赖边数": getattr(multi, "E", 0),
        "设备台数": multi.K,
        "makespan（分钟）": round(makespan, 1),
    }
    if lb:
        meta_doc["关键路径下界"] = round(lb.get("lb_cp", 0), 1)
        meta_doc["机器负荷下界"] = round(lb.get("lb_machine", 0), 1)
        meta_doc["理论下界"] = round(lb.get("lb", 0), 1)
        if lb.get("lb", 0) > 0:
            gap = (makespan - lb["lb"]) / lb["lb"] * 100
            meta_doc["与下界差距"] = f"{gap:.1f}%"
    if serial_makespan and serial_makespan > 0:
        saved = serial_makespan - makespan
        speedup = serial_makespan / max(1e-9, makespan)
        meta_doc["串行总耗时"] = round(serial_makespan, 1)
        meta_doc["节省时间"] = round(saved, 1)
        meta_doc["加速比"] = round(speedup, 2)

    # ---------- 2. 概览文字 ----------
    lines = []
    lines.append(
        f"本次调度共涉及 {meta_doc['菜谱数']} 道菜谱，"
        f"总步骤 {multi.N} 步，依赖 {meta_doc['依赖边数']} 条，"
        f"设备 {multi.K} 台。"
    )
    lines.append(
        f"所有菜品总完成时间 {makespan:.1f} 分钟。"
    )
    if serial_makespan and serial_makespan > 0:
        lines.append(
            f"若逐道菜单独制作，串行总耗时 {serial_makespan:.1f} 分钟，"
            f"并行后节省 {saved:.1f} 分钟"
            f"（{saved / serial_makespan * 100:.1f}%），"
            f"加速比 {speedup:.2f} 倍。"
        )
    if lb and lb.get("lb", 0) > 0:
        lines.append(
            f"理论下界 {lb['lb']:.1f} 分钟，"
            f"当前方案与下界差距 {gap:.1f}%。"
        )
    overview_text = "\n".join(lines)

    # ---------- 3. 菜谱明细 ----------
    recipe_docs = []
    if is_multi:
        n_recipes = len(multi.recipe_names)
    else:
        n_recipes = 1

    for rid in range(n_recipes):
        if is_multi:
            recipe_name = multi.recipe_names[rid]
            recipe_id = (multi.recipe_ids[rid]
                         if rid < len(multi.recipe_ids) else "")
            nodes_in_recipe = [
                nid for nid in range(multi.N)
                if multi.recipe_of_node[nid] == rid
            ]
        else:
            recipe_name = "单菜"
            recipe_id = ""
            nodes_in_recipe = list(range(multi.N))

        nodes_in_recipe.sort(key=lambda x: res.node_start[x])

        steps_doc = []
        for seq, nid in enumerate(nodes_in_recipe):
            s = res.node_start[nid]
            e = res.node_end[nid]
            raw = multi.step_names[nid].split("#")[-1]
            step_name = raw if raw else f"步骤{seq+1}"

            ops = []
            for rec in res.op_records:
                nid_ = rec[0]
                if nid_ != nid:
                    continue
                mid = rec[1]
                bid = rec[2]
                op_s = rec[3]
                op_e = rec[4]
                ops.append((mid, bid, op_s, op_e))
            devices_desc = []
            for mid, bid, _, _ in ops:
                if mid == 7:
                    devices_desc.append("人工操作")
                else:
                    dev_name = machine_display_name(
                        multi.machines[mid],
                        mid=mid
                    )
                    devices_desc.append(dev_name)
            if not devices_desc:
                devices_desc = ["无"]

            # 前置依赖
            preds = (multi.predecessors[nid]
                     if hasattr(multi, "predecessors") else [])
            pred_names = []
            for p in preds:
                raw_p = multi.step_names[p].split("#")[-1]
                pred_names.append(raw_p or f"步骤{p+1}")

            steps_doc.append({
                "序号": seq + 1,
                "步骤名": step_name,
                "开始时间": round(s, 1),
                "结束时间": round(e, 1),
                "耗时": round(e - s, 1),
                "占用资源": devices_desc,
                "前置步骤": pred_names if pred_names else [],
                "是否需要人工": bool(
                    multi.needs_human[nid]
                    if hasattr(multi, "needs_human") and multi.needs_human
                    else False
                ),
            })

        recipe_docs.append({
            "菜谱名": recipe_name,
            "菜谱ID": recipe_id,
            "步骤数": len(nodes_in_recipe),
            "开始时间": round(min(
                (res.node_start[n] for n in nodes_in_recipe),
                default=0.0
            ), 1),
            "完成时间": round(max(
                (res.node_end[n] for n in nodes_in_recipe),
                default=0.0
            ), 1),
            "步骤明细": steps_doc,
        })

    # ---------- 4. 设备明细 ----------
    machine_docs = []
    for k in range(multi.K):
        m = multi.machines[k]
        name = m.note or f"设备{k}"
        ops = []
        for rec in res.op_records:
            mid_ = rec[1]
            if mid_ != k:
                continue
            nid = rec[0]
            bid = rec[2]
            s = rec[3]
            e = rec[4]
            ops.append((nid, bid, s, e))

        if not ops:
            machine_docs.append({
                "设备名": name,
                "编号": k,
                "容量": multi.capacities[k],
                "模式": getattr(m, "mode", "none"),
                "详细模式": getattr(m, "mode_detail", "none"),
                "总占用时长": 0.0,
                "利用率": "0%",
                "占用记录": [],
            })
            continue

        ops.sort(key=lambda x: x[2])
        total_busy = sum(e - s for _, _, s, e in ops)
        cap = max(1, multi.capacities[k])
        util = total_busy / (cap * makespan) if makespan > 0 else 0

        records = []
        for nid, bid, s, e in ops:
            rid = (multi.recipe_of_node[nid]
                   if is_multi else 0)
            rname = (multi.recipe_names[rid]
                     if is_multi else "单菜")
            raw = multi.step_names[nid].split("#")[-1]
            records.append({
                "菜谱": rname,
                "步骤": raw or f"N{nid}",
                "实例": bid,
                "开始": round(s, 1),
                "结束": round(e, 1),
                "时长": round(e - s, 1),
            })

        machine_docs.append({
            "设备名": name,
            "编号": k,
            "容量": multi.capacities[k],
            "模式": getattr(m, "mode", "none"),
            "详细模式": getattr(m, "mode_detail", "none"),
            "总占用时长": round(total_busy, 1),
            "利用率": f"{util * 100:.1f}%",
            "占用记录": records,
        })

    # ---------- 5. 同步完成度 ----------
    sync_doc = None
    if is_multi:
        recipe_finish = {}
        for nid in range(multi.N):
            rid = multi.recipe_of_node[nid]
            recipe_finish[rid] = max(
                recipe_finish.get(rid, 0.0),
                res.node_end[nid]
            )
        if recipe_finish:
            finishes = list(recipe_finish.values())
            diff = max(finishes) - min(finishes)
            sync_doc = {
                "各菜完成时间": {
                    multi.recipe_names[rid]: round(f, 1)
                    for rid, f in recipe_finish.items()
                },
                "最大时间差": round(diff, 1),
                "是否优秀": diff < 5.0,
                "评价": "优秀（<5分钟）" if diff < 5.0
                        else f"待优化（差 {diff:.1f} 分钟）",
            }

    # ---------- 6. 提醒清单 ----------
    reminder_docs = []
    if notifications:
        for r in notifications:
            reminder_docs.append({
                "级别": r.get("level", "info"),
                "触发时刻": round(
                    r.get("time", 0) - r.get("delay_min", 0), 1
                ),
                "菜谱": r.get("recipe", ""),
                "步骤": r.get("step", ""),
                "消息": r.get("message", ""),
            })

    # ---------- 7. 组装 ----------
    doc = {
        "文档信息": meta_doc,
        "概览": overview_text,
        "菜谱明细": recipe_docs,
        "设备明细": machine_docs,
    }
    if sync_doc:
        doc["同步完成度"] = sync_doc
    if reminder_docs:
        doc["关键节点提醒"] = reminder_docs

    # ---------- 8. 保存 ----------
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 中文解析文档已保存: {save_path}")

    return doc

def to_required_format(
    multi,
    res,
    ingredients_list=None,
    lb=None,
    serial_makespan=None,
    notifications=None,
):
    """
    转成赛题要求的 5 字段响应格式：
        overview / cookingTimeline / ingredientsSummary /
        detailTimeline / recipeDetail
    """
    from llm_parser import MODE_DETAIL_NAMES
    from collections import defaultdict

    makespan = float(res.makespan)
    is_multi = (hasattr(multi, "recipe_of_node")
                and hasattr(multi, "recipe_names")
                and multi.recipe_names)

    # ---------- 1. overview ----------
    overview = {
        "totalRecipes": len(multi.recipe_names) if is_multi else 1,
        "totalSteps": int(multi.N),
        "makespan": round(makespan, 1),
        "totalCookingTime": round(makespan, 1),
    }
    if serial_makespan:
        overview["serialMakespan"] = round(serial_makespan, 1)
        overview["speedup"] = round(
            serial_makespan / max(1e-9, makespan), 2
        )
    if lb:
        overview["lowerBound"] = round(lb.get("lb", 0), 1)

    # ---------- 2. cookingTimeline ----------
    # 按设备分组的工序时间轴
    by_machine = defaultdict(list)
    for rec in res.op_records:
        if len(rec) >= 8:
            nid, mid, bid, s, e, temp, heat, mode = rec[:8]
        else:
            nid, mid, bid, s, e = rec[:5]
            temp, heat, mode = 0, 0, 0

        rid = multi.recipe_of_node[nid] if is_multi else 0
        rname = multi.recipe_names[rid] if is_multi else "单菜"
        mname = multi.machines[mid].note or f"设备{mid}"

        by_machine[mname].append({
            "recipeName": rname,
            "machineName": mname,
            "batchId": int(bid),
            "modeDetail": MODE_DETAIL_NAMES.get(mode, "none"),
            "tempC": int(temp),
            "heatLevel": int(heat),
            "start": round(s, 1),
            "end": round(e, 1),
            "duration": round(e - s, 1),
        })

    cooking_timeline = []
    for mname, ops in by_machine.items():
        ops.sort(key=lambda x: x["start"])
        cooking_timeline.append({
            "machineName": mname,
            "operations": ops,
        })

    # ---------- 3. ingredientsSummary ----------
    ingredients_summary = []
    if ingredients_list:
        for rid, ings in enumerate(ingredients_list):
            if is_multi:
                rname = (multi.recipe_names[rid]
                         if rid < len(multi.recipe_names)
                         else f"菜谱{rid}")
            else:
                rname = "单菜"
            for ing in ings:
                ingredients_summary.append({
                    "recipeName": rname,
                    "name": ing.get("name", ""),
                    "quantity": ing.get("quantity", 0),
                    "unit": ing.get("unit", ""),
                })

    # ---------- 4. detailTimeline ----------
    detail_timeline = []
    for nid in range(multi.N):
        rid = multi.recipe_of_node[nid] if is_multi else 0
        rname = multi.recipe_names[rid] if is_multi else "单菜"
        step_name = (
            multi.step_names[nid].split("#")[-1]
            if hasattr(multi, "step_names") else f"S{nid}"
        )
        needs_human = bool(
            multi.needs_human[nid]
            if hasattr(multi, "needs_human") else False
        )

        # 找该步骤占用的设备
        devices = []
        for rec in res.op_records:
            if rec[0] != nid:
                continue
            mid = rec[1]
            mname = multi.machines[mid].note or f"设备{mid}"
            temp = rec[5] if len(rec) >= 8 else 0
            mode = rec[7] if len(rec) >= 8 else 0
            devices.append({
                "machineName": mname,
                "tempC": int(temp),
                "modeDetail": MODE_DETAIL_NAMES.get(mode, "none"),
            })

        detail_timeline.append({
            "recipeName": rname,
            "stepName": step_name,
            "start": round(res.node_start[nid], 1),
            "end": round(res.node_end[nid], 1),
            "duration": round(
                res.node_end[nid] - res.node_start[nid], 1
            ),
            "needsHuman": needs_human,
            "devices": devices,
        })

    # ---------- 5. recipeDetail ----------
    recipe_detail = []
    if is_multi:
        for rid in range(len(multi.recipe_names)):
            nodes = [nid for nid in range(multi.N)
                     if multi.recipe_of_node[nid] == rid]
            if not nodes:
                continue
            start = min(res.node_start[n] for n in nodes)
            end = max(res.node_end[n] for n in nodes)
            recipe_detail.append({
                "recipeName": multi.recipe_names[rid],
                "start": round(start, 1),
                "end": round(end, 1),
                "duration": round(end - start, 1),
                "stepCount": len(nodes),
            })
    else:
        recipe_detail.append({
            "recipeName": "单菜",
            "start": round(min(res.node_start), 1),
            "end": round(max(res.node_end), 1),
            "duration": round(makespan, 1),
            "stepCount": int(multi.N),
        })

    return {
        "overview": overview,
        "cookingTimeline": cooking_timeline,
        "ingredientsSummary": ingredients_summary,
        "detailTimeline": detail_timeline,
        "recipeDetail": recipe_detail,
    }

def to_competition_format(
    multi,
    res,
    ingredients_list=None,
    start_time=None,
):
    """
    转成赛事组要求的 5 字段格式：
        overview / cookingTimeline / ingredientsSummary /
        detailTimeline / recipeDetail

    start_time: datetime，默认用当前时间
    """
    from datetime import datetime, timedelta
    from collections import defaultdict
    from llm_parser import MODE_DETAIL_NAMES

    if start_time is None:
        start_time = datetime.now()

    makespan = float(res.makespan)
    is_multi = (hasattr(multi, "recipe_of_node")
                and hasattr(multi, "recipe_names")
                and multi.recipe_names)

    def fmt_time(offset_min):
        t = start_time + timedelta(minutes=float(offset_min))
        return t.strftime("%H:%M")

    # ============================================================
    # 1. overview
    # ============================================================
    # 串行总耗时
    serial_total = 0.0
    for rid in range(len(multi.recipe_names)):
        nodes = [n for n in range(multi.N)
                 if multi.recipe_of_node[n] == rid]
        if nodes:
            s = min(res.node_start[n] for n in nodes)
            e = max(res.node_end[n] for n in nodes)
            serial_total += (e - s)

    overview = {
        "finishTime": fmt_time(makespan),
        "timeSpent": str(int(round(makespan))),
        "timeSave": str(int(round(max(0.0, serial_total - makespan)))),
        "recipeCount": len(multi.recipe_names) if is_multi else 1,
    }

    # ============================================================
    # 2. cookingTimeline
    # ============================================================
    cooking_timeline = []
    n_recipes = len(multi.recipe_names) if is_multi else 1
    for rid in range(n_recipes):
        if is_multi:
            nodes = [n for n in range(multi.N)
                     if multi.recipe_of_node[n] == rid]
            rname = multi.recipe_names[rid]
        else:
            nodes = list(range(multi.N))
            rname = "单菜"
        if not nodes:
            continue
        s = min(res.node_start[n] for n in nodes)
        e = max(res.node_end[n] for n in nodes)

        # 找主设备（用时最长的设备，排除人工 id=7）
        dev_time = defaultdict(float)
        for rec in res.op_records:
            nid = rec[0]
            if nid not in nodes:
                continue
            mid = rec[1]
            if mid == 7:
                continue
            dev_time[mid] += (rec[4] - rec[3])

        if dev_time:
            main_mid = max(dev_time.keys(), key=lambda k: dev_time[k])
            product = multi.machines[main_mid].note or f"设备{main_mid}"
        else:
            product = "人工"

        cooking_timeline.append({
            "name": rname,
            "product": product,
            "startTime": fmt_time(s),
            "endTime": fmt_time(e),
            "timeSpent": str(int(round(e - s))),
        })

    # ============================================================
    # 3. ingredientsSummary
    # ============================================================
    # 食材分类（如 LLM 没输出 type，本地补）
    MEAT_KW = ["鸡", "鸭", "鹅", "鱼", "虾", "蟹", "贝", "鲍",
               "猪", "牛", "羊", "肉", "排骨", "培根", "火腿",
               "腊", "香肠", "蛋", "鸽", "蛙", "鳝", "鳗",
               "鱿", "墨", "海参", "鱼头", "鱼身", "鱼块"]
    SEASONING_KW = ["盐", "糖", "油", "酱", "醋", "酒", "胡椒",
                    "花椒", "辣椒", "粉", "精", "汁", "料",
                    "葱", "姜", "蒜", "香", "蜂蜜", "蚝",
                    "味", "腐乳", "料酒", "生抽", "老抽",
                    "香料", "黄油", "奶油", "淀粉", "芥末",
                    "豆豉", "剁椒", "辣椒油", "孜然", "肉桂",
                    "八角", "桂皮", "香叶", "小茴香", "草果"]

    def classify(name):
        # 优先用 LLM 给的 type
        # （在下方合并时判断）
        for kw in MEAT_KW:
            if kw in name:
                return "荤菜"
        for kw in SEASONING_KW:
            if kw in name:
                return "调味品"
        return "素菜"

    # 合并相同食材
    # merged[cat][(name, unit)] = total_quantity
    merged = defaultdict(lambda: defaultdict(float))
    for ings in (ingredients_list or []):
        for ing in ings:
            nm = str(ing.get("name", "")).strip()
            if not nm:
                continue
            qty = float(ing.get("quantity", 0) or 0)
            unit = str(ing.get("unit", "")).strip()
            llm_type = str(ing.get("type", "")).strip()
            if llm_type in ("荤菜", "素菜", "调味品"):
                cat = llm_type
            else:
                cat = classify(nm)
            merged[cat][(nm, unit)] += qty

    ingredients_summary = []
    for cat in ["荤菜", "素菜", "调味品"]:
        items = []
        for (nm, unit), total in merged[cat].items():
            if total <= 0:
                unit_str = unit if unit else "适量"
            elif total == int(total):
                unit_str = f"{int(total)}{unit}"
            else:
                unit_str = f"{total}{unit}"
            items.append({"name": nm, "unit": unit_str})
        if items:
            ingredients_summary.append({
                "type": cat,
                "list": items,
            })

    # ============================================================
    # 4. detailTimeline
    # ============================================================
    def classify_step_type(nid):
        """判断步骤 type 1~6"""
        step_name = (
            multi.step_names[nid].split("#")[-1]
            if hasattr(multi, "step_names") else ""
        )
        # 收集该步骤使用的设备
        devs = set()
        for rec in res.op_records:
            if rec[0] == nid:
                devs.add(rec[1])

        # 6: 出锅装盘
        if "装盘" in step_name or "出锅" in step_name \
                or "摆盘" in step_name or "撒" in step_name:
            return 6

        # 5: 蒸烤类（预热+蒸+烤）
        if 2 in devs or 3 in devs:
            return 5

        # 4: 炒制类
        if 1 in devs:
            if any(k in step_name for k in ["炒", "煎", "爆", "炸"]):
                return 4
            if any(k in step_name for k in ["炖", "煮", "焯", "煲"]):
                return 3
            return 4

        # 2: 腌制备料
        if any(k in step_name for k in ["腌", "调", "拌", "浸泡"]):
            return 2

        # 1: 食材处理
        return 1

    detail_timeline = []
    for nid in range(multi.N):
        s = res.node_start[nid]
        e = res.node_end[nid]
        if is_multi:
            rid = multi.recipe_of_node[nid]
            rname = multi.recipe_names[rid]
        else:
            rname = "单菜"

        step_name = (
            multi.step_names[nid].split("#")[-1]
            if hasattr(multi, "step_names") else f"S{nid}"
        )
        stype = classify_step_type(nid)

        # 收集该步骤的设备参数
        params = []
        recipe_names = []
        for rec in res.op_records:
            if rec[0] != nid:
                continue
            mid = rec[1]
            temp = rec[5] if len(rec) >= 8 else 0
            mode = rec[7] if len(rec) >= 8 else 0
            dur = rec[4] - rec[3]

            if mid == 7:
                continue   # 人工不写参数

            # 温度或模式名
            temp_val = temp if temp > 0 else MODE_DETAIL_NAMES.get(mode, "none")
            params.append({
                "recipeName": rname,
                "temperature": temp_val if temp > 0
                               else str(MODE_DETAIL_NAMES.get(mode, "")),
                "time": int(round(dur)),
            })
            recipe_names.append(rname)

        # 描述
        if stype == 1:
            desc = f"准备{step_name}，处理备用"
        elif stype == 2:
            desc = f"将食材{step_name}"
        elif stype == 6:
            desc = f"{step_name}"
        else:
            mname = ""
            for rec in res.op_records:
                if rec[0] == nid and rec[1] != 7:
                    mname = multi.machines[rec[1]].note or ""
                    break
            desc = f"将{step_name}放入{mname}，执行操作"

        entry = {
            "timeInterval": f"{int(round(s))}-{int(round(e))}",
            "title": step_name,
            "type": stype,
            "list": [desc],
        }
        if params:
            entry["recipeNames"] = list(dict.fromkeys(recipe_names))
            entry["parameters"] = params
        detail_timeline.append(entry)

    # 排序：先按 type 升序，再按 timeInterval 开始时间升序
    def sort_key(item):
        t = item["type"]
        iv = item["timeInterval"].split("-")
        s = int(iv[0]) if iv else 0
        return (t, s)
    detail_timeline.sort(key=sort_key)

    # ============================================================
    # 5. recipeDetail
    # ============================================================
    recipe_detail = []
    for rid in range(n_recipes):
        if is_multi:
            nodes = [n for n in range(multi.N)
                     if multi.recipe_of_node[n] == rid]
            rname = multi.recipe_names[rid]
        else:
            nodes = list(range(multi.N))
            rname = "单菜"
        if not nodes:
            continue

        # 食材拆分主料/辅料
        ings = []
        if ingredients_list and rid < len(ingredients_list):
            ings = ingredients_list[rid]

        major, minor = [], []
        for ing in ings:
            nm = str(ing.get("name", "")).strip()
            qty = float(ing.get("quantity", 0) or 0)
            unit = str(ing.get("unit", "")).strip()
            if not nm:
                continue
            if qty > 0:
                unit_str = (f"{int(qty)}{unit}"
                            if qty == int(qty) else f"{qty}{unit}")
            else:
                unit_str = unit if unit else "适量"
            item = {"name": nm, "unit": unit_str}

            # 判断主料/辅料
            llm_type = str(ing.get("type", "")).strip()
            cat = llm_type if llm_type in ("荤菜", "素菜", "调味品") \
                else classify(nm)
            if cat == "调味品":
                minor.append(item)
            else:
                major.append(item)

        # 步骤
        nodes.sort(key=lambda n: res.node_start[n])
        steps = []
        for nid in nodes:
            sname = (
                multi.step_names[nid].split("#")[-1]
                if hasattr(multi, "step_names") else f"步骤{nid}"
            )
            stype = classify_step_type(nid)

            # 描述
            if stype == 1:
                desc = f"准备{sname}，处理备用"
            elif stype == 2:
                desc = f"将食材{sname}"
            elif stype == 6:
                desc = f"{sname}"
            else:
                mname = ""
                for rec in res.op_records:
                    if rec[0] == nid and rec[1] != 7:
                        mname = multi.machines[rec[1]].note or ""
                        break
                # 找温度
                temp_str = ""
                for rec in res.op_records:
                    if rec[0] == nid and rec[1] != 7:
                        t = rec[5] if len(rec) >= 8 else 0
                        if t > 0:
                            temp_str = str(t)
                        break
                desc = f"将{sname}放入{mname}，设置{temp_str}℃" \
                    if temp_str else f"将{sname}放入{mname}"

            step = {"describe": desc}

            # 设备工作步骤加 cookingParameters
            for rec in res.op_records:
                if rec[0] != nid or rec[1] == 7:
                    continue
                mid = rec[1]
                temp = rec[5] if len(rec) >= 8 else 0
                mode = rec[7] if len(rec) >= 8 else 0
                dur = rec[4] - rec[3]
                step["cookingParameters"] = {
                    "mode": MODE_DETAIL_NAMES.get(mode, "none"),
                    "temperature": str(temp) if temp > 0
                                   else str(MODE_DETAIL_NAMES.get(mode, "")),
                    "time": str(int(round(dur))),
                }
                break

            steps.append(step)

        recipe_detail.append({
            "name": rname,
            "majorIngredients": major,
            "minorIngredients": minor,
            "cookingSteps": steps,
        })

    return {
        "overview": overview,
        "cookingTimeline": cooking_timeline,
        "ingredientsSummary": ingredients_summary,
        "detailTimeline": detail_timeline,
        "recipeDetail": recipe_detail,
    }