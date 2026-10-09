# -*- coding: utf-8 -*-
"""FT 网页接入层：放在项目根目录，运行 uvicorn ft_web_server:app --reload。
不修改原有 run.py / api.py / CP-SAT 调度器。
"""
from __future__ import annotations

import csv
import time
import threading
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
APP_HTML = ROOT / "ft_web_frontend.html"
CSV_PATH = ROOT / "recipes_100.csv"
_LOCK = threading.RLock()

app = FastAPI(title="FT 智能烹饪调度 · Web", version="1.0")


class CreateTaskRequest(BaseModel):
    recipe_ids: List[str] = Field(default_factory=list)


class ReplanRequest(BaseModel):
    now: float = Field(ge=0)
    add_recipe_ids: List[str] = Field(default_factory=list)


def _catalog():
    if not CSV_PATH.is_file():
        raise HTTPException(500, "未找到 recipes_100.csv，请将网页文件放在 FT 项目根目录")
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = csv.DictReader(fh)
        for row in rows:
            rid = (row.get("菜谱id") or row.get("recipe_id") or row.get("id") or "").strip()
            name = (row.get("名称") or row.get("name") or "").strip()
            if rid:
                yield {
                    "id": rid,
                    "name": name,
                    "ingredients": (row.get("食材清单") or row.get("ingredients") or "").strip(),
                    "steps": (row.get("烹饪步骤") or row.get("steps") or "").strip(),
                }


def _resolve(recipe_ids, *, already=()):
    ids = [str(v).strip() for v in recipe_ids]
    if not ids or any(not v for v in ids):
        raise HTTPException(400, "至少选择一道菜品")
    if len(set(ids)) != len(ids):
        raise HTTPException(400, "同一菜品不能重复添加")
    if set(ids) & set(already):
        raise HTTPException(400, "不能重复添加已经在制作任务中的菜品")
    catalog = {r["id"]: r for r in _catalog()}
    missing = [rid for rid in ids if rid not in catalog]
    if missing:
        raise HTTPException(404, "未找到菜品 ID：" + "、".join(missing))
    return [catalog[rid] for rid in ids]


def _parse(targets):
    from llm_parser import recipe_to_schedule_json, json_to_schedule_text
    from greedy_scheduler import parse_schedule_text
    inst, infos, names = [], [], []
    for dish in targets:
        try:
            data = recipe_to_schedule_json(
                dish["id"], dish["name"], dish["ingredients"], dish["steps"], verbose=False
            )
            text = json_to_schedule_text(data)
            obj = parse_schedule_text(text)
        except Exception as exc:
            raise HTTPException(502, f"解析《{dish['name']}》失败，请检查 LLM 配置或缓存：{exc}") from exc
        actual = [str(v.get("name") or f"S{i}").strip() for i, v in enumerate(data.get("steps", []))]
        if len(actual) != obj.N:
            raise HTTPException(422, f"《{dish['name']}》解析步骤名称数与节点数不一致")
        inst.append(obj)
        infos.append({"recipe_id": dish["id"], "name": dish["name"]})
        names.append(actual)
    return inst, infos, names


def _optimize(multi, snapshot=None):
    from multi_scheduler import optimize_lexicographic_schedule
    try:
        return optimize_lexicographic_schedule(
            multi, snapshot=snapshot, max_extra_minutes=0.0,
            stage1_seconds=12.0, stage2_seconds=12.0, verbose=False,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "ortools":
            raise HTTPException(503, "尚未安装 OR-Tools：请运行 python -m pip install ortools") from exc
        raise


def _validate(multi, result):
    from multi_scheduler import check_multi_schedule
    report = check_multi_schedule(multi, result)
    if report.get("conflicts") or report.get("dependency_violations"):
        raise HTTPException(422, "排程存在设备冲突或依赖违规，请检查菜谱 DAG")



def _ensure_serial_baseline(state):
    """缓存单菜独立调度的 makespan；新增菜品只计算新增部分。"""
    inst_list = list(state.get("inst_list") or ())
    if not inst_list:
        return None
    existing = [float(v) for v in (state.get("single_makespans") or ())]
    if len(existing) > len(inst_list):
        existing = []
    if len(existing) < len(inst_list):
        from greedy_scheduler import greedy_schedule
        for inst in inst_list[len(existing):]:
            try:
                existing.append(float(greedy_schedule(inst).makespan))
            except Exception:
                # 不因辅助统计失败而中断成功的正式排程；避免显示错误数字。
                state["single_makespans"] = existing
                state["serial_makespan"] = None
                return None
    state["single_makespans"] = existing
    state["serial_makespan"] = round(sum(existing), 6)
    return state["serial_makespan"]


def _longest_human_work(multi, result, min_rest_minutes=5.0):
    """按真实人工操作区间而非菜谱节点总时长统计最长连续工作时间。"""
    human_ids = set()
    for mid, machine in enumerate(multi.machines):
        mode = str(getattr(machine, "mode", "") or "").lower()
        label = str(getattr(machine, "note", "") or "")
        if mode == "human" or "人工" in label or "手动操作" in label:
            human_ids.add(mid)

    intervals = set()
    for rec in result.op_records:
        nid, mid, _bid, s, e = rec[:5]
        s, e = float(s), float(e)
        if int(mid) in human_ids and e > s + 1e-7:
            intervals.add((int(nid), s, e))

    if not intervals:
        return 0.0
    order = sorted((s, e) for _, s, e in intervals)
    longest = running = 0.0
    previous_end = None
    for start, end in order:
        if previous_end is None or start - previous_end >= min_rest_minutes - 1e-7:
            running = 0.0
        running += end - start
        longest = max(longest, running)
        previous_end = max(end, previous_end) if previous_end is not None else end
    return round(longest, 2)


def _response(state):
    multi, result = state["multi_obj"], state["current_res"]
    names = list(multi.recipe_names)
    colors = ["#4388E8", "#EC9953", "#49AE8A", "#AB7EDC", "#D8799A", "#7E95AB"]
    now = float(state.get("last_now", 0.0))

    recipes = []
    for rid, name in enumerate(names):
        nodes = [i for i in range(multi.N) if multi.recipe_of_node[i] == rid]
        recipes.append({
            "id": multi.recipe_ids[rid], "name": name, "rid": rid,
            "color": colors[rid % len(colors)], "steps": len(nodes),
            "start": round(min((result.node_start[i] for i in nodes), default=0), 2),
            "finish": round(max((result.node_end[i] for i in nodes), default=0), 2),
        })

    # 工序详情供图上 tooltip 和执行提示使用
    steps = []
    for nid in range(multi.N):
        rid = multi.recipe_of_node[nid]
        s, e = float(result.node_start[nid]), float(result.node_end[nid])
        steps.append({
            "id": nid, "rid": rid, "recipe": names[rid],
            "name": multi.step_names[nid].split("#")[-1],
            "start": round(s, 2), "end": round(e, 2),
            "human": bool(multi.needs_human[nid]),
            "status": "已完成" if e <= now else ("进行中" if s <= now < e else "待执行"),
        })

    # 将 op_records 同一 (machine,batch_id) 的多个菜品合并为一个共享块
    grouped = defaultdict(list)
    for rec in result.op_records:
        nid, mid, bid, s, e = rec[:5]
        if float(e) - float(s) > 1e-7:
            grouped[(int(mid), str(bid))].append(rec)

    from ft_equipment_adapter import machine_display_name

    machines = []
    for mid in range(multi.K):
        machines.append({
            "id": mid,
            "name": machine_display_name(
                multi.machines[mid], mid=mid
            ),
            "capacity": int(multi.capacities[mid]),
        })

    batches = []
    for (mid, bid), records in grouped.items():
        member_steps = [int(r[0]) for r in records]
        rids = sorted(set(multi.recipe_of_node[nid] for nid in member_steps))
        first = records[0]
        batches.append({
            "id": f"{mid}:{bid}", "machine": mid,
            "start": round(min(float(r[3]) for r in records), 2),
            "end": round(max(float(r[4]) for r in records), 2),
            "rids": rids, "shared": len(rids) > 1,
            "members": member_steps,
            "temperature": first[5] if len(first) >= 8 else 0,
            "heat": first[6] if len(first) >= 8 else 0,
            "mode": first[7] if len(first) >= 8 else 0,
            "label": " · ".join(steps[nid]["name"] for nid in member_steps[:2]),
        })
    # 自动预热是设备真实占用，也必须进入网页甘特图。
    for p in getattr(result, "preheat_records", ()) or ():
        rids = list(p.get("recipe_ids") or [0])
        batches.append({
            "id": f"{int(p['machine_id'])}:{p['batch_id']}",
            "machine": int(p["machine_id"]),
            "start": round(float(p["start"]), 2),
            "end": round(float(p["end"]), 2),
            "rids": rids, "shared": False, "members": [],
            "temperature": p["temp"], "heat": p["heat"], "mode": p["mode"],
            "label": "设备自动预热", "auto_preheat": True,
        })
    batches.sort(key=lambda b: (b["machine"], b["start"]))

    # 三类事件：提前提醒、操作开始、无人值守结束。
    # 前端依据“当前执行分钟”进行一次性弹窗，避免服务端重复推送。
    notices = []
    for st in steps:
        if st["end"] <= st["start"]:
            continue
        if st["human"]:
            if st["start"] >= 3:
                notices.append({
                    "key": f"warn-{st['id']}", "at": round(st["start"] - 3, 2),
                    "level": "warning", "title": "即将开始",
                    "message": f"3 分钟后：{st['recipe']} · {st['name']}", "node": st["id"],
                })
            notices.append({
                "key": f"start-{st['id']}", "at": st["start"],
                "level": "action", "title": "该动手了",
                "message": f"请开始：{st['recipe']} · {st['name']}", "node": st["id"],
            })
        else:
            notices.append({
                "key": f"end-{st['id']}", "at": st["end"],
                "level": "info", "title": "步骤完成",
                "message": f"{st['recipe']} · {st['name']} 已到计划结束时间", "node": st["id"],
            })
    notices.sort(key=lambda x: (x["at"], x["key"]))

    finish_list = [r["finish"] for r in recipes]
    serial = state.get("serial_makespan")
    if serial is not None:
        serial = float(serial)
    settings = getattr(result, "optimize_info", None) or {}
    rest_minutes = float(settings.get("human_min_rest_minutes") or 5.0)
    human_longest = _longest_human_work(multi, result, rest_minutes)
    return {
        "task_id": state["task_id"], "round": int(state.get("round", 0)),
        "now": now, "recipes": recipes, "steps": steps,
        "machines": machines, "batches": batches, "notices": notices,
        "metrics": {
            "makespan": round(float(result.makespan), 2),
            "finish_gap": round(max(finish_list) - min(finish_list), 2) if finish_list else 0,
            "steps": multi.N, "recipes": len(recipes),
            "shared_batches": sum(b["shared"] for b in batches),
            "serial_makespan": round(serial, 2) if serial is not None else None,
            "time_saved": round(serial - float(result.makespan), 2) if serial is not None else None,
            "human_longest_work": human_longest,
            "human_rest_minutes": rest_minutes,
            "last_elapsed_seconds": state.get("last_elapsed_seconds"),
            "total_elapsed_seconds": state.get("total_elapsed_seconds"),
        },
        "optimization": getattr(result, "optimize_info", {}),
    }


@app.get("/")
def home():
    if not APP_HTML.exists():
        raise HTTPException(500, "未找到 ft_web_frontend.html")
    return FileResponse(APP_HTML, media_type="text/html; charset=utf-8")


@app.get("/api/recipes")
def search_recipes(q: str = "", limit: int = Query(default=30, ge=1, le=100)):
    keyword = q.strip().casefold()
    found = [
        {"id": r["id"], "name": r["name"]}
        for r in _catalog()
        if not keyword or keyword in r["id"].casefold() or keyword in r["name"].casefold()
    ][:limit]
    return {"items": found, "total": len(found)}


@app.post("/api/tasks")
def create_task(request: CreateTaskRequest):
    started_at = time.perf_counter()
    from multi_scheduler import merge_instances
    import run as ft_run
    targets = _resolve(request.recipe_ids)
    if len(targets) > 5:
        raise HTTPException(400, "目前每个制作任务最多支持5道菜")

    with _LOCK:
        inst, infos, names = _parse(targets)
        multi = merge_instances(inst, infos, step_names_list=names)
        result = _optimize(multi)
        _validate(multi, result)

        task_id = ft_run.create_task_id()
        state = {
            "task_id": task_id, "round": 0, "last_now": 0.0,
            "multi_obj": multi, "current_res": result,
            "inst_list": inst, "recipe_infos": infos,
            "step_names_list": names,
            "recipe_ids": [t["id"] for t in targets],
            "recipe_names": [t["name"] for t in targets],
            "history": [],
        }
        _ensure_serial_baseline(state)
        elapsed = round(time.perf_counter() - started_at, 3)
        state["last_elapsed_seconds"] = elapsed
        state["total_elapsed_seconds"] = elapsed
        ft_run.save_task_state(task_id, state)
        return _response(state)


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    import run as ft_run
    with _LOCK:
        state = ft_run.load_task_state(task_id)
        if state is None:
            raise HTTPException(404, "任务不存在，可能已被清理")
        if state.get("finished"):
            raise HTTPException(410, "该制作任务已结束，请重新选择菜品")
        before = len(state.get("single_makespans") or ())
        _ensure_serial_baseline(state)
        if len(state.get("single_makespans") or ()) > before:
            ft_run.save_task_state(task_id, state)
        return _response(state)


@app.post("/api/tasks/{task_id}/reschedule")
def replan_task(task_id: str, request: ReplanRequest):
    started_at = time.perf_counter()
    import run as ft_run

    with _LOCK:
        state = ft_run.load_task_state(task_id)
        if state is None:
            raise HTTPException(404, "任务不存在")
        if state.get("finished"):
            raise HTTPException(409, "该制作任务已结束，不能继续动态重排")
        from multi_scheduler import merge_instances
        from rescheduler import dynamic_reschedule
        if request.now + 1e-6 < float(state.get("last_now", 0.0)):
            raise HTTPException(400, "动态重排时间不能早于上一次重排时间")

        new_targets = (
            _resolve(request.add_recipe_ids, already=state.get("recipe_ids", []))
            if request.add_recipe_ids else []
        )
        if len(state["recipe_ids"]) + len(new_targets) > 5:
            raise HTTPException(400, "一个制作任务最多支持5道菜")

        multi = state["multi_obj"]
        old = state["current_res"]
        inst = list(state["inst_list"])
        infos = list(state["recipe_infos"])
        names = list(state["step_names_list"])
        baseline = old

        if new_targets:
            add_inst, add_infos, add_names = _parse(new_targets)
            inst += add_inst
            infos += add_infos
            names += add_names
            multi = merge_instances(inst, infos, step_names_list=names)
            # 新节点必须在 now 后进入 PENDING，不能假装在过去执行过。
            old_n = len(old.node_start)
            if multi.N < old_n:
                raise HTTPException(422, "新增菜品后节点数异常，请检查合并结果")
            placeholder = float(request.now) + 0.001
            baseline = SimpleNamespace(
                node_start=list(old.node_start) + [placeholder] * (multi.N - old_n),
                node_end=list(old.node_end) + [placeholder] * (multi.N - old_n),
                op_records=list(old.op_records),
                preheat_records=[dict(x) for x in (getattr(old, "preheat_records", None) or ())],
                optimize_info=dict(getattr(old, "optimize_info", None) or {}),
                makespan=max(float(old.makespan), float(request.now)),
                algo_time_ms=float(getattr(old, "algo_time_ms", 0)),
            )

        # 出错前绝不更新已存状态；让错误在前端显式显示。
        try:
            result = dynamic_reschedule(multi, baseline, now=float(request.now), verbose=False)
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(422, f"动态重排失败：{exc}") from exc

        optimized = result["new_res"]
        _validate(multi, optimized)
        state.update({
            "multi_obj": multi, "current_res": optimized,
            "inst_list": inst, "recipe_infos": infos,
            "step_names_list": names,
            "recipe_ids": list(state["recipe_ids"]) + [t["id"] for t in new_targets],
            "recipe_names": list(state["recipe_names"]) + [t["name"] for t in new_targets],
            "round": int(state.get("round", 0)) + 1,
            "last_now": float(request.now),
        })
        state.setdefault("history", []).append({
            "round": state["round"], "now": float(request.now),
            "added": [t["id"] for t in new_targets],
            "makespan": optimized.makespan,
        })
        _ensure_serial_baseline(state)
        elapsed = round(time.perf_counter() - started_at, 3)
        state["last_elapsed_seconds"] = elapsed
        previous_elapsed = state.get("total_elapsed_seconds")
        state["total_elapsed_seconds"] = round(float(previous_elapsed or 0.0) + elapsed, 3)
        ft_run.save_task_state(task_id, state)
        return _response(state)




@app.post("/api/tasks/{task_id}/finish")
def finish_task(task_id: str):
    """将任务标记为结束，保留历史状态但禁止恢复或动态重排。"""
    import run as ft_run
    from datetime import datetime, timezone
    with _LOCK:
        state = ft_run.load_task_state(task_id)
        if state is None:
            raise HTTPException(404, "任务不存在")
        if not state.get("finished"):
            state["finished"] = True
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            ft_run.save_task_state(task_id, state)
        return {"ok": True, "task_id": task_id, "finished": True}

@app.get("/health")
def health():
    return {"ok": True, "catalog_exists": CSV_PATH.exists(), "html_exists": APP_HTML.exists()}
