# run.py
# -*- coding: utf-8 -*-
"""
菜谱智能调度 - 单文件完整版

用法：只改 __main__ 里的 RECIPE_ID，然后运行。
"""

import os
import csv
import io
import sys
import time
import json
import contextlib
import traceback
import uuid
import pickle
from datetime import datetime
from types import SimpleNamespace

from multi_scheduler import run_multi
from llm_parser import recipe_to_schedule_text
from greedy_scheduler import parse_schedule_text
from json_utils import to_json_safe, _json_default
import dataclasses
from llm_parser import recipe_to_schedule_text
from greedy_scheduler import parse_schedule_text
from gantt_plot import plot_gantt_matplotlib
from llm_parser import recipe_to_schedule_text
from greedy_scheduler import (
    parse_schedule_text,
    greedy_schedule,
    print_gantt,
    print_time_table,
)

# ============================================================
# 依赖你自己已有的两个模块
# 如果不想依赖，把 llm_parser 和 greedy_scheduler 的代码
# 也贴到本文件即可
# ============================================================
from llm_parser import recipe_to_schedule_text
from greedy_scheduler import (
    parse_schedule_text,
    greedy_schedule,
    print_gantt,
    print_time_table,
)


# ============================================================
# 全局配置
# ============================================================
CSV_PATH = "recipes_100.csv"
RESULT_DIR = "result"
BAR_WIDTH = 60
ENCODING = "utf-8-sig"
TASK_STATE_DIR = os.path.join(RESULT_DIR, "task_states")


# ============================================================
# 工具函数
# ============================================================
def safe_filename(s: str) -> str:
    keep = "-_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keep else "_" for c in s)

def write_text(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hr(title: str = "", width: int = 78, ch: str = "="):
    if title:
        print(f"\n{ch * width}\n{title}\n{ch * width}")
    else:
        print(ch * width)


def capture_print(func, *args, **kwargs) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        func(*args, **kwargs)
    return buf.getvalue()


# ============================================================
# 读取 CSV
# ============================================================
def read_recipes(csv_path: str = CSV_PATH) -> list:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到 CSV: {csv_path}")

    recipes = []
    with open(csv_path, "r", encoding=ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        id_key = next((k for k in ("菜谱id", "recipe_id", "id", "ID")
                       if k in fields), None)
        if id_key is None:
            raise ValueError(f"CSV 缺少菜谱 ID 列，实际表头: {fields}")

        name_key = "名称" if "名称" in fields else "name"
        ing_key = "食材清单" if "食材清单" in fields else "ingredients"
        step_key = "烹饪步骤" if "烹饪步骤" in fields else "steps"

        for i, row in enumerate(reader, 1):
            rid = (row.get(id_key) or "").strip()
            if not rid:
                continue
            recipes.append({
                "recipe_id":   rid,
                "name":        (row.get(name_key) or "").strip(),
                "ingredients": (row.get(ing_key) or "").strip(),
                "steps":       (row.get(step_key) or "").strip(),
                "row_no":      i,
            })
    return recipes


def find_recipe(recipes: list, rid: str):
    for r in recipes:
        if r["recipe_id"] == rid:
            return r
    return None


# ============================================================
# 主流程函数（就是它没被定义导致的报错）
# ============================================================
def run(recipe_id: str,
        csv_path: str = CSV_PATH,
        bar_width: int = BAR_WIDTH,
        save: bool = True) -> dict:
    """
    指定菜谱 ID，跑完整流程：
        CSV 读取 -> 定位菜谱 -> LLM 解析 -> 解析数据
        -> 贪心调度 -> 甘特图 -> 详细流程 -> 落盘
    """
    t_total_0 = time.perf_counter()
    os.makedirs(RESULT_DIR, exist_ok=True)

    hr("#" * 78, ch="#")
    print("# 菜谱智能调度")
    print(f"# 菜谱 ID  : {recipe_id}")
    print(f"# 启动时间 : {timestamp()}")
    print(f"# CSV 文件 : {csv_path}")
    hr("#" * 78, ch="#")

    # ---------- 0. 读取 CSV，找到菜谱 ----------
    print(f"\n[0/4] 读取 CSV 并定位菜谱...")
    try:
        recipes = read_recipes(csv_path)
    except Exception as e:
        print(f"  ✗ 读取 CSV 失败: {e}")
        return {"success": False, "error": f"csv: {e}"}

    recipe = find_recipe(recipes, recipe_id)
    if recipe is None:
        print(f"  ✗ 未找到菜谱 ID: {recipe_id}")
        print(f"  可用 ID 前 5 个:")
        for r in recipes[:5]:
            print(f"    {r['recipe_id']}  {r['name']}")
        return {"success": False, "error": "recipe not found"}

    print(f"  ✓ 共 {len(recipes)} 条菜谱，已找到:")
    print(f"    名称     : {recipe['name']}")
    print(f"    食材     : {recipe['ingredients'][:60]}...")
    print(f"    步骤长度 : {len(recipe['steps'])} 字符")

    # ---------- 1. LLM 解析 ----------
    print(f"\n[1/4] 调用大模型解析菜谱...")
    t0 = time.perf_counter()
    try:
        llm_text = recipe_to_schedule_text(
            recipe_id=recipe["recipe_id"],
            name=recipe["name"],
            ingredients=recipe["ingredients"],
            steps=recipe["steps"],
            stream=True,
            verbose=True,
        )
    except Exception as e:
        print(f"  ✗ LLM 调用失败: {e}")
        traceback.print_exc()
        return {"success": False, "error": f"llm: {e}"}

    llm_ms = (time.perf_counter() - t0) * 1000
    print(f"  ✓ LLM 耗时 {llm_ms:.0f} ms，输出 {len(llm_text)} 字符")

    llm_file = os.path.join(RESULT_DIR, f"{safe_filename(recipe_id)}_llm.txt")
    if save:
        write_text(llm_file, llm_text)
        print(f"  → 保存: {llm_file}")

    # ---------- 2. 解析成调度实例 ----------
    print(f"\n[2/4] 解析调度数据...")
    t0 = time.perf_counter()
    try:
        inst = parse_schedule_text(llm_text)
    except Exception as e:
        print(f"  ✗ 解析失败: {e}")
        print(f"  LLM 输出前 600 字符:\n{llm_text[:600]}")
        return {"success": False, "error": f"parse: {e}"}

    parse_ms = (time.perf_counter() - t0) * 1000
    print(f"  ✓ N={inst.N}  E={inst.E}  K={inst.K}  "
          f"耗时 {parse_ms:.3f} ms")

    # ---------- 3. 贪心调度 ----------
    print(f"\n[3/4] 堆栈贪心调度...")
    try:
        res = greedy_schedule(inst)
    except Exception as e:
        print(f"  ✗ 调度失败: {e}")
        traceback.print_exc()
        return {"success": False, "error": f"sched: {e}"}

    print(f"  ✓ makespan={res.makespan} 分钟，"
          f"耗时 {res.algo_time_ms:.3f} ms")

    # ---------- 4. 甘特图 + 详细表 ----------
    print(f"\n[4/4] 生成甘特图与详细流程...")

    # 4.1 ASCII 甘特图（终端展示）
    gantt_text = capture_print(print_gantt, inst, res, bar_width=bar_width)
    table_text = capture_print(print_time_table, inst, res)

    # 4.2 matplotlib 甘特图（保存 PNG）
    png_file = ""
    if save:
        png_file = os.path.join(
            RESULT_DIR, f"{safe_filename(recipe_id)}_gantt.png"
        )
        try:
            plot_gantt_matplotlib(
                inst, res,
                save_path=png_file,
                show=False,
                figsize=(14, 8),
                dpi=200,
                title=f"做菜甘特图  {recipe['name']}  "
                      f"makespan={res.makespan} 分钟",
            )
        except Exception as e:
            print(f"  ✗ matplotlib 甘特图失败: {e}")
            traceback.print_exc()
            png_file = ""


    # 导出结构化排程 JSON
    if save:
        from schedule_exporter import export_schedule_json
        schedule_json = os.path.join(
            RESULT_DIR, f"{safe_filename(recipe_id)}_schedule.json"
        )
        export_schedule_json(
            inst, res,
            save_path=schedule_json,
            mode="single",
        )

    total_ms = (time.perf_counter() - t_total_0) * 1000

    summary_text = (
        f"{'=' * 70}\n"
        f"运行摘要\n"
        f"{'=' * 70}\n"
        f"菜谱 ID    : {recipe_id}\n"
        f"菜谱名称   : {recipe['name']}\n"
        f"节点数 N   : {inst.N}\n"
        f"依赖边 E   : {inst.E}\n"
        f"设备种类 K : {inst.K}\n"
        f"makespan   : {res.makespan} 分钟\n"
        f"LLM 耗时   : {llm_ms:.1f} ms\n"
        f"解析耗时   : {parse_ms:.3f} ms\n"
        f"调度耗时   : {res.algo_time_ms:.3f} ms\n"
        f"总耗时     : {total_ms:.1f} ms\n"
        f"{'=' * 70}\n"
    )

    print(gantt_text)
    print(table_text)
    print(summary_text)

    log_file = ""
    if save:
        log_content = (
            f"{'#' * 78}\n"
            f"# 菜谱: {recipe['name']}  (id={recipe_id})\n"
            f"# 时间: {timestamp()}\n"
            f"{'#' * 78}\n\n"
            f"【LLM 原始输出】\n{llm_text}\n\n"
            f"{gantt_text}\n{table_text}\n{summary_text}\n"
        )
        log_file = os.path.join(
            RESULT_DIR, f"{safe_filename(recipe_id)}_log.txt"
        )
        write_text(log_file, log_content)
        print(f"[结果] 日志已保存: {log_file}")

    return {
        "success": True,
        "recipe_id": recipe_id,
        "name": recipe["name"],
        "makespan": res.makespan,
        "llm_ms": llm_ms,
        "parse_ms": parse_ms,
        "algo_ms": res.algo_time_ms,
        "total_ms": total_ms,
        "llm_file": llm_file,
        "log_file": log_file,
        "png_file": png_file,  # 新增
    }



# ============================================================
# task_id / 多轮动态重排状态管理
# ============================================================
def create_task_id() -> str:
    """生成一次完整调度任务的唯一 ID。"""
    return f"cook_{uuid.uuid4().hex[:12]}"


def get_task_state_path(task_id: str) -> str:
    """返回 task_id 对应的本地状态文件路径。"""
    if not task_id or not str(task_id).strip():
        raise ValueError("task_id 不能为空")
    os.makedirs(TASK_STATE_DIR, exist_ok=True)
    return os.path.join(TASK_STATE_DIR, f"{safe_filename(str(task_id))}.pkl")


def save_task_state(task_id: str, state: dict) -> str:
    """
    保存任务状态。

    这里使用 pickle 是因为 multi_obj / res 是 Python 调度对象，不能直接 JSON 序列化。
    状态文件仅供本服务自身读写，不要加载外部不可信 pickle 文件。
    """
    path = get_task_state_path(task_id)
    tmp = path + ".tmp"
    state["task_id"] = task_id
    state["updated_at"] = timestamp()
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def load_task_state(task_id: str):
    """读取 task_id 的最新状态；不存在时返回 None。"""
    path = get_task_state_path(task_id)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        state = pickle.load(f)
    if not isinstance(state, dict):
        raise ValueError(f"task_id={task_id} 状态文件格式错误")
    if state.get("task_id") not in (None, task_id):
        raise ValueError(f"task_id={task_id} 与状态文件不匹配")
    state["task_id"] = task_id
    return state



def build_initial_task_state(task_id: str, result: dict, targets: list) -> dict:
    """保存初始任务状态，同时缓存每道菜的串行基线。"""
    return {
        "task_id": task_id,
        "round": 0,
        "created_at": timestamp(),
        "updated_at": timestamp(),
        "multi_obj": result["multi"],
        "current_res": result["res"],
        "inst_list": result.get("inst_list", []),
        "recipe_infos": result.get("recipe_infos", []),
        "step_names_list": result.get("step_names_list", []),
        "ingredients_list": result.get("ingredients_list"),
        "single_makespans": [
            float(x)
            for x in result.get("single_makespans", [])
        ],
        "serial_makespan": result.get("serial_makespan"),
        "lb": result.get("lb"),
        "recipe_ids": [t["recipe_id"] for t in targets],
        "recipe_names": [t["name"] for t in targets],
        "history": [],
    }



def state_to_result(state: dict) -> dict:
    """
    将持久化状态映射成当前 run.py 后续导出逻辑需要的 result 结构。
    注意：这是恢复任务，不会重新调用 LLM / run_multi。
    """
    return {
        "multi": state["multi_obj"],
        "res": state["current_res"],
        "inst_list": state.get("inst_list", []),
        "recipe_infos": state.get("recipe_infos", []),
        "step_names_list": state.get("step_names_list", []),
        "ingredients_list": state.get("ingredients_list"),
        "serial_makespan": state.get("serial_makespan"),
        "lb": state.get("lb"),
    }


def make_api_payload(task_id: str,
                     round_no: int,
                     required: dict,
                     notifications=None) -> dict:
    """
    接口外层携带 task_id / round；原有 5 字段保持在 data 内，避免破坏内部格式。
    """
    return {
        "success": True,
        "task_id": task_id,
        "round": int(round_no),
        "notifications": list(notifications or []),
        "data": required,
    }

# run.py: 新增两个完整函数（无需替换任何调度器主体函数）

def expand_schedule_preserving_preheats(old_res, new_node_count, now):
    """动态加菜时扩展旧排程，完整继承历史预热和调度元数据。"""
    from copy import deepcopy
    from types import SimpleNamespace

    old_count = len(old_res.node_start)
    new_node_count = int(new_node_count)
    now = float(now)
    if new_node_count < old_count:
        raise ValueError(f"新增菜品后节点数减少：{old_count} -> {new_node_count}")
    if len(old_res.node_end) != old_count:
        raise ValueError("旧排程 node_start/node_end 长度不一致")
    if now < 0:
        raise ValueError("动态重排时间不能为负")

    preheats = deepcopy(list(getattr(old_res, "preheat_records", None) or []))
    info = deepcopy(dict(getattr(old_res, "optimize_info", None) or {}))
    expected_count = int(info.get("auto_preheat_count", 0) or 0)
    if expected_count and not preheats:
        raise RuntimeError(
            "旧任务记录过设备自动预热，但 current_res.preheat_records 为空；"
            "预热数据已在上游丢失。请检查初始任务状态的保存，"
            "并用修复后的代码重新创建 task_id。"
        )

    for i, p in enumerate(preheats):
        for key in ("machine_id", "start", "end", "temp", "heat", "mode"):
            if key not in p:
                raise ValueError(f"第{i}条预热记录缺少字段 {key}")
        if float(p["end"]) <= float(p["start"]):
            raise ValueError(f"第{i}条预热记录的时间区间无效：{p}")

    # take_snapshot 的规则为 start > now 才算 pending。
    # placeholder 只用于新增节点，不是真实工序执行时间。
    placeholder = now + 1e-3
    extra = new_node_count - old_count
    result = SimpleNamespace(
        node_start=list(old_res.node_start) + [placeholder] * extra,
        node_end=list(old_res.node_end) + [placeholder] * extra,
        op_records=deepcopy(list(old_res.op_records)),
        preheat_records=preheats,
        optimize_info=info,
        makespan=max(float(old_res.makespan), now),
        algo_time_ms=float(getattr(old_res, "algo_time_ms", 0.0)),
    )
    print(
        f"[预热继承] 旧记录={len(preheats)}，"
        f"已开始={sum(float(p['start']) < now for p in preheats)}，"
        f"未来待重排={sum(float(p['start']) >= now for p in preheats)}"
    )
    return result


def verify_dynamic_preheat_history(before, after, now):
    """重排完成后验证所有已经开始的自动预热记录没有丢失。"""
    from collections import Counter

    def signature(p):
        return (
            int(p["machine_id"]),
            round(float(p["start"]), 6),
            round(float(p["end"]), 6),
            str(p.get("temp")),
            str(p.get("heat")),
            str(p.get("mode")),
        )

    historical = [
        p for p in (getattr(before, "preheat_records", None) or [])
        if float(p["start"]) < float(now) - 1e-8
    ]
    produced = list(getattr(after, "preheat_records", None) or [])
    missing = Counter(signature(p) for p in historical) - Counter(
        signature(p) for p in produced
    )
    if missing:
        raise AssertionError(
            "动态重排丢失了已经执行的设备预热记录："
            + repr(list(missing.elements()))
            + "。请检查 take_snapshot() 和 "
            "reschedule_minimal_perturbation() 的 preheat_records 传递。"
        )
    print(
        f"[预热校验] 历史已开始={len(historical)}条，"
        f"重排最终记录={len(produced)}条，历史预热全部保留"
    )
    return True



def refresh_final_schedule_analysis(
    state, task_id=None, show=False, save_dir=None
):
    """
    快速刷新最终统计：
    1. 同一轮排程不重复计算
    2. 串行基线只计算新增菜品
    3. 输出时直接复用结果
    4. 支持多轮动态重排
    """
    import json
    import os
    import time
    from multi_scheduler import (
        check_multi_schedule,
        compute_lower_bound,
        print_comparison,
    )
    from greedy_scheduler import greedy_schedule

    t0 = time.perf_counter()

    multi = state["multi_obj"]
    res = state["current_res"]
    round_no = int(state.get("round", 0))

    # 排程签名：用于判断统计是否需要刷新
    signature = (
        round_no,
        int(multi.N),
        int(multi.E),
        round(float(res.makespan), 6),
        tuple(round(float(x), 6) for x in res.node_start),
        tuple(round(float(x), 6) for x in res.node_end),
        tuple(
            (
                r[0], r[1], str(r[2]),
                round(float(r[3]), 6),
                round(float(r[4]), 6),
            )
            for r in res.op_records
        ),
    )

    cache = state.get("_analysis_cache") or {}

    if (
        cache.get("signature") == signature
        and state.get("final_analysis")
    ):
        summary = state["final_analysis"]
        lb = state["lb"]
        check = state["check"]
        serial = state.get("serial_makespan")
        reused = True

    else:
        reused = False

        # 每轮新排程只计算一次
        lb = compute_lower_bound(multi)
        check = check_multi_schedule(multi, res)

        instances = list(state.get("inst_list") or [])
        serial = None

        if (
            instances
            and len(instances) == len(multi.recipe_names)
        ):
            baseline = [
                float(x)
                for x in state.get("single_makespans", [])
            ]

            if len(baseline) > len(instances):
                baseline = []

            # 已有菜品直接复用，新增菜品单独计算
            for idx in range(len(baseline), len(instances)):
                r = greedy_schedule(instances[idx])
                baseline.append(float(r.makespan))

            state["single_makespans"] = baseline
            serial = sum(baseline)

        state["lb"] = lb
        state["check"] = check
        state["serial_makespan"] = serial

        finish = {}

        for nid, rid in enumerate(multi.recipe_of_node):
            finish[rid] = max(
                finish.get(rid, 0.0),
                float(res.node_end[nid]),
            )

        spread = (
            max(finish.values()) - min(finish.values())
            if finish else 0.0
        )

        history = state.get("history") or []

        now = (
            float(history[-1]["now"])
            if history else None
        )

        optimization = (
            getattr(res, "optimize_info", None) or {}
        )

        summary = {
            "task_id": task_id or state.get("task_id"),
            "round": round_no,
            "recipe_count": len(multi.recipe_names),
            "recipe_names": list(multi.recipe_names),
            "N": multi.N,
            "E": multi.E,
            "K": multi.K,
            "makespan": float(res.makespan),
            "serial_makespan": serial,
            "lb_cp": lb["lb_cp"],
            "lb_machine": lb["lb_machine"],
            "lb": lb["lb"],
            "device_conflicts": len(check["conflicts"]),
            "dependency_violations": len(
                check["dependency_violations"]
            ),
            "recipe_finish_times": {
                multi.recipe_names[rid]: value
                for rid, value in finish.items()
            },
            "finish_spread": spread,
            "now": now,
            "remaining_minutes": (
                max(0.0, float(res.makespan) - now)
                if now is not None else None
            ),
            "optimize_info": optimization,
        }

        state["final_analysis"] = summary
        state["_analysis_cache"] = {
            "signature": signature,
        }

    # 需要导出时才写入文件
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        path = os.path.join(
            save_dir, "multi_summary.json"
        )

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                summary,
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

    # 只有程序最后才完整打印
    if show:
        print("\n" + "#" * 72)
        print(
            f"# 最终调度分析 "
            f"task={summary['task_id']} "
            f"round={round_no}"
        )
        print("#" * 72)

        if summary["now"] is not None:
            print(
                f"  重排时刻     : "
                f"{summary['now']:.1f} 分钟"
            )
            print(
                f"  剩余调度时间 : "
                f"{summary['remaining_minutes']:.1f} 分钟"
            )

        print_comparison(
            multi,
            res,
            lb,
            check,
            serial_makespan=serial,
        )

        optimization = summary.get("optimize_info") or {}

        if "human_longest_streak_after" in optimization:
            print("\n【人工连续工作】")
            print(
                "  最长连续工作时间："
                f"{optimization['human_longest_streak_after']:.1f} 分钟"
            )

        elapsed = (
            time.perf_counter() - t0
        ) * 1000.0

        print(
            f"\n[最终统计] "
            f"{'复用缓存' if reused else '重新计算'}，"
            f"耗时 {elapsed:.2f} ms"
        )

    return summary



# ============================================================
# 入口：三种模式
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # 模式选择
    # ============================================================
    MODE = "reschedule"        # "single" 单菜 / "multi" 多菜 / "reschedule" 动态重排

    # -------- task_id 配置 --------
    # 第一次启动新任务：TASK_ID = None
    # 后续多轮重排：把上一次输出的 task_id 填到这里，例如：
    # TASK_ID = "cook_a83f92e760bc"
    # 只要 TASK_ID 已存在，reschedule 模式就会直接恢复上一轮状态，
    # 不再重新执行 run_multi()。
    TASK_ID = None

    # -------- 单菜配置 --------
    RECIPE_ID = "5d54bae2a9114174727c8b20"

    # -------- 多菜配置 --------这里为菜品ID
    MULTI_IDS = [
        "66715324bfbee338853895c7",
        "5cc7e0404a21a4301960aef1",
        "5cc7e0404a21a4301960aef1",
        "5f73fee8a6daaa3b53ba043c",
    ]
    MULTI_MODE = "ids"    # "ids" 用 MULTI_IDS；"all" 用全部

    # -------- 动态重排配置 --------
    RESCHEDULE_NOW = 20.0   #可以修改新菜品插入时间

    # 场景 1：步骤执行进度 / 强制延迟
    RESCHEDULE_PROGRESS = {
        # 2: 0.5,
    }
    FORCED_DELAYS = {
        # 2: 10.0,
    }

    # 场景 2：设备故障
    MACHINE_OUTAGES = {
        # (3, 0): (20.0, 35.0),
        # (3, None): (20.0, 35.0),
    }

    # 场景 3：取消步骤
    CANCELLED_NODES = [
        # 5,
    ]

    # 场景 4：临时加菜
    # 防止同一道临时加菜因为重复运行脚本而被重复追加。
    EXTRA_RECIPE_IDS = [
        "5d565e87a9114174727c8bad", #插入菜品ID
    ]

    SAVE = True
    WIDTH = 60

    # ============================================================
    # 模式 1：单菜
    # ============================================================
    if MODE == "single":
        result = run(RECIPE_ID, csv_path=CSV_PATH,
                     bar_width=WIDTH, save=SAVE)
        if result.get("success"):
            print(f"\n[完成] {result['name']}  "
                  f"makespan={result['makespan']} 分钟  "
                  f"总耗时={result['total_ms']:.0f} ms")
            if result.get("png_file"):
                print(f"[甘特图] {result['png_file']}")
        else:
            print(f"\n[失败] {result.get('error', '未知错误')}")

    # ============================================================
    # 模式 2 & 3：多菜 / 动态重排
    # ============================================================
    elif MODE in ("multi", "reschedule"):
        os.makedirs(RESULT_DIR, exist_ok=True)
        t_all_0 = time.perf_counter()

        # --------------------------------------------------------
        # A. 判断是“新任务”还是“已有 task_id 的后续重排”
        # --------------------------------------------------------
        resume_existing = bool(TASK_ID)
        state = None
        targets = []

        if resume_existing:
            if MODE != "reschedule":
                print("[错误] 已提供 TASK_ID 时请使用 MODE='reschedule'")
                sys.exit(1)

            try:
                state = load_task_state(TASK_ID)
            except Exception as e:
                print(f"[错误] task_id 状态读取失败: {e}")
                traceback.print_exc()
                sys.exit(1)

            if state is None:
                print(f"[错误] 未找到 task_id: {TASK_ID}")
                print(f"[提示] 状态目录: {TASK_STATE_DIR}")
                sys.exit(1)

            task_id = TASK_ID
            result = state_to_result(state)
            current_round = int(state.get("round", 0))

            print("\n" + "#" * 78)
            print("# 恢复已有调度任务")
            print(f"# task_id   : {task_id}")
            print(f"# 当前轮次  : {current_round}")
            print(f"# 当前 makespan: {result['res'].makespan:.1f} 分钟")
            print("# 本次不会重新执行 run_multi / LLM 初始解析")
            print("#" * 78)

        else:
            # ----------------------------------------------------
            # B. 新任务：读取菜谱并执行一次初始多菜调度
            # ----------------------------------------------------
            all_recipes = read_recipes(CSV_PATH)
            print(f"\n[读取] CSV 共 {len(all_recipes)} 条菜谱")

            if MULTI_MODE == "all":
                targets = all_recipes
            else:
                for rid in MULTI_IDS:
                    r = find_recipe(all_recipes, rid)
                    if r:
                        targets.append(r)
                    else:
                        print(f"[警告] 未找到: {rid}")

            if not targets:
                print("[错误] 没有可选菜谱")
                sys.exit(1)

            print(f"[选择] 共 {len(targets)} 道菜谱：")
            for t in targets:
                print(f"    - {t['name']}  ({t['recipe_id']})")

            result = run_multi(
                targets,
                parse_schedule_text=parse_schedule_text,
                recipe_to_schedule_text=recipe_to_schedule_text,
                verbose=False,
                save_png=SAVE,
                save_dir=RESULT_DIR,
            )

            task_id = create_task_id()
            current_round = 0
            state = build_initial_task_state(task_id, result, targets)
            state_path = save_task_state(task_id, state)

            print("\n" + "#" * 78)
            print("# 新调度任务已创建")
            print(f"# task_id   : {task_id}")
            print(f"# 当前轮次  : {current_round}")
            print(f"# 状态文件  : {state_path}")
            print("# 后续重排请把该 task_id 原样传回")
            print("#" * 78)

            # ---------- 初始接口响应 ----------
            if SAVE:
                try:
                    from schedule_exporter import to_competition_format

                    required = to_competition_format(
                        result["multi"],
                        result["res"],
                        ingredients_list=result.get("ingredients_list"),
                        start_time=datetime.now(),
                    )

                    expected = {
                        "overview", "cookingTimeline", "ingredientsSummary",
                        "detailTimeline", "recipeDetail"
                    }
                    actual = set(required.keys())
                    missing = expected - actual
                    extra = actual - expected
                    if missing:
                        print(f"  ⚠️  5 字段内部响应缺失: {missing}")
                    if extra:
                        print(f"  ⚠️  5 字段内部响应多余: {extra}")
                    if not missing and not extra:
                        print("  ✓ data 内部 5 个字段齐全")

                    api_payload = make_api_payload(
                        task_id=task_id,
                        round_no=0,
                        required=required,
                        notifications=result.get("reminders", []),
                    )
                    api_json = os.path.join(RESULT_DIR, "api_response.json")
                    with open(api_json, "w", encoding="utf-8") as f:
                        json.dump(api_payload, f, ensure_ascii=False,
                                  indent=2, default=_json_default)
                    print(f"  ✓ 初始接口响应已保存: {api_json}")
                except Exception as e:
                    print(f"  ✗ 初始接口响应生成失败: {e}")
                    traceback.print_exc()

            # ---------- 初始多菜汇总 ----------
            if SAVE:
                out = os.path.join(RESULT_DIR, "multi_summary.json")
                with open(out, "w", encoding="utf-8") as f:
                    json.dump({
                        "generated_at": timestamp(),
                        "task_id": task_id,
                        "round": 0,
                        "recipe_ids": [t["recipe_id"] for t in targets],
                        "recipe_names": [t["name"] for t in targets],
                        "serial_makespan": result["serial_makespan"],
                        "parallel_makespan": result["res"].makespan,
                        "speedup": (
                            result["serial_makespan"] /
                            max(1e-9, result["res"].makespan)
                        ),
                        "lb_cp": result["lb"]["lb_cp"],
                        "lb_machine": result["lb"]["lb_machine"],
                        "lb": result["lb"]["lb"],
                        "conflicts": len(result["check"]["conflicts"]),
                        "dependency_violations": len(
                            result["check"]["dependency_violations"]
                        ),
                        "score": result["check"]["score"],
                        "llm_total_ms": result["llm_total_ms"],
                        "parse_total_ms": result["parse_total_ms"],
                        "algo_ms": result["res"].algo_time_ms,
                    }, f, ensure_ascii=False, indent=2)
                print(f"[结果] 多菜汇总: {out}")

                from schedule_exporter import export_chinese_doc_json
                chinese_path = os.path.join(RESULT_DIR, "中文解析文档.json")
                export_chinese_doc_json(
                    result["multi"],
                    result["res"],
                    save_path=chinese_path,
                    lb=result["lb"],
                    serial_makespan=result["serial_makespan"],
                    notifications=result.get("reminders", []),
                    mode="multi",
                )
                print(f"[结果] 中文解析: {chinese_path}")

        # ============================================================
        # 模式 3：动态重排
        # ============================================================
        if MODE == "reschedule":
            from rescheduler import dynamic_reschedule
            from schedule_exporter import export_schedule_json
            from reminder_generator import generate_reminders, print_reminders

            print("\n" + "#" * 78)
            print("# 进入动态重排模式")
            print(f"# task_id   : {task_id}")
            print(f"# 当前轮次  : {current_round}")
            print(f"# 当前时刻  : {RESCHEDULE_NOW} 分钟")
            print("#" * 78)

            # ★ 多轮重排核心：始终从 state 的最新排程继续，而不是 result 初始排程
            multi_obj = state["multi_obj"]
            orig_res = state["current_res"]

            # --------------------------------------------------------
            # 临时加菜：扩展任务模型，并把扩展后的模型保存回 task state
            # --------------------------------------------------------
            if EXTRA_RECIPE_IDS:
                existing_ids = {
                    str(info.get("recipe_id"))
                    for info in state.get("recipe_infos", [])
                    if isinstance(info, dict) and info.get("recipe_id")
                }
                existing_ids.update(str(x) for x in state.get("recipe_ids", []))

                all_recipes2 = read_recipes(CSV_PATH)
                new_targets = []
                for rid in EXTRA_RECIPE_IDS:
                    if str(rid) in existing_ids:
                        print(f"[临时加菜] 已存在，跳过重复 recipe_id: {rid}")
                        continue
                    r = find_recipe(all_recipes2, rid)
                    if r:
                        new_targets.append(r)
                    else:
                        print(f"[临时加菜] 未找到: {rid}")

                if new_targets:
                    print("\n" + "#" * 78)
                    print(f"# 临时加菜：{len(new_targets)} 道")
                    print("#" * 78)
                    for r in new_targets:
                        print(f"  + {r['name']}  ({r['recipe_id']})")

                    from llm_parser import recipe_to_schedule_json, json_to_schedule_text
                    from multi_scheduler import merge_instances, greedy_schedule_multi

                    old_inst_list = list(state.get("inst_list", []))
                    old_infos = list(state.get("recipe_infos", []))
                    old_step_names = list(state.get("step_names_list", []))

                    if not old_inst_list:
                        raise RuntimeError(
                            "当前 task state 缺少 inst_list，无法安全执行临时加菜。"
                            "请使用本修改版重新创建一次初始 task。"
                        )

                    new_inst_list = []
                    new_infos = []
                    new_step_names = []

                    for t in new_targets:
                        data_json = recipe_to_schedule_json(
                            t["recipe_id"], t["name"],
                            t["ingredients"], t["steps"],
                            verbose=True,
                        )

                        step_names = []
                        for j, s in enumerate(data_json["steps"]):
                            nm = s.get("name")
                            if not nm or not str(nm).strip():
                                nm = f"S{j}"
                            step_names.append(str(nm).strip())

                        text = json_to_schedule_text(data_json)
                        inst = parse_schedule_text(text)
                        new_inst_list.append(inst)
                        new_infos.append({
                            "recipe_id": t["recipe_id"],
                            "name": t["name"],
                        })
                        new_step_names.append(step_names)

                    all_inst = old_inst_list + new_inst_list
                    all_infos = old_infos + new_infos
                    all_step_names = old_step_names + new_step_names

                    multi_obj = merge_instances(
                        all_inst,
                        all_infos,
                        step_names_list=all_step_names,
                    )

                    # ============================================================
                    # 动态加菜：扩展旧排程并完整继承自动预热
                    # ============================================================
                    old_res = state["current_res"]

                    old_n = len(old_res.node_start)
                    new_n = multi_obj.N

                    if new_n < old_n:
                        raise RuntimeError(
                            f"临时加菜后节点数异常: old={old_n}, new={new_n}"
                        )

                    # 统一由这个函数扩展排程
                    # 不再重复构造 SimpleNamespace
                    orig_res = expand_schedule_preserving_preheats(
                        old_res=old_res,
                        new_node_count=new_n,
                        now=RESCHEDULE_NOW,
                    )

                    print(f"  ✓ 保留旧排程节点 {old_n} 个")
                    print(f"  ✓ 新增待调度节点 {new_n - old_n} 个")
                    print(
                        f"  ✓ 新增节点从 now={RESCHEDULE_NOW:.1f} "
                        f"之后进入动态调度"
                    )

                    print(
                        f"  ✓ 已继承自动预热记录 "
                        f"{len(orig_res.preheat_records)} 条"
                    )

                    print(
                        f"  ✓ 保留旧排程节点 {old_n} 个"
                    )

                    print(
                        f"  ✓ 新增待调度节点 {new_n - old_n} 个"
                    )

                    print(
                        f"  ✓ 新增节点从 now={RESCHEDULE_NOW:.1f} "
                        f"之后进入动态调度"
                    )

                    state["inst_list"] = all_inst
                    state["recipe_infos"] = all_infos
                    state["step_names_list"] = all_step_names
                    state["multi_obj"] = multi_obj
                    state["recipe_ids"] = (
                        list(state.get("recipe_ids", [])) +
                        [t["recipe_id"] for t in new_targets]
                    )
                    state["recipe_names"] = (
                        list(state.get("recipe_names", [])) +
                        [t["name"] for t in new_targets]
                    )

                    print(f"  ✓ 合并后 N={multi_obj.N} E={multi_obj.E} K={multi_obj.K}")
                    print(
                        f"  ✓ 已扩展动态任务模型："
                        f"old_N={old_n}, new_N={new_n}"
                    )

                    print(
                        f"  ✓ 旧计划保持至当前时刻 "
                        f"{RESCHEDULE_NOW:.1f} 分钟，"
                        f"新增任务等待动态重排"
                    )

            # -------- 构造标签函数 --------
            counter = {}
            labels = {}
            for nid in range(multi_obj.N):
                rid = multi_obj.recipe_of_node[nid]
                seq = counter.get(rid, 0)
                counter[rid] = seq + 1
                short = multi_obj.recipe_names[rid][:3]
                labels[nid] = f"{short}_S{seq}"

            # -------- 执行重排 --------
            try:
                rs = dynamic_reschedule(
                    multi_obj,
                    orig_res,
                    now=RESCHEDULE_NOW,
                    running_progress=RESCHEDULE_PROGRESS,
                    forced_delays=FORCED_DELAYS,
                    machine_outages=MACHINE_OUTAGES,
                    cancelled_nodes=CANCELLED_NODES,
                    label_fn=lambda nid: labels.get(nid, f"N{nid}"),
                    verbose=True,
                )

                verify_dynamic_preheat_history(
                    before=orig_res,
                    after=rs["new_res"],
                    now=RESCHEDULE_NOW,
                )
            except AssertionError as e:
                print(f"\n[重排失败] {e}")
                sys.exit(2)

            # --------------------------------------------------------
            # ★ 多轮重排状态提交
            # --------------------------------------------------------
            new_round = current_round + 1
            history_item = {
                "round": new_round,
                "generated_at": timestamp(),
                "now": RESCHEDULE_NOW,
                "old_makespan": orig_res.makespan,
                "new_makespan": rs["new_res"].makespan,
                "frozen_nodes": list(rs["new_res"].frozen_nodes),
                "changed_nodes": list(rs["new_res"].changed_nodes),
                "notifications": rs.get("notifications", []),
                "events": {
                    "running_progress": dict(RESCHEDULE_PROGRESS),
                    "forced_delays": dict(FORCED_DELAYS),
                    "machine_outages": {
                        str(k): list(v) if isinstance(v, tuple) else v
                        for k, v in MACHINE_OUTAGES.items()
                    },
                    "cancelled_nodes": list(CANCELLED_NODES),
                    "extra_recipe_ids": [t["recipe_id"] for t in new_targets]
                    if 'new_targets' in locals() else [],
                },
            }

            state.setdefault("history", []).append(history_item)
            state["round"] = new_round
            state["multi_obj"] = multi_obj
            state["current_res"] = rs["new_res"]

            refresh_final_schedule_analysis(
                state,
                task_id=task_id,
                show=False,
                save_dir=RESULT_DIR if SAVE else None,
            )

            # 更新后再保存，保证下次恢复使用最新数据
            state_path = save_task_state(task_id, state)

            print("\n" + "#" * 78)
            print("# 动态重排状态已提交")
            print(f"# task_id   : {task_id}")
            print(f"# 新轮次    : {new_round}")
            print(f"# 状态文件  : {state_path}")
            print("# 下一轮请继续使用同一个 task_id")
            print("#" * 78)

            # -------- 保存重排报告 --------
            if SAVE:
                out = os.path.join(
                    RESULT_DIR,
                    f"reschedule_result_{safe_filename(task_id)}_r{new_round}.json"
                )
                data = {
                    "generated_at": timestamp(),
                    "task_id": task_id,
                    "round": new_round,
                    "now": RESCHEDULE_NOW,
                    "old_makespan": orig_res.makespan,
                    "new_makespan": rs["new_res"].makespan,
                    "frozen_nodes": list(rs["new_res"].frozen_nodes),
                    "changed_count": len(rs["new_res"].changed_nodes),
                    "notifications": rs["notifications"],
                    "compare": {
                        "orig_device_conflicts": rs["compare"]["orig_device_conflicts"],
                        "new_device_conflicts": rs["compare"]["new_device_conflicts"],
                        "orig_dep_violations": rs["compare"]["orig_dep_violations"],
                        "new_dep_violations": rs["compare"]["new_dep_violations"],
                    },
                }
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2,
                              default=_json_default)
                print(f"[结果] 重排报告: {out}")

                # 结构化排程 JSON：使用 task_id + round 防止多轮覆盖
                orig_schedule_json = os.path.join(
                    RESULT_DIR,
                    f"schedule_before_{safe_filename(task_id)}_r{new_round}.json"
                )
                export_schedule_json(
                    multi_obj,
                    orig_res,
                    save_path=orig_schedule_json,
                    lb=state.get("lb"),
                    serial_makespan=state.get("serial_makespan"),
                    mode="multi",
                )

                reschedule_json = os.path.join(
                    RESULT_DIR,
                    f"schedule_after_{safe_filename(task_id)}_r{new_round}.json"
                )
                export_schedule_json(
                    multi_obj,
                    rs["new_res"],
                    save_path=reschedule_json,
                    lb=state.get("lb"),
                    serial_makespan=state.get("serial_makespan"),
                    mode="reschedule",
                    now=RESCHEDULE_NOW,
                    notifications=rs["notifications"],
                )
                print(f"[排程 JSON] 重排前: {orig_schedule_json}")
                print(f"[排程 JSON] 重排后: {reschedule_json}")

                # ---------- 重排后的接口响应 ----------
                try:
                    from schedule_exporter import to_required_format

                    required = to_required_format(
                        multi_obj,
                        rs["new_res"],
                        ingredients_list=state.get("ingredients_list"),
                        lb=state.get("lb"),
                        serial_makespan=state.get("serial_makespan"),
                    )
                    api_payload = make_api_payload(
                        task_id=task_id,
                        round_no=new_round,
                        required=required,
                        notifications=rs["notifications"],
                    )

                    # latest 文件：方便接口直接读取
                    api_json = os.path.join(RESULT_DIR, "api_response_rescheduled.json")
                    with open(api_json, "w", encoding="utf-8") as f:
                        json.dump(api_payload, f, ensure_ascii=False,
                                  indent=2, default=_json_default)

                    # 历史文件：每轮保留
                    api_history_json = os.path.join(
                        RESULT_DIR,
                        f"api_response_{safe_filename(task_id)}_r{new_round}.json"
                    )
                    with open(api_history_json, "w", encoding="utf-8") as f:
                        json.dump(api_payload, f, ensure_ascii=False,
                                  indent=2, default=_json_default)

                    print(f"  ✓ 重排接口响应: {api_json}")
                    print(f"  ✓ 本轮历史响应: {api_history_json}")
                except Exception as e:
                    print(f"  ✗ 重排接口响应生成失败: {e}")
                    traceback.print_exc()

                # -------- 中文解析文档 --------
                from schedule_exporter import export_chinese_doc_json
                chinese_path = os.path.join(
                    RESULT_DIR,
                    f"中文解析文档_重排后_{safe_filename(task_id)}_r{new_round}.json"
                )
                export_chinese_doc_json(
                    multi_obj,
                    rs["new_res"],
                    save_path=chinese_path,
                    lb=state.get("lb"),
                    serial_makespan=state.get("serial_makespan"),
                    notifications=rs["notifications"],
                    mode="reschedule",
                )
                print(f"[结果] 中文解析（重排后）: {chinese_path}")

                # -------- 甘特图 --------
                names = "_".join([n[:4] for n in multi_obj.recipe_names])
                png2 = os.path.join(
                    RESULT_DIR,
                    f"multi_gantt_{safe_filename(task_id)}_r{new_round}_{names}.png"
                )
                try:
                    plot_gantt_matplotlib(
                        multi_obj,
                        rs["new_res"],
                        save_path=png2,
                        show=False,
                        figsize=(16, 9),
                        dpi=200,
                        title=(
                            f"重排后甘特图  task={task_id}  round={new_round}  "
                            f"now={RESCHEDULE_NOW}min  "
                            f"makespan={rs['new_res'].makespan:.1f}分钟"
                        ),
                    )
                    print(f"[甘特图] {png2}")
                except Exception as e:
                    print(f"  ✗ 重排甘特图失败: {e}")

            # -------- 实时提醒演示 --------
            print("\n" + "#" * 78)
            print("# 实时提醒演示")
            print("#" * 78)
            makespan = rs["new_res"].makespan
            t_now = RESCHEDULE_NOW
            while t_now <= makespan:
                reminders = generate_reminders(
                    multi_obj,
                    rs["new_res"],
                    now=t_now,
                    lookahead=5.0,
                )
                if reminders:
                    print(f"\n[时刻 {t_now:.1f} 分钟]")
                    print_reminders(reminders)
                t_now += 2.0

            # -------- 汇总 --------
            print("\n" + "=" * 72)
            print("动态重排 - 结果摘要")
            print("=" * 72)
            print(f"  task_id            : {task_id}")
            print(f"  round              : {new_round}")
            print(f"  原 makespan        : {orig_res.makespan:.1f} 分钟")
            print(f"  新 makespan        : {rs['new_res'].makespan:.1f} 分钟")
            print(f"  冻结节点数         : {len(rs['new_res'].frozen_nodes)}")
            print(f"  变更节点数         : {len(rs['new_res'].changed_nodes)}")
            print(
                "  新增设备冲突       : "
                f"{rs['compare']['new_device_conflicts'] - rs['compare']['orig_device_conflicts']}"
            )
            print(f"  依赖违规数         : {rs['compare']['new_dep_violations']}")
            print(f"  重排耗时           : {rs['new_res'].algo_time_ms:.3f} ms")
            print("=" * 72)

        t_all = time.perf_counter() - t_all_0
        print(f"\n[总耗时] {t_all:.2f}s")

        # ============================================================
        # 最后统一输出最新排程的完整结果分析
        # ============================================================
        refresh_final_schedule_analysis(
            state,
            task_id=task_id,
            show=True,
            save_dir=RESULT_DIR if SAVE else None,
        )

    else:
        print(f"[错误] 未知模式: {MODE}，可选 single / multi / reschedule")
        sys.exit(1)
