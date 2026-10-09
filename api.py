# api.py
# -*- coding: utf-8 -*-
"""
FastAPI 接口（赛事组规范）

接口清单:
    POST /schedule     多菜并行烹饪规划（主接口）
    POST /reschedule   动态重排
    POST /parse        单菜结构化解析
    GET  /health       健康检查
"""

import time as _time
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_parser import (
    recipe_to_schedule_json, json_to_schedule_text
)
from greedy_scheduler import parse_schedule_text, greedy_schedule
from multi_scheduler import (
    merge_instances,
    greedy_schedule_multi,
    optimize_lexicographic_schedule,
)
from schedule_exporter import to_competition_format
import run as run_module


app = FastAPI(
    title="智能烹饪调度 Agent API",
    version="3.0",
    description="多菜谱并行烹饪规划",
)


# ============================================================
# 请求模型
# ============================================================
class RecipeItem(BaseModel):
    id: str
    name: str
    ingredients: Optional[str] = None
    steps: Optional[str] = None


class RescheduleRequest(BaseModel):
    recipes: List[RecipeItem]
    now: float
    event: Optional[dict] = None


# ============================================================
# 辅助：解析请求中的 {id, name} -> 菜谱记录
# ============================================================
def _resolve_targets(items):
    """
    优先从 CSV 查；查不到则用请求里的 ingredients/steps 降级。
    """
    all_recipes = run_module.read_recipes(run_module.CSV_PATH)
    targets = []
    for item in items:
        r = run_module.find_recipe(all_recipes, item.id)
        if r is not None:
            # 菜名校验
            if r["name"] != item.name:
                raise HTTPException(
                    400,
                    f"菜名不一致：请求 '{item.name}'，"
                    f"库中 '{r['name']}'"
                )
            targets.append(r)
        else:
            # 降级：用请求里的内容
            if not item.ingredients or not item.steps:
                raise HTTPException(
                    404,
                    f"菜谱 {item.id} 不在库中，"
                    f"请提供 ingredients 和 steps"
                )
            targets.append({
                "recipe_id": item.id,
                "name": item.name,
                "ingredients": item.ingredients,
                "steps": item.steps,
            })
    return targets


# ============================================================
# 辅助：批量解析（可选并发）
# ============================================================
def _parse_all(targets, verbose=False):
    """
    批量解析菜谱。单菜失败不影响其他。
    返回 (inst_list, recipe_infos, ingredients_list, step_names_list)
    """
    inst_list = []
    recipe_infos = []
    ingredients_list = []
    step_names_list = []

    for i, t in enumerate(targets, 1):
        if verbose:
            print(f"  [{i}/{len(targets)}] 解析 {t['name']}...")

        data_json = recipe_to_schedule_json(
            t["recipe_id"], t["name"],
            t["ingredients"], t["steps"],
            verbose=False,
        )

        step_names = []
        for j, s in enumerate(data_json.get("steps", [])):
            nm = s.get("name")
            if not nm or not str(nm).strip():
                nm = f"S{j}"
            step_names.append(str(nm).strip())
        step_names_list.append(step_names)

        ingredients_list.append(data_json.get("ingredients", []))

        text = json_to_schedule_text(data_json)
        inst = parse_schedule_text(text)
        inst_list.append(inst)

        recipe_infos.append({
            "recipe_id": t["recipe_id"],
            "name": t["name"],
        })

    return (inst_list, recipe_infos,
            ingredients_list, step_names_list)


# ============================================================
# 主接口：多菜并行烹饪规划
# ============================================================
@app.post("/schedule")
def schedule_recipes(req: List[RecipeItem]):
    """
    接收菜谱数组，返回赛事组规范 5 字段裸 JSON。

    请求：
        [{"id": "1001", "name": "照烧鸡腿"}, ...]

    响应：
        {overview, cookingTimeline, ingredientsSummary,
         detailTimeline, recipeDetail}
    """
    t_start = _time.perf_counter()

    # 数量校验
    n = len(req)
    if n < 3 or n > 5:
        raise HTTPException(
            400, f"菜谱数量需 3~5，当前 {n}"
        )

    # 唯一性校验
    ids = [r.id for r in req]
    if len(ids) != len(set(ids)):
        raise HTTPException(400, "菜谱 id 不得重复")

    try:
        targets = _resolve_targets(req)

        (inst_list, recipe_infos,
         ingredients_list, step_names_list) = _parse_all(targets)

        multi = merge_instances(
            inst_list, recipe_infos,
            step_names_list=step_names_list,
        )
        res = optimize_lexicographic_schedule(
            multi,
            max_extra_minutes=0.0,
            verbose=False,
        )

        response = to_competition_format(
            multi, res,
            ingredients_list=ingredients_list,
            start_time=datetime.now(),
        )

        # 端到端耗时
        t_end = _time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0
        print(f"\n[API] /schedule 端到端耗时: {total_ms:.1f} ms")
        print(f"[API]   调度算法: {res.algo_time_ms:.3f} ms")

        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"调度失败: {e}")


# ============================================================
# 动态重排
# ============================================================
@app.post("/reschedule")
def reschedule(req: RescheduleRequest):
    """动态重排（含端到端计时）"""
    t_start = _time.perf_counter()

    try:
        from rescheduler import dynamic_reschedule

        targets = _resolve_targets(req.recipes)

        (inst_list, recipe_infos,
         ingredients_list, step_names_list) = _parse_all(targets)

        multi = merge_instances(
            inst_list, recipe_infos,
            step_names_list=step_names_list,
        )
        orig_res = optimize_lexicographic_schedule(
            multi,
            max_extra_minutes=0.0,
            verbose=False,
        )

        rs = dynamic_reschedule(
            multi, orig_res,
            now=req.now, verbose=False,
        )

        response = to_competition_format(
            multi, rs["new_res"],
            ingredients_list=ingredients_list,
            start_time=datetime.now(),
        )
        response["notifications"] = rs["notifications"]

        # 端到端耗时
        t_end = _time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0

        print(f"\n[API] /reschedule 端到端耗时: {total_ms:.1f} ms")
        # 如果 rescheduler 返回 timing 字段才打印
        timing = rs.get("timing", {})
        if timing:
            print(f"[API]   快照   : "
                  f"{timing.get('snapshot_ms', 0):.3f} ms")
            print(f"[API]   重排   : "
                  f"{timing.get('reschedule_ms', 0):.3f} ms")
            print(f"[API]   冲突   : "
                  f"{timing.get('compare_ms', 0):.3f} ms")
            print(f"[API]   通知   : "
                  f"{timing.get('notify_ms', 0):.3f} ms")

        return response

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"重排失败: {e}")


# ============================================================
# 兼容接口
# ============================================================
@app.post("/parse")
def parse_recipe(req: RecipeItem):
    """单菜结构化解析（调试用）"""
    try:
        data = recipe_to_schedule_json(
            req.id, req.name,
            req.ingredients or "", req.steps or "",
            verbose=False,
        )
        return data
    except Exception as e:
        raise HTTPException(500, f"解析失败: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}