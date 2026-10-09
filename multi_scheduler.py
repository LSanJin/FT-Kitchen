# multi_scheduler.py
# -*- coding: utf-8 -*-
"""
多菜并行调度模块

功能：
    1. merge_instances          合并多个单菜 Instance
    2. greedy_schedule_multi    跨菜谱堆栈贪心调度（共享设备）
    3. check_multi_schedule     设备冲突 + 依赖校验 + 扣分
    4. compute_lower_bound      关键路径 / 机器负荷 下界
    5. print_multi_gantt        多菜甘特图
    6. print_comparison         串行 vs 并行 vs 下界 对比
"""

import heapq
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from gantt_plot import plot_gantt_matplotlib
from greedy_scheduler import allocate_batch
from llm_parser import BATCH_CAPACITY, DEFAULT_BATCH_CAPACITY
from schedule_exporter import export_schedule_json




def _json_default(obj):
    """让 json.dump 支持 set / tuple / 自定义对象"""
    if isinstance(obj, set):
        return sorted(obj, key=str)
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)

# ============================================================
# 1. 数据结构
# ============================================================
@dataclass
class MultiInstance:
    N: int
    E: int
    K: int
    capacities: list                      # [cap0, cap1, ...]
    edges: list                           # [(frm, to), ...]
    nodes: list                           # [[(mid, dur), ...], ...]
    machines: list                        # Machine 对象列表
    recipe_of_node: list                  # 每个全局节点属于哪个菜谱（索引）
    recipe_names: list                    # 菜谱名
    recipe_ids: list                      # 菜谱 id
    step_names: list                      # 每个全局节点的步骤名
    needs_human: list = field(default_factory=list)
    successors: list = field(default_factory=list)
    predecessors: list = field(default_factory=list)
    indegree: list = field(default_factory=list)
    node_duration: list = field(default_factory=list)

    def build(self):
        n = self.N
        self.successors = [[] for _ in range(n)]
        self.predecessors = [[] for _ in range(n)]
        self.indegree = [0] * n
        self.node_duration = [0] * n
        for f, t in self.edges:
            self.successors[f].append(t)
            self.predecessors[t].append(f)
            self.indegree[t] += 1
        for i, ops in enumerate(self.nodes):
            # ops 可能是 2 元组或 5 元组
            total = 0
            for item in ops:
                if len(item) >= 2:
                    total += item[1]
            self.node_duration[i] = total




def _machine_name(m):
    from ft_equipment_adapter import machine_merge_key

    return machine_merge_key(m)




def merge_instances(inst_list, recipe_infos, step_names_list=None,
                    optimize_parallel=True):
    """
    合并菜谱、共享真实设备，并修补已确认的烹饪先后关系。

    原则：
    1. 默认保留LLM生成的原始依赖。
    2. 补充已确认的成菜必需依赖。
    3. 唯一白名单重连：蛋糕的“预热烤箱->煮牛奶黄油”
       改为“预热烤箱->正式烘烤”。
    4. 发现循环依赖时明确报错。
    5. optimize_parallel保留以兼容原接口。
    """
    from collections import deque

    if len(inst_list) != len(recipe_infos):
        raise ValueError(
            "inst_list与recipe_infos数量不一致"
        )

    machines = []
    global_ids = {}
    maps = []

    # 1. 合并全局物理设备
    for inst in inst_list:
        mapping = {}

        for local_mid, machine in enumerate(inst.machines):
            key = _machine_name(machine)

            if key not in global_ids:
                global_ids[key] = len(machines)
                machines.append(machine)

            mapping[local_mid] = global_ids[key]

        maps.append(mapping)

    capacity = [1] * len(machines)

    for rid, inst in enumerate(inst_list):
        for mid in range(inst.K):
            gmid = maps[rid][mid]

            capacity[gmid] = max(
                capacity[gmid],
                int(
                    inst.capacities[mid]
                    if mid < len(inst.capacities)
                    else 1
                )
            )

    multi = MultiInstance(
        N=0,
        E=0,
        K=len(machines),
        capacities=capacity,
        edges=[],
        nodes=[],
        machines=machines,
        recipe_of_node=[],
        recipe_names=[],
        recipe_ids=[],
        step_names=[],
        needs_human=[]
    )

    # 2. 合并所有菜谱节点
    for rid, (inst, info) in enumerate(
        zip(inst_list, recipe_infos)
    ):
        offset = multi.N

        multi.recipe_names.append(info["name"])
        multi.recipe_ids.append(info["recipe_id"])

        local_names = (
            step_names_list[rid]
            if step_names_list is not None
            and rid < len(step_names_list)
            else None
        )

        if (
            local_names is not None
            and len(local_names) != inst.N
        ):
            raise ValueError(
                f"{info['name']}：步骤名称数"
                f"{len(local_names)}与节点数"
                f"{inst.N}不一致"
            )

        actual_names = [
            (
                str(local_names[j]).strip()
                if local_names is not None
                else f"S{j}"
            )
            for j in range(inst.N)
        ]

        for j, ops in enumerate(inst.nodes):
            multi.nodes.append([
                (
                    maps[rid][int(it[0])],
                    *it[1:]
                )
                for it in ops
            ])

            multi.recipe_of_node.append(rid)

            multi.step_names.append(
                f"{info['name']}#{actual_names[j]}"
            )

            multi.needs_human.append(
                bool(inst.needs_human[j])
                if hasattr(inst, "needs_human")
                and j < len(inst.needs_human)
                else True
            )

        multi.edges.extend(
            (u + offset, v + offset)
            for u, v in inst.edges
        )

        multi.N += inst.N

    edges = set(multi.edges)
    added = []
    rewired = []

    # 3. 根据明确的食材工艺修复必要依赖
    for rid, recipe_name in enumerate(
        multi.recipe_names
    ):
        ids = [
            i
            for i, r in enumerate(multi.recipe_of_node)
            if r == rid
        ]

        by_name = {
            i: multi.step_names[i]
            .split("#", 1)[-1]
            .replace(" ", "")
            for i in ids
        }

        def pick(predicate, label, required=False):
            hits = [
                i
                for i in ids
                if predicate(by_name[i])
            ]

            if len(hits) > 1:
                raise ValueError(
                    f"{recipe_name}：语义节点"
                    f"“{label}”匹配多个步骤："
                    f"{[(i, by_name[i]) for i in hits]}"
                    "，请核实步骤名称"
                )

            if required and not hits:
                raise ValueError(
                    f"{recipe_name}：缺少关键步骤"
                    f"“{label}”，请核对缓存JSON"
                )

            return hits[0] if hits else None

        def require(before, after, reason):
            if before is None or after is None:
                return

            if before == after:
                raise ValueError(
                    f"{recipe_name}：{reason}"
                    "匹配到同一节点"
                )

            if (before, after) not in edges:
                edges.add((before, after))

                added.append((
                    recipe_name,
                    by_name[before],
                    by_name[after]
                ))

        # 3.1 糯米烧麦
        if "糯米烧麦" in recipe_name:

            rice = pick(
                lambda s: (
                    "蒸糯米" in s
                    and "烧麦" not in s
                ),
                "蒸糯米",
                required=True
            )

            mix = pick(
                lambda s: (
                    ("糯米饭" in s or "糯米" in s)
                    and any(
                        x in s for x in ("拌", "混")
                    )
                    and not any(
                        x in s
                        for x in ("准备", "浸泡", "蒸")
                    )
                ),
                "拌糯米饭"
            )

            pack = pick(
                lambda s: (
                    "烧麦" in s and "包" in s
                ),
                "包烧麦",
                required=True
            )

            final_steam = pick(
                lambda s: (
                    "烧麦" in s
                    and "蒸" in s
                    and "预热" not in s
                ),
                "最终蒸烧麦",
                required=True
            )

            soak = pick(
                lambda s: (
                    "糯米" in s
                    and "浸泡" in s
                    and "等待" in s
                ),
                "浸泡糯米_等待"
            )

            require(
                soak, rice,
                "浸泡完成后蒸米"
            )

            require(
                rice,
                mix if mix is not None else pack,
                "蒸米后拌饭或包烧麦"
            )

            require(
                rice, pack,
                "糯米蒸熟后才能包烧麦"
            )

            if mix is not None:
                require(
                    mix, pack,
                    "拌饭后包烧麦"
                )

            require(
                pack, final_steam,
                "包好烧麦后最终蒸制"
            )

            require(
                rice, final_steam,
                "蒸糯米必须早于最终蒸烧麦"
            )

        # 3.2 轻松一锅蒸
        elif "轻松一锅蒸" in recipe_name:

            fish = pick(
                lambda s: (
                    ("鳙鱼头" in s or "鱼头" in s)
                    and any(
                        x in s
                        for x in ("放置", "放入", "摆入")
                    )
                ),
                "放置鱼头"
            )

            cook = pick(
                lambda s: s in (
                    "开始烹饪",
                    "蒸制",
                    "正式蒸制",
                    "开始蒸制"
                ),
                "最终蒸制"
            )

            all_ready = pick(
                lambda s: "准备所有食材" in s,
                "准备所有食材"
            )

            if cook is not None:
                require(
                    fish, cook,
                    "鱼头放入后再蒸制"
                )

                require(
                    all_ready, cook,
                    "食材准备好后再蒸制"
                )

        # 3.3 半熟芝士蛋糕
        elif "半熟芝士" in recipe_name:

            preheat = pick(
                lambda s: (
                    "预热" in s
                    and "烤箱" in s
                ),
                "预热烤箱"
            )

            milk = pick(
                lambda s: "煮牛奶黄油" in s,
                "煮牛奶黄油"
            )

            bake = pick(
                lambda s: (
                    "烘烤" in s
                    or "烤蛋糕" in s
                ),
                "烘烤蛋糕"
            )

            if (
                preheat is not None
                and milk is not None
                and bake is not None
            ):
                # 原料准备与预热没有必要串行
                if (preheat, milk) in edges:
                    edges.remove((preheat, milk))

                    rewired.append((
                        recipe_name,
                        by_name[preheat],
                        by_name[milk],
                        by_name[bake]
                    ))

                require(
                    preheat, bake,
                    "预热后正式烘烤"
                )

        # 3.4 美式薯条
        elif "美式薯条" in recipe_name:

            put = pick(
                lambda s: "放入薯条" in s,
                "放入薯条"
            )

            cook = pick(
                lambda s: (
                    "开始烹饪" in s
                    or "烤薯条" in s
                ),
                "正式烹饪"
            )

            require(
                put, cook,
                "放入薯条后开始烹饪"
            )

    # 4. 重建DAG
    multi.edges = sorted(edges)
    multi.E = len(multi.edges)
    multi.build()

    # 5. 拓扑排序验证
    indeg = list(multi.indegree)

    q = deque(
        i
        for i in range(multi.N)
        if indeg[i] == 0
    )

    seen = 0

    while q:
        u = q.popleft()
        seen += 1

        for v in multi.successors[u]:
            indeg[v] -= 1

            if indeg[v] == 0:
                q.append(v)

    if seen != multi.N:
        involved = [
            multi.step_names[i]
            for i, value in enumerate(indeg)
            if value > 0
        ]

        raise ValueError(
            "原菜谱依赖与必要烹饪顺序冲突，"
            "发现循环依赖："
            + "、".join(involved[:15])
            + "。请核查JSON中的predecessors，"
            "不能自动删除不确定的原始依赖。"
        )

    # 6. 输出修复日志
    for dish, a, b, c in rewired:
        print(
            f"[工艺修正] {dish}: "
            f"{a}->{b} 改为 {a}->{c}"
        )
    if added:
        print(
            f"[工序语义校验] "
            f"补充{len(added)}条关键依赖"
        )
        for dish, a, b in added:
            print(
                f"  {dish}: {a} -> {b}"
            )
    return multi



# ============================================================
# 3. 多菜堆栈贪心调度
# ============================================================
@dataclass
class MultiScheduleResult:
    node_start: list
    node_end: list
    op_records: list      # [(nid, mid, bid, start, end), ...]
    makespan: float
    algo_time_ms: float


def greedy_schedule_multi(
    multi,
    sync_target=None,
    snapshot=None,
    enable_shared=True,
):
    """
    多菜并行调度，支持跨菜谱共享设备。

    共享条件：
    1. 同一物理设备
    2. 相同模式、温度、火力
    3. 同类工序（预热不能与正式烹饪混批）
    4. 批次容量允许
    5. 能够共同开始

    snapshot 用于动态重排，冻结已完成和进行中节点。
    """
    from collections import defaultdict, deque
    from copy import deepcopy
    from greedy_scheduler import allocate_batch
    from llm_parser import (
        BATCH_CAPACITY,
        DEFAULT_BATCH_CAPACITY,
    )
    import heapq
    import time

    t0 = time.perf_counter()
    EPS = 1e-7

    n = multi.N
    K = multi.K

    node_start = [0.0] * n
    node_end = [0.0] * n
    records = []

    machine_batches = [[] for _ in range(K)]
    human_busy = []

    now = (
        float(snapshot.now)
        if snapshot is not None
        else 0.0
    )

    states = (
        snapshot.node_states
        if snapshot is not None
        else {}
    )

    def machine_name(mid):
        return str(
            getattr(multi.machines[mid], "note", "") or ""
        )

    def batch_cap(mid):
        if not enable_shared:
            return 1

        name = machine_name(mid)

        if any(x in name for x in ("蒸箱", "烤箱", "冰箱")):
            return 3

        original_id = getattr(
            multi.machines[mid], "id", mid
        )

        if original_id in (2, 3, 4) and not name:
            return BATCH_CAPACITY.get(
                original_id,
                DEFAULT_BATCH_CAPACITY,
            )

        return 1

    def op_key(nid, item):
        if len(item) >= 5:
            temp, heat, mode = item[2:5]
        else:
            temp, heat, mode = 0, 0, 0

        name = (
            str(multi.step_names[nid]).split("#")[-1]
            if hasattr(multi, "step_names")
            else ""
        )

        kind = "预热" if "预热" in name else "操作"

        # 不再包含 recipe_id 和 duration
        return (kind, mode, temp, heat)

    # 恢复动态重排中已完成、正在执行的节点
    grouped = defaultdict(list)
    frozen = set()

    for nid, st in states.items():
        if st.status not in ("done", "running"):
            continue

        frozen.add(nid)

        node_start[nid] = float(st.start)
        node_end[nid] = float(st.end)

        if st.status == "done" and st.actual_end is not None:
            node_end[nid] = float(st.actual_end)

        original_node_end = max(
            (float(e) for _, _, _, e in st.machine_assignments),
            default=float(st.end),
        )

        extension = (
            max(0.0, float(st.end) - original_node_end)
            if st.status == "running"
            else 0.0
        )

        if (
            multi.needs_human[nid]
            and node_end[nid] > node_start[nid] + EPS
        ):
            human_busy.append(
                (node_start[nid], node_end[nid])
            )

        for mid, bid, s, e in st.machine_assignments:
            s = float(s)
            e = float(e) + extension

            op_item = next(
                (
                    it for it in multi.nodes[nid]
                    if it[0] == mid
                ),
                (mid, e - s, 0, 0, 0),
            )

            if len(op_item) >= 5:
                temp, heat, mode = op_item[2:5]
            else:
                temp, heat, mode = 0, 0, 0

            records.append((
                nid, mid, bid, s, e,
                temp, heat, mode,
            ))

            grouped[(mid, bid)].append(
                (nid, s, e, op_key(nid, op_item))
            )

    # 恢复真实物理批次，而不是逐条重复登记
    for (mid, bid), members in grouped.items():
        keys = {m[3] for m in members}

        if len(keys) != 1:
            raise ValueError(
                f"冻结批次 {(mid, bid)} 工况不一致"
            )

        machine_batches[mid].append({
            "key": members[0][3],
            "start": min(m[1] for m in members),
            "end": max(m[2] for m in members),
            "used": len(members),
            "cap": batch_cap(mid),
            "id": bid,
        })

    unavailable_until = [now] * K

    if snapshot is not None:
        for (mid, _iid), unavailable in (
            snapshot.machine_avail.items()
        ):
            if 0 <= mid < K:
                unavailable_until[mid] = max(
                    unavailable_until[mid],
                    float(unavailable),
                )

    pending = {
        i for i in range(n)
        if i not in frozen
    }

    indeg = {
        i: sum(
            1 for p in multi.predecessors[i]
            if p in pending
        )
        for i in pending
    }

    # 完整拓扑排序及关键路径优先级
    dag_indeg = list(multi.indegree)
    q = deque(
        i for i in range(n)
        if dag_indeg[i] == 0
    )

    topo = []

    while q:
        u = q.popleft()
        topo.append(u)

        for v in multi.successors[u]:
            dag_indeg[v] -= 1

            if dag_indeg[v] == 0:
                q.append(v)

    if len(topo) != n:
        raise ValueError("多菜调度存在循环依赖")

    rank = [0.0] * n

    for nid in reversed(topo):
        longest = max(
            (rank[j] for j in multi.successors[nid]),
            default=0.0,
        )

        current_duration = max(
            (float(it[1]) for it in multi.nodes[nid]),
            default=0.0,
        )

        rank[nid] = current_duration + longest

    ready = [
        (-rank[i], i)
        for i in pending
        if indeg[i] == 0
    ]

    heapq.heapify(ready)
    scheduled = 0

    def human_earliest(t, dur):
        for _ in range(len(human_busy) + 2):
            conflict_ends = [
                e for s, e in human_busy
                if s < t + dur - EPS
                and e > t + EPS
            ]

            if not conflict_ends:
                return t

            t = max(t, min(conflict_ends))

        raise RuntimeError("无法分配人工资源")

    # 正式调度
    while ready:
        _, sid = heapq.heappop(ready)

        est = max(
            now,
            max(
                (node_end[p] for p in multi.predecessors[sid]),
                default=0.0,
            ),
        )

        ops = list(multi.nodes[sid])

        max_dur = max(
            (float(it[1]) for it in ops),
            default=0.0,
        )

        need_human = bool(multi.needs_human[sid])
        start = est

        # 通过副本寻找多个设备共同可执行的开始时间
        # 不允许试探阶段污染真实批次
        for _ in range(2 * (n + len(records) + 3)):
            if need_human:
                start = human_earliest(start, max_dur)

            trial_batches = deepcopy(machine_batches)
            proposed = [start]

            for it in ops:
                mid = int(it[0])
                dur = float(it[1])

                key = op_key(sid, it)

                mid_floor = (
                    max(start, unavailable_until[mid])
                    if snapshot is not None
                    else start
                )

                op_s, _, _ = allocate_batch(
                    trial_batches[mid],
                    key,
                    mid_floor,
                    dur,
                    batch_cap(mid),
                    machine_capacity=multi.capacities[mid],
                )

                proposed.append(op_s)

            next_start = max(proposed)

            if next_start <= start + EPS:
                break

            start = next_start

        else:
            raise RuntimeError(
                f"节点 {sid} 批次对齐未收敛"
            )

        # 仅在这里正式登记一次设备占用
        for it in ops:
            mid = int(it[0])
            dur = float(it[1])

            s, e, bid = allocate_batch(
                machine_batches[mid],
                op_key(sid, it),
                start,
                dur,
                batch_cap(mid),
                machine_capacity=multi.capacities[mid],
            )

            if abs(s - start) > EPS:
                raise RuntimeError(
                    f"节点 {sid} 设备 {mid} 无法同步开始"
                )

            if len(it) >= 5:
                temp, heat, mode = it[2:5]
            else:
                temp, heat, mode = 0, 0, 0

            records.append((
                sid, mid, bid,
                start, start + dur,
                temp, heat, mode,
            ))

        node_start[sid] = start
        node_end[sid] = start + max_dur

        if need_human and max_dur > EPS:
            human_busy.append(
                (start, node_end[sid])
            )

        scheduled += 1

        for v in multi.successors[sid]:
            if v not in indeg:
                continue

            indeg[v] -= 1

            if indeg[v] == 0:
                heapq.heappush(
                    ready,
                    (-rank[v], v),
                )

    if scheduled != len(pending):
        raise RuntimeError(
            f"调度失败：{scheduled}/{len(pending)} "
            "个待执行节点完成"
        )

    return MultiScheduleResult(
        node_start=node_start,
        node_end=node_end,
        op_records=records,
        makespan=max(node_end, default=0.0),
        algo_time_ms=(
            time.perf_counter() - t0
        ) * 1000.0,
    )




def optimize_human_work_streak(
    multi, res, snapshot=None, min_rest_minutes=5.0,
    max_continuous_minutes=None, solver_seconds=10.0,
    tick_per_minute=10, verbose=True,
):
    """第三层优化：保持总完工时间、同出锅差及共享批次不变，缩短最长连续人工工作时段。

    相邻人工工序间的空档 < min_rest_minutes 视为同一段连续工作。
    第三层保持上两层求得的共享分组（但允许改变批次执行时间/顺序），不保证全局第三目标最优。
    """
    import time
    from collections import defaultdict
    from ortools.sat.python import cp_model

    t0 = time.perf_counter()
    scale = int(tick_per_minute)
    if scale <= 0 or min_rest_minutes <= 0:
        raise ValueError("时间精度和最小休息时间必须大于0")
    tick = lambda x: int(round(float(x) * scale))
    minute = lambda x: round(float(x) / scale, 6)
    n = multi.N
    if n == 0:
        return res
    now = tick(snapshot.now) if snapshot is not None else 0
    frozen = ({i for i, st in snapshot.node_states.items()
               if st.status in ('done', 'running')}
              if snapshot is not None else set())
    fixed_s = [tick(t) for t in res.node_start]
    fixed_e = [tick(t) for t in res.node_end]
    duration = [e - s for s, e in zip(fixed_s, fixed_e)]
    if any(d < 0 for d in duration):
        raise ValueError("节点结束时间不能早于开始时间")
    horizon = max(max(fixed_e) + tick(120),
                  now + sum(max(0, d) for d in duration) + tick(120))
    rest = max(1, tick(min_rest_minutes))
    # 自动预热属于真实设备占用。第三阶段固定所有热设备工序，
    # 避免移动蒸箱/烤箱节点后破坏前两阶段的设备热状态约束。
    thermal_nodes = {
        nid
        for nid in range(n)
        if any(
            any(x in str(getattr(multi.machines[int(op[0])], "note", "") or "")
                for x in ("蒸箱", "烤箱"))
            for op in multi.nodes[nid]
        )
    } if getattr(res, "preheat_records", None) else set()
    model = cp_model.CpModel()
    S = [model.NewIntVar(0, horizon, f'h_s_{i}') for i in range(n)]
    E = [model.NewIntVar(0, horizon, f'h_e_{i}') for i in range(n)]
    for i in range(n):
        model.Add(E[i] == S[i] + duration[i])
        if i in frozen or i in thermal_nodes:
            model.Add(S[i] == fixed_s[i])
        else:
            model.Add(S[i] >= now)
    for a, b in multi.edges:
        model.Add(S[b] >= E[a])

    # 将同一真实共享批次建成一个设备区间；拆分后不允许伪装成共享。
    groups = defaultdict(list)
    for rec in res.op_records:
        nid, mid, bid, os, oe = rec[:5]
        if oe > os + 1e-7:
            groups[(mid, bid)].append(rec)
    by_machine = defaultdict(list)
    for (mid, bid), ops in groups.items():
        members = {rec[0] for rec in ops}
        if len(members) != len(ops):
            # 一个节点的同一设备出现重复的同批记录，不能安全二次优化。
            raise ValueError(f"共享批次 {(mid, bid)} 内同一节点重复")
        representative = next(iter(members))
        for i in members:
            model.Add(S[i] == S[representative])
        batch_dur = max(tick(rec[4] - rec[3]) for rec in ops)
        interval = model.NewIntervalVar(
            S[representative], batch_dur,
            S[representative] + batch_dur, f'h_batch_{mid}_{bid}'
        )
        by_machine[mid].append(interval)
    for mid, intervals in by_machine.items():
        model.AddCumulative(intervals, [1] * len(intervals),
                            max(1, int(multi.capacities[mid])))

    # 人工区间只取“人工/手动操作”设备操作本身的时长。
    # 煮/蒸节点可能持续30分钟，但人工只占前2分钟。
    manual_mids = {
        mid for mid, machine in enumerate(multi.machines)
        if any(k in str(getattr(machine, "note", "") or "")
               for k in ("人工", "手动操作"))
    }
    human_minutes = []
    for nid in range(n):
        spans = [tick(op[1]) for op in multi.nodes[nid]
                 if int(op[0]) in manual_mids]
        human_minutes.append(min(duration[nid], max(spans, default=0)))

    # 对缺少人工设备记录的旧排程采取保守兜底。
    for nid in range(n):
        if human_minutes[nid] == 0 and bool(multi.needs_human[nid]):
            if not any(int(op[0]) in manual_mids for op in multi.nodes[nid]):
                human_minutes[nid] = duration[nid]

    def is_cancelled_placeholder(nid):
        """Only exclude synthetic cancellations, not genuine completed manual work."""
        if snapshot is None or nid not in frozen:
            return False
        st = snapshot.node_states[nid]
        return (
            st.status == "done"
            and not st.machine_assignments
            and abs(float(st.start)) < 1e-7
            and abs(float(st.end) - float(snapshot.now)) < 1e-7
        )

    human = [nid for nid in range(n)
             if human_minutes[nid] > 0 and not is_cancelled_placeholder(nid)]
    intervals = [model.NewIntervalVar(
        S[nid], human_minutes[nid], S[nid] + human_minutes[nid],
        f'human_{nid}') for nid in human]
    if intervals:
        model.AddNoOverlap(intervals)

    # 使用稳定的全局节点ID恢复工艺时间窗，不能依赖可重复的菜名/步骤名。
    # 旧排程只有名称时，只有存在唯一可验证的DAG+时间匹配才采用。
    info = getattr(res, "optimize_info", None) or {}
    by_name = defaultdict(list)
    for nid, step_name in enumerate(multi.step_names):
        by_name[str(step_name)].append(nid)

    def reachable(a, b):
        todo = [a]
        seen = {a}
        while todo:
            u = todo.pop()
            if u == b:
                return True
            for v in multi.successors[u]:
                if v not in seen:
                    seen.add(v)
                    todo.append(v)
        return False

    for spec in info.get("bounded_gaps", ()):
        if isinstance(spec, dict):
            gap = spec.get("max_gap_minutes")
            a_id = spec.get("predecessor_id")
            b_id = spec.get("successor_id")
            a_name = spec.get("predecessor")
            b_name = spec.get("successor")
        else:
            if len(spec) < 3:
                raise ValueError(f"非法工艺时间窗：{spec!r}")
            a_name, b_name, gap = spec[:3]
            a_id = b_id = None

        if gap is None or float(gap) < 0:
            raise ValueError(f"工艺最大等待时间无效：{spec!r}")
        limit = tick(gap)

        # 新版元数据：直接按照唯一节点ID定位。
        if a_id is not None and b_id is not None:
            a_id, b_id = int(a_id), int(b_id)
            if not (0 <= a_id < n and 0 <= b_id < n):
                raise ValueError(f"工艺时间窗节点ID超界：{spec!r}")
            if a_id == b_id:
                raise ValueError(f"工艺时间窗前后节点相同：{spec!r}")
            pairs = [(a_id, b_id)]
        else:
            # 兼容此前仅使用名称的元数据：允许名称重复，但不猜节点。
            # 先按当前DAG的可达性，再按原排程是否满足时间窗确定唯一一对。
            aa = by_name.get(str(a_name), ())
            bb = by_name.get(str(b_name), ())
            pairs = [
                (a, b)
                for a in aa for b in bb
                if a != b
                and reachable(a, b)
                and fixed_e[a] <= fixed_s[b]
                and fixed_s[b] - fixed_e[a] <= limit
            ]
            if len(pairs) != 1:
                if verbose:
                    print(
                        "[阶段3] 无法唯一还原旧版工艺时间窗："
                        f"{a_name} -> {b_name}，"
                        f"源节点候选{len(aa)}个、目标节点候选{len(bb)}个、"
                        f"合法组合{len(pairs)}个。"
                        "安全保留第二阶段结果；新排程请使用节点ID元数据。"
                    )
                return res

        a_id, b_id = pairs[0]
        # 为避免旧快照结构不一致导致错误约束，校验输入本身已符合该时间窗。
        if not (fixed_e[a_id] <= fixed_s[b_id]
                <= fixed_e[a_id] + limit):
            if verbose:
                print(
                    f"[阶段3] 输入排程不满足工艺时间窗 "
                    f"{a_id}->{b_id}，保留第二阶段结果"
                )
            return res
        model.Add(S[b_id] >= E[a_id])
        model.Add(S[b_id] <= E[a_id] + limit)

    # 不允许恶化前两层已经实现的目标值。
    old_makespan = max(fixed_e)
    for i in range(n):
        model.Add(E[i] <= old_makespan)
    recipes = defaultdict(list)
    for i, rid in enumerate(multi.recipe_of_node):
        recipes[rid].append(i)
    finish = []
    for rid, ids in recipes.items():
        f = model.NewIntVar(0, horizon, f'finish_{rid}')
        model.AddMaxEquality(f, [E[i] for i in ids])
        finish.append(f)
    old_recipe_finishes = [max(fixed_e[i] for i in ids) for ids in recipes.values()]
    old_spread = max(old_recipe_finishes) - min(old_recipe_finishes)
    top = model.NewIntVar(0, horizon, 'high')
    bottom = model.NewIntVar(0, horizon, 'low')
    model.AddMaxEquality(top, finish)
    model.AddMinEquality(bottom, finish)
    model.Add(top - bottom <= old_spread)

    # 优化动态重排后仍可调整的人工工作；过去已完成的工作不能改动。
    adjustable = [i for i in human if i not in frozen]
    if not adjustable:
        return res

    # 最近一个冻结的人工步骤作为历史边界，计入已连续工作的长度。
    history = sorted((i for i in human if i in frozen),
                     key=lambda i: (fixed_s[i] + human_minutes[i], fixed_s[i]))
    anchor = None
    anchor_work = 0
    if snapshot is not None and history:
        last = history[-1]
        if fixed_s[last] + human_minutes[last] >= now - rest:
            anchor = last
            anchor_work = human_minutes[last]
            cursor = fixed_s[last]
            for i in reversed(history[:-1]):
                if cursor - (fixed_s[i] + human_minutes[i]) >= rest:
                    break
                anchor_work += human_minutes[i]
                cursor = fixed_s[i]

    if len(adjustable) == 1 and anchor is None:
        return res
    jobs = ([anchor] if anchor is not None else []) + adjustable
    m = len(jobs)
    work_durations = [anchor_work if i == anchor else human_minutes[i] for i in jobs]
    max_possible_work = sum(work_durations)
    # 第0点表示虚拟起点，1..m为需要排序的人工工序。
    arcs = []
    arc_vars = {}
    for j in range(1, m + 1):
        first = model.NewBoolVar(f'first_{j}')
        last = model.NewBoolVar(f'last_{j}')
        arcs.extend([(0, j, first), (j, 0, last)])
        arc_vars[(0, j)] = first
    streak = [model.NewIntVar(0, max_possible_work, f'streak_{j}')
              for j in range(m)]
    for j, nid in enumerate(jobs):
        model.Add(streak[j] == work_durations[j]).OnlyEnforceIf(arc_vars[(0, j + 1)])
    if anchor is not None:
        model.Add(arc_vars[(0, 1)] == 1)

    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            i, j = jobs[a], jobs[b]
            follows = model.NewBoolVar(f'after_{a}_{b}')
            arcs.append((a + 1, b + 1, follows))
            model.Add(S[j] >= S[i] + human_minutes[i]).OnlyEnforceIf(follows)
            has_rest = model.NewBoolVar(f'rest_{a}_{b}')
            model.Add(S[j] >= S[i] + human_minutes[i] + rest).OnlyEnforceIf([follows, has_rest])
            model.Add(S[j] <= S[i] + human_minutes[i] + rest - 1).OnlyEnforceIf([follows, has_rest.Not()])
            model.Add(streak[b] == work_durations[b]).OnlyEnforceIf([follows, has_rest])
            model.Add(streak[b] == streak[a] + work_durations[b]).OnlyEnforceIf(
                [follows, has_rest.Not()]
            )
    model.AddCircuit(arcs)
    max_streak = model.NewIntVar(max(work_durations), max_possible_work,
                                 'max_continuous_human')
    model.AddMaxEquality(max_streak, streak)
    if max_continuous_minutes is not None:
        if tick(max_continuous_minutes) < max(work_durations):
            raise ValueError("目标最长连续工作时间短于单道人工工序，请调大上限")
        model.Add(max_streak <= tick(max_continuous_minutes))

    # 用足够大的权重优先缩短最长连续工作，再降低对现有计划的修改幅度。
    drifts = []
    for i in range(n):
        if i in frozen:
            continue
        d = model.NewIntVar(0, horizon, f'drift_{i}')
        model.AddAbsEquality(d, S[i] - fixed_s[i])
        drifts.append(d)
    weight = len(drifts) * horizon + 1
    model.Minimize(max_streak * weight + sum(drifts))
    engine = cp_model.CpSolver()
    engine.parameters.max_time_in_seconds = max(0.2, float(solver_seconds))
    engine.parameters.num_search_workers = 8
    engine.parameters.random_seed = 42
    code = engine.Solve(model)
    if code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if verbose:
            print(f'[第三阶段 人工连续工作] 未得到可行改进：{engine.StatusName(code)}，保留二阶段结果')
        return res

    starts = [minute(engine.Value(v)) for v in S]
    ends = [minute(engine.Value(v)) for v in E]
    new_records = []
    for rec in res.op_records:
        nid = rec[0]
        span = rec[4] - rec[3]
        new_records.append((rec[0], rec[1], rec[2], starts[nid],
                            round(starts[nid] + span, 6), *rec[5:]))
    result = MultiScheduleResult(starts, ends, new_records, max(ends),
                                 (time.perf_counter() - t0) * 1000.0)
    # 保留自动预热明细，供冲突校验、动态快照和甘特图继续使用。
    result.preheat_records = [dict(p) for p in (getattr(res, "preheat_records", ()) or ())]
    report = check_multi_schedule(multi, result)
    if report['has_any']:
        raise AssertionError(f'人工优化后违反设备/依赖条件: {report}')
    # 最终校验整个人工时间轴；不能仅依赖已有的设备DAG冲突检查。
    manual_intervals = sorted((starts[i], starts[i] + minute(human_minutes[i])) for i in human)
    for (_, pe), (ns, _) in zip(manual_intervals, manual_intervals[1:]):
        if ns < pe - 1e-6:
            raise AssertionError('人工资源重叠')

    def longest_run(starts_, ends_):
        intervals_ = sorted((starts_[i], starts_[i] + minute(human_minutes[i])) for i in human)
        if not intervals_:
            return 0.0
        best = total = intervals_[0][1] - intervals_[0][0]
        for (prev_s, prev_e), (ns, ne) in zip(intervals_, intervals_[1:]):
            if ns - prev_e < min_rest_minutes - 1e-7:
                total += ne - ns
            else:
                total = ne - ns
            best = max(best, total)
        return best

    before = longest_run(res.node_start, res.node_end)
    after = longest_run(starts, ends)
    if after > before + 1e-6:
        # 极端情况：已有历史人工连续段比优化时的前向边界更长。
        if verbose:
            print('[第三阶段] 完整历史人工连续时长没有改善，保留原排程')
        return res
    if getattr(res, 'optimize_info', None) is not None:
        result.optimize_info = dict(res.optimize_info)
    else:
        result.optimize_info = {}
    result.optimize_info.update({
        'stage3_status': engine.StatusName(code),
        'human_min_rest_minutes': float(min_rest_minutes),
        'human_longest_streak_before': round(before, 3),
        'human_longest_streak_after': round(after, 3),
        'stage3_optimal_proven': code == cp_model.OPTIMAL,
        'stage3_preserves_batch_membership': True,
    })
    if verbose:
        print(f'[阶段3 人工连续工作] {before:.1f} -> {after:.1f} 分钟; '
              f'总时长={result.makespan:.1f}分钟，状态={engine.StatusName(code)}')
    return result



def align_recipe_finish(
    multi,
    res,
    target=None,
    max_finish_gap=2.0,
    search_step=0.2,
    critical_eps=1e-6,
    strict=False,
    verbose=True,
    max_recipe_start_delay=25.0,
    max_recipe_start_delay_ratio=0.20,
    freeze_recipe_roots=True,
    force_unshare=False,
):
    """
    同步出锅：
    - 优先保持makespan
    - 尽量减小各菜完成时间差
    - 支持共享批次
    - 必要时允许把共享批次拆成独立批次
    """
    from collections import defaultdict, deque
    import math

    EPS = 1e-6
    n = multi.N

    if n == 0:
        return res, {
            "target": 0.0,
            "finish_times": {},
            "finish_spread": 0.0,
            "aligned": True,
        }

    s = list(res.node_start)
    e = list(res.node_end)
    dur = [e[i] - s[i] for i in range(n)]

    recipes = defaultdict(list)

    for i, rid in enumerate(multi.recipe_of_node):
        recipes[rid].append(i)

    before = {
        rid: max(e[i] for i in ids)
        for rid, ids in recipes.items()
    }

    original_start = {
        rid: min(s[i] for i in ids)
        for rid, ids in recipes.items()
    }

    goal = (
        max(before.values())
        if target is None
        else float(target)
    )
    goal = min(goal, float(res.makespan))

    anchors = {
        rid for rid in recipes
        if abs(before[rid] - goal) <= critical_eps
    }

    roots = {
        rid: {
            i for i in ids
            if not any(
                p in ids for p in multi.predecessors[i]
            )
        }
        for rid, ids in recipes.items()
    }

    caps = {
        rid: original_start[rid] + min(
            max_recipe_start_delay,
            max(0, goal - before[rid])
            * max_recipe_start_delay_ratio,
        )
        for rid in recipes
    }

    for rid in anchors:
        caps[rid] = original_start[rid]

    # 拓扑排序
    indeg = list(multi.indegree)
    q = deque(
        i for i in range(n)
        if indeg[i] == 0
    )

    topo = []

    while q:
        i = q.popleft()
        topo.append(i)

        for v in multi.successors[i]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(topo) != n:
        raise ValueError("工序存在循环依赖")

    # 保护关键路径
    critical = set()

    for rid in anchors:
        latest = {}
        ids = set(recipes[rid])

        for i in reversed(topo):
            if i not in ids:
                continue

            succ = [
                v for v in multi.successors[i]
                if v in ids
            ]

            latest[i] = (
                min(latest[v] for v in succ)
                if succ
                else before[rid]
            ) - dur[i]

            if latest[i] - s[i] <= critical_eps:
                critical.add(i)

    def feasible(starts, ends):
        # 依赖关系
        for p, v in multi.edges:
            if ends[p] > starts[v] + EPS:
                return False

        # 按设备及批次检查物理占用
        by_device = defaultdict(dict)

        for rec in res.op_records:
            nid, mid, bid, _, _ = rec[:5]

            a = starts[nid]
            b = a + (rec[4] - rec[3])

            if b <= a + EPS:
                continue

            # 错峰以后视为不同批次
            group_key = (
                nid if force_unshare else bid,
                round(a, 6),
            )

            old = by_device[mid].get(group_key)

            if old is None:
                by_device[mid][group_key] = (a, b)
            else:
                by_device[mid][group_key] = (
                    min(a, old[0]),
                    max(b, old[1]),
                )

        for mid, groups in by_device.items():
            events = []

            for a, b in groups.values():
                events.extend([
                    (a, 1),
                    (b, -1),
                ])

            events.sort(key=lambda x: (x[0], x[1]))
            active = 0

            for _, change in events:
                active += change

                if active > multi.capacities[mid]:
                    return False

        # 人工资源容量检查
        human = []

        for i in range(n):
            if (
                multi.needs_human[i]
                and ends[i] > starts[i] + EPS
            ):
                human.extend([
                    (starts[i], 1),
                    (ends[i], -1),
                ])

        human.sort(key=lambda x: (x[0], x[1]))
        active = 0

        for _, change in human:
            active += change
            if active > 1:
                return False

        return True

    def passive(i):
        name = (
            multi.step_names[i].split("#")[-1]
            if hasattr(multi, "step_names")
            else ""
        )

        return any(
            x in name
            for x in (
                "等待", "静置", "浸泡",
                "冷却", "醒发", "发酵", "腌制",
            )
        )

    moved = []
    step = max(0.1, float(search_step))

    # 两轮逆拓扑ALAP调整
    for _ in range(2):
        for is_passive in (True, False):
            for i in reversed(topo):
                rid = multi.recipe_of_node[i]

                if i in critical:
                    continue

                if (
                    freeze_recipe_roots
                    and i in roots[rid]
                ):
                    continue

                if passive(i) != is_passive:
                    continue

                old_s = s[i]
                old_e = e[i]

                upper = min(
                    (
                        s[v]
                        for v in multi.successors[i]
                    ),
                    default=goal,
                ) - dur[i]

                if upper <= old_s + EPS:
                    continue

                for k in range(
                    int(
                        math.ceil(
                            (upper - old_s) / step
                        )
                    ) + 1
                ):
                    candidate = upper - k * step

                    if candidate <= old_s + EPS:
                        break

                    s[i] = candidate
                    e[i] = candidate + dur[i]

                    start_ok = (
                        min(
                            s[j] for j in recipes[rid]
                        ) <= caps[rid] + EPS
                    )

                    if start_ok and feasible(s, e):
                        moved.append({
                            "nid": i,
                            "recipe_id": rid,
                            "old_start": old_s,
                            "new_start": candidate,
                            "delay": candidate - old_s,
                        })
                        break

                if not (
                    s[i] > old_s + EPS
                    and feasible(s, e)
                ):
                    s[i] = old_s
                    e[i] = old_e

    if not feasible(s, e):
        raise AssertionError(
            "同步对齐后存在工序/设备/人工冲突"
        )

    # 更新批次ID
    starts_per_batch = defaultdict(set)

    for rec in res.op_records:
        starts_per_batch[
            (rec[1], rec[2])
        ].add(
            round(s[rec[0]], 6)
        )

    id_map = {}

    next_id = max(
        (
            rec[2]
            for rec in res.op_records
            if isinstance(rec[2], int)
        ),
        default=0,
    ) + 1

    for old_key, begin_times in (
        starts_per_batch.items()
    ):
        for j, t in enumerate(
            sorted(begin_times)
        ):
            id_map[(old_key, t)] = (
                old_key[1] if j == 0 else next_id
            )

            next_id += (j != 0)

    op_records = []

    for rec in res.op_records:
        nid, mid, bid = rec[:3]

        start = s[nid]
        end = start + (
            rec[4] - rec[3]
        )

        assigned_bid = id_map[
            ((mid, bid), round(start, 6))
        ]

        op_records.append(
            (
                nid, mid, assigned_bid,
                start, end,
            ) + tuple(rec[5:])
        )

    new_res = MultiScheduleResult(
        node_start=s,
        node_end=e,
        op_records=op_records,
        makespan=max(e),
        algo_time_ms=res.algo_time_ms,
    )

    for attr in (
        "frozen_nodes", "changed_nodes"
    ):
        if hasattr(res, attr):
            setattr(
                new_res,
                attr,
                getattr(res, attr),
            )

    after = {
        rid: max(e[i] for i in ids)
        for rid, ids in recipes.items()
    }

    spread = (
        max(after.values())
        - min(after.values())
    )

    info = {
        "target": goal,
        "anchor_recipes": sorted(anchors),
        "critical_nodes": sorted(critical),
        "moved_nodes": moved,
        "recipe_start_before": original_start,
        "recipe_start_after": {
            rid: min(s[i] for i in ids)
            for rid, ids in recipes.items()
        },
        "recipe_start_cap": caps,
        "finish_times": after,
        "finish_spread": spread,
        "max_finish_gap": max_finish_gap,
        "aligned": spread <= max_finish_gap + EPS,
    }

    # 共享限制了同步时，额外探索一个拆批候选方案
    if (
        not force_unshare
        and spread > max_finish_gap + EPS
    ):
        try:
            candidate, detail = align_recipe_finish(
                multi,
                res,
                target=goal,
                max_finish_gap=max_finish_gap,
                search_step=step,
                critical_eps=critical_eps,
                strict=False,
                verbose=False,
                max_recipe_start_delay=max_recipe_start_delay,
                max_recipe_start_delay_ratio=max_recipe_start_delay_ratio,
                freeze_recipe_roots=freeze_recipe_roots,
                force_unshare=True,
            )

            if (
                candidate.makespan
                <= new_res.makespan + EPS
                and detail["finish_spread"]
                < spread - EPS
                and not check_multi_schedule(
                    multi, candidate
                )["has_any"]
            ):
                new_res = candidate
                info = detail

        except AssertionError:
            pass

    if verbose:
        print(
            f"[同步出锅] "
            f"makespan={new_res.makespan:.1f}分钟 "
            f"完成时间差={info['finish_spread']:.1f}分钟"
        )

    if strict and not info["aligned"]:
        raise AssertionError(
            f"完成时间差 {info['finish_spread']:.1f} "
            f"超过允许的 {max_finish_gap:.1f}"
        )

    return new_res, info


def merge_shared_preheat_nodes(multi):
    """
    相同设备、相同温度/模式的预热只物理执行一次。

    关键原则：
    1. canonical 预热：真正占用设备、有时长
    2. duplicate 预热：保留节点和原来的 DAG 边，但变成 0 时长同步节点
    3. canonical -> duplicate
    4. 绝对不删除 duplicate 原来的 predecessor / successor
    """

    from collections import defaultdict

    def is_preheat(nid):
        name = (
            multi.step_names[nid].split("#")[-1]
            if hasattr(multi, "step_names")
            else ""
        )
        return "预热" in name

    def signature(nid):
        """
        生成预热共享签名。

        关键：
        预热节点自己的 mode/temp 可能被 LLM 解析错，
        因此优先从后续真正的蒸/烤工序读取设备状态。
        """

        if not is_preheat(nid):
            return None

        if not multi.nodes[nid]:
            return None

        sig = []

        for item in multi.nodes[nid]:

            mid = int(item[0])

            # --------------------------------------------------
            # 1. 先读取预热本身参数
            # --------------------------------------------------
            if len(item) >= 5:
                temp = item[2]
                heat = item[3]
                mode = item[4]
            else:
                temp = 0
                heat = 0
                mode = 0

            # --------------------------------------------------
            # 2. 优先使用后继正式烹饪步骤的参数
            # --------------------------------------------------
            for succ in multi.successors[nid]:

                succ_name = (
                    multi.step_names[succ].split("#")[-1]
                    if hasattr(multi, "step_names")
                    else ""
                )

                # 跳过其他预热/同步节点
                if "预热" in succ_name:
                    continue

                for succ_item in multi.nodes[succ]:

                    succ_mid = int(succ_item[0])

                    # 必须是同一台设备
                    if succ_mid != mid:
                        continue

                    if len(succ_item) >= 5:

                        succ_temp = succ_item[2]
                        succ_heat = succ_item[3]
                        succ_mode = succ_item[4]

                        # 后继参数优先
                        if succ_temp:
                            temp = succ_temp

                        if succ_heat:
                            heat = succ_heat

                        if succ_mode not in (None, 0, "0", "none", ""):
                            mode = succ_mode

                    break

            # --------------------------------------------------
            # 3. 做一次兜底标准化
            # --------------------------------------------------

            # 蒸箱：100℃且模式缺失，默认普通蒸
            if mid == 2:
                if mode in (None, 0, "0", "none", ""):
                    if 91 <= float(temp) <= 100:
                        mode = 11  # 普通蒸

            # 烤箱：模式缺失
            if mid == 3:
                if mode in (None, 0, "0", "none", ""):
                    # 默认上下火
                    mode = 1

            sig.append((
                mid,
                int(temp),
                int(heat),
                mode,
            ))

        return tuple(sorted(sig, key=str))

    groups = defaultdict(list)

    for nid in range(multi.N):
        sig = signature(nid)

        if sig is not None:
            groups[sig].append(nid)

    edges = set(tuple(e) for e in multi.edges)

    merged_count = 0

    for sig, nids in groups.items():

        if len(nids) <= 1:
            continue

        # 第一个节点作为真正的物理预热
        canonical = min(nids)

        print(
            f"[共享预热] canonical=N{canonical}, "
            f"group={nids}, signature={sig}"
        )

        for dup in nids:

            if dup == canonical:
                continue

            # --------------------------------------------------
            # ★ 不删除任何原来的边
            #
            # 原来：
            # prep -> dup -> cooking
            #
            # 保持不变，再增加：
            # canonical -> dup
            #
            # 这样 dup 成为 AND 同步门
            # --------------------------------------------------

            edges.add((canonical, dup))

            # duplicate 不再真正使用设备
            multi.nodes[dup] = []

            if hasattr(multi, "needs_human"):
                multi.needs_human[dup] = False

            # 名称不要包含“预热”
            # 防止后续 rescheduler 再把它识别成物理预热
            old_name = multi.step_names[dup]

            if "#" in old_name:
                recipe = old_name.split("#", 1)[0]
            else:
                recipe = old_name

            multi.step_names[dup] = (
                f"{recipe}#共享预热同步"
            )

            merged_count += 1

        # canonical 名字美化
        if sig:
            mid = sig[0][0]
            temp = sig[0][1]

            try:
                machine_name = (
                    multi.machines[mid].note
                    or f"设备{mid}"
                )
            except Exception:
                machine_name = f"设备{mid}"

            if temp:
                label = f"共享{machine_name}{temp}℃预热"
            else:
                label = f"共享{machine_name}预热"

            multi.step_names[canonical] = (
                f"共享设备#{label}"
            )

    multi.edges = sorted(edges)
    multi.E = len(multi.edges)

    # 重新计算 predecessor / successor / indegree / duration
    multi.build()

    print(
        f"[共享预热] 合并 {merged_count} 个重复物理预热"
    )

    return multi


def validate_recipe_dag(inst, step_names):
    problems = []

    for nid, name in enumerate(step_names):

        # 正式蒸/烤节点
        is_cooking = (
            ("蒸" in name and "预热" not in name)
            or ("烘烤" in name)
            or ("烤" in name and "预热" not in name)
        )

        if not is_cooking:
            continue

        preds = inst.predecessors[nid]

        non_preheat_preds = [
            p for p in preds
            if "预热" not in step_names[p]
        ]

        if not non_preheat_preds:
            problems.append(
                f"{name} 只有预热前置，"
                f"缺少食材准备/装盘等烹饪前置"
            )

    if problems:
        print("\n[DAG 警告]")
        for x in problems:
            print("  -", x)

    return problems

def check_multi_schedule(multi, res):
    """按物理批次核对设备容量、共享兼容性和节点依赖。"""
    from collections import defaultdict
    EPS = 1e-7
    conflicts, dep_violations = [], []
    groups = defaultdict(list)
    for rec in res.op_records:
        nid, mid, bid, s, e = rec[:5]
        if e > s + EPS:
            groups[(mid, bid)].append(rec)

    by_machine = defaultdict(list)
    for (mid, bid), records in groups.items():
        start = min(r[3] for r in records)
        end = max(r[4] for r in records)
        by_machine[mid].append((bid, start, end))
        name = str(getattr(multi.machines[mid], "note", "") or "")
        cap = 3 if any(x in name for x in ("蒸箱", "烤箱", "冰箱")) else 1
        if len(records) > cap:
            conflicts.append({"machine_id": mid, "batch_id": bid, "type": "batch_capacity", "overlap": 0.0})
        kinds = set()
        params = set()
        for rec in records:
            nid = rec[0]
            step = str(multi.step_names[nid]).split("#")[-1] if hasattr(multi, "step_names") else ""
            kinds.add("预热" if "预热" in step else "操作")
            params.add(tuple(rec[5:8]) if len(rec) >= 8 else (0, 0, 0))
            if abs(rec[3] - start) > EPS:
                conflicts.append({"machine_id": mid, "batch_id": bid, "type": "batch_not_co_started", "overlap": 0.0})
        if len(params) != 1 or len(kinds) != 1:
            conflicts.append({"machine_id": mid, "batch_id": bid, "type": "incompatible_shared_conditions", "overlap": 0.0})

    # 自动预热占据与烹饪相同的真实设备，必须计入物理冲突校验。
    for i, p in enumerate(getattr(res, "preheat_records", ()) or ()):
        mid = int(p["machine_id"])
        start, end = float(p["start"]), float(p["end"])
        if end <= start + EPS:
            conflicts.append({"machine_id": mid, "type": "invalid_auto_preheat",
                              "time": start, "overlap": 0.0})
        else:
            by_machine[mid].append((f"autoheat_{i}", start, end))

    for mid, batches in by_machine.items():
        events = []
        for bid, s, e in batches:
            events.extend(((s, 1, bid), (e, -1, bid)))
        events.sort(key=lambda x: (x[0], x[1]))  # 先结束再开始
        active = set()
        cap = max(1, int(multi.capacities[mid]))
        for t, delta, bid in events:
            if delta == -1:
                active.discard(bid)
            else:
                active.add(bid)
                if len(active) > cap:
                    conflicts.append({
                        "machine_id": mid, "batch_id": bid, "type": "physical_capacity",
                        "time": t, "active_batches": len(active), "capacity": cap,
                        "overlap": 0.0,
                    })

    for nid in range(multi.N):
        for p in multi.predecessors[nid]:
            if res.node_end[p] > res.node_start[nid] + EPS:
                dep_violations.append({
                    "node": nid, "pred": p,
                    "pred_end": res.node_end[p], "node_start": res.node_start[nid],
                    "delay": res.node_end[p] - res.node_start[nid],
                })

    score = 100.0 * len(conflicts) + 1000.0 * len(dep_violations)
    return {
        "conflicts": conflicts,
        "dependency_violations": dep_violations,
        "score": score,
        "has_any": bool(conflicts or dep_violations),
    }

# ============================================================
# 5. 理论下界
# ============================================================
def optimize_human_work_streak_safe(multi, res, snapshot=None, **kwargs):
    """运行原人工优化器，但只接受不移动蒸箱/烤箱节点的候选，保持自动预热有效。

    目前人工阶段属于独立求解器；如果它改变热设备工序顺序或时刻，
    就回退到已验证的双目标方案，避免破坏物理预热。
    """
    thermal_nodes = set()
    for nid in range(multi.N):
        for op in multi.nodes[nid]:
            name = str(getattr(multi.machines[int(op[0])], "note", "") or "")
            if "蒸箱" in name or "烤箱" in name:
                thermal_nodes.add(nid)

    base_optimizer = globals().get("optimize_human_work_streak")
    if base_optimizer is None:
        return res
    candidate = base_optimizer(multi, res, snapshot=snapshot, **kwargs)
    if candidate is not res and any(
        abs(candidate.node_start[i] - res.node_start[i]) > 1e-7 or
        abs(candidate.node_end[i] - res.node_end[i]) > 1e-7
        for i in thermal_nodes
    ):
        if kwargs.get("verbose", True):
            print("[人工阶段] 为避免破坏设备预热/保温关系，保留前两阶段排程")
        return res

    candidate.preheat_records = [dict(p) for p in getattr(res, "preheat_records", ())]
    candidate.optimize_info = {
        **(getattr(res, "optimize_info", None) or {}),
        **(getattr(candidate, "optimize_info", None) or {}),
    }
    check = check_multi_schedule(multi, candidate)
    if check["has_any"]:
        if kwargs.get("verbose", True):
            print("[人工阶段] 发现设备冲突，回退至前两阶段排程")
        return res
    return candidate

def compute_lower_bound(multi):
    """
    多菜调度的安全下界。

    LB_CP: 最长依赖路径，下界节点时长按该节点最长并行操作计算（不能相加）。
    LB_Machine: 设备总负荷 / (物理设备数 * 同批理论容量)。
    注意：共享容积的值是乐观容量；它只用于构造下界，不保证实际可同时开工。
    """
    from collections import deque
    from llm_parser import BATCH_CAPACITY, DEFAULT_BATCH_CAPACITY

    n = multi.N
    indeg = list(multi.indegree)
    q = deque(i for i in range(n) if indeg[i] == 0)
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in multi.successors[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(topo) != n:
        raise ValueError("计算下界失败：工序依赖图包含环")

    duration = [
        max((float(it[1]) for it in multi.nodes[i]), default=0.0)
        for i in range(n)
    ]
    rank = [0.0] * n
    for u in reversed(topo):
        rank[u] = duration[u] + max(
            (rank[v] for v in multi.successors[u]), default=0.0
        )
    lb_cp = max(rank, default=0.0)

    machine_load = [0.0] * multi.K
    for ops in multi.nodes:
        for it in ops:
            machine_load[int(it[0])] += max(0.0, float(it[1]))

    lb_machine = 0.0
    for mid in range(multi.K):
        name = str(getattr(multi.machines[mid], "note", "") or "")
        orig_id = getattr(multi.machines[mid], "id", mid)
        batch_cap = (
            3 if any(word in name for word in ("蒸箱", "烤箱", "冰箱"))
            else BATCH_CAPACITY.get(orig_id, DEFAULT_BATCH_CAPACITY)
            if not name else 1
        )
        parallel_capacity = max(1, int(multi.capacities[mid])) * max(1, int(batch_cap))
        lb_machine = max(lb_machine, machine_load[mid] / parallel_capacity)

    return {
        "lb_cp": lb_cp,
        "lb_machine": lb_machine,
        "lb": max(lb_cp, lb_machine),
        "machine_load": machine_load,
    }



# ============================================================
# 6. 多菜甘特图
# ============================================================
def print_multi_gantt(multi, res, bar_width=70):
    makespan = res.makespan
    if makespan <= 0:
        print("(makespan=0)")
        return

    scale = makespan / bar_width

    by_mb = defaultdict(list)
    for rec in res.op_records:
        nid, mid, bid, s, e = rec[0], rec[1], rec[2], rec[3], rec[4]
        by_mb[(mid, bid)].append((nid, s, e))

    # 菜谱 -> 字母
    chars = {}
    for i in range(len(multi.recipe_names)):
        chars[i] = chr(ord("A") + i % 26)

    line_w = 22 + bar_width + 2
    print()
    print("=" * line_w)
    print(f"多菜并行甘特图  (makespan = {makespan:.1f} 分钟, "
          f"每字符 ≈ {scale:.2f} 分钟)")
    print("=" * line_w)

    # 图例
    legend = "  ".join(
        f"{chars[i]}={multi.recipe_names[i]}"
        for i in range(len(multi.recipe_names))
    )
    print(f"菜谱: {legend}")
    print(f"字符: 字母=该菜谱占用   .=空闲")
    print("-" * line_w)

    # 时间刻度
    ruler = [" "] * bar_width
    num_ticks = 6
    for t in range(num_ticks):
        pos = int(t * (bar_width - 1) / (num_ticks - 1))
        label = str(int(round(t * makespan / (num_ticks - 1))))
        start = min(pos, bar_width - len(label))
        for j, ch in enumerate(label):
            if 0 <= start + j < bar_width:
                ruler[start + j] = ch
    print(" " * 20 + "|" + "".join(ruler) + "|")

    # 每台设备实例一行
    for k in range(multi.K):
        mname = multi.machines[k].note or f"m{k}"
        cap = multi.capacities[k]
        for bid in range(cap):
            ops = []
            for (mid, bid), group in by_mb.items():
                if mid == k:
                    ops.extend(group)
            if not ops and cap > 1 and bid > 0:
                continue  # 多实例时省掉空行

            label = f"[{k}] {mname}"
            if cap > 1:
                label += f"#{bid}"
            label = label[:20].ljust(20)

            bar = ["."] * bar_width
            for nid, s, e in ops:
                rid = multi.recipe_of_node[nid]
                ch = chars[rid]
                i0 = max(0, min(bar_width - 1, int(s / scale)))
                i1 = max(i0 + 1, min(bar_width, int(round(e / scale))))
                for j in range(i0, i1):
                    bar[j] = ch

            print(f"{label}  |" + "".join(bar) + "|")

    print("=" * line_w)
    print()


# ============================================================
# 7. 对比分析
# ============================================================
def print_comparison(multi, res, lb, check, serial_makespan=None):
    print()
    print("=" * 72)
    print("多菜并行调度 - 结果分析")
    print("=" * 72)

    print(f"\n【任务规模】")
    print(f"  菜谱数      : {len(multi.recipe_names)}")
    print(f"  步骤总数 N  : {multi.N}")
    print(f"  依赖边 E    : {multi.E}")
    print(f"  设备种类 K  : {multi.K}")

    print(f"\n【调度结果】")
    print(f"  并行 makespan          : {res.makespan:.1f} 分钟")
    print(f"  调度算法耗时           : {res.algo_time_ms:.3f} ms")

    if serial_makespan and serial_makespan > 0:
        saved = serial_makespan - res.makespan
        speedup = serial_makespan / max(1e-9, res.makespan)
        print(f"  串行总耗时（独立做）    : {serial_makespan:.1f} 分钟")
        print(f"  并行比串行节约时间      : {saved:.1f} 分钟 "
              f"({saved / serial_makespan * 100:.1f}%)")
        print(f"  加速比                 : {speedup:.2f}x")

    # ---------- 人工操作统计（实际资源区间而非节点总时长） ----------
    manual_mids = {
        mid for mid, machine in enumerate(multi.machines)
        if any(k in str(getattr(machine, "note", "") or "")
               for k in ("人工", "手动操作"))
    }
    human_rows = []
    for rec in res.op_records:
        nid, mid, bid, s, e = rec[:5]
        if int(mid) in manual_mids and float(e) > float(s) + 1e-7:
            human_rows.append((int(nid), float(s), float(e)))
    if human_rows:
        human_total = sum(e - s for _, s, e in human_rows)
        human_last_end = max(e for _, s, e in human_rows)
        human_max_dur = max(e - s for _, s, e in human_rows)
        rest_limit = float((getattr(res, 'optimize_info', None) or {})
                           .get('human_min_rest_minutes', 5.0))
        seq = sorted((s, e) for _, s, e in human_rows)
        current = longest = seq[0][1] - seq[0][0]
        for (_ps, pe), (s, e) in zip(seq, seq[1:]):
            if s < pe - 1e-7:
                raise AssertionError('人工记录存在重叠，检查人工机器资源约束')
            current = current + e - s if s - pe < rest_limit - 1e-7 else e - s
            longest = max(longest, current)
        print(f"\n【人工操作（真实占用）】")
        print(f"  人工操作区间数           : {len(human_rows)}")
        print(f"  人工最后操作完成时间     : {human_last_end:.1f} 分钟")
        print(f"  人工实际总占用时间       : {human_total:.1f} 分钟")
        print(f"  单次最长人工操作时长     : {human_max_dur:.1f} 分钟")
        print(f"  最长连续人工工作时长     : {longest:.1f} 分钟")
    else:
        print(f"\n【人工操作】\n  无需要人工的记录")

    print(f"\n【理论下界】")
    print(f"  关键路径下界 LB_CP     : {lb['lb_cp']:.1f} 分钟")
    print(f"  机器负荷下界 LB_Machine: {lb['lb_machine']:.1f} 分钟")
    print(f"  理论下界 LB = max      : {lb['lb']:.1f} 分钟")

    if lb["lb"] > 0:
        gap = (res.makespan - lb["lb"]) / lb["lb"] * 100
        print(f"  与下界的差距           : {gap:.1f}%")

        # ---- 差距归因分析 ----
        print(f"\n【差距归因】")
        diff = res.makespan - lb["lb"]

        # 找出瓶颈设备（负荷最大的设备）
        machine_load = lb.get("machine_load", [])
        if machine_load:
            bottleneck_k = max(
                range(len(machine_load)),
                key=lambda k: machine_load[k] / max(1, multi.capacities[k])
            )
            bottleneck_name = (
                multi.machines[bottleneck_k].note
                if bottleneck_k < len(multi.machines) else f"m{bottleneck_k}"
            )
            bottleneck_ratio = (
                    machine_load[bottleneck_k]
                    / max(1, multi.capacities[bottleneck_k])
            )
            print(f"  瓶颈设备    : {bottleneck_name} "
                  f"(负荷 {bottleneck_ratio:.1f} 分钟)")

        # 判断下界来源
        if lb["lb_cp"] >= lb["lb_machine"]:
            print(f"  下界来源    : 关键路径（LB_CP 主导）")
            print(f"  差距来源    : 多菜对共享设备的排队等待")
        else:
            print(f"  下界来源    : 设备负荷（LB_Machine 主导）")
            print(f"  差距来源    : 设备容量限制，加容量可直接改善")

        # 给出改进方向
        print(f"  改进方向    :")
        if machine_load and lb["lb_machine"] >= lb["lb_cp"] * 0.8:
            print(f"    · 增加瓶颈设备容量可降低 5~15 分钟")
        if gap > 50:
            print(f"    · 差距 {gap:.1f}% 主要来自多菜竞争，"
                  f"属正常范围")
        elif gap > 20:
            print(f"    · 差距 {gap:.1f}% 属中等，"
                  f"可通过更精细的调度压缩")
        else:
            print(f"    · 差距 {gap:.1f}%，调度接近最优")

    print(f"\n【质量校验】")
    print(f"  设备冲突数             : {len(check['conflicts'])}")
    print(f"  依赖违规数             : {len(check['dependency_violations'])}")
    print(f"  总扣分                 : {check['score']:.2f}")

    if check["has_any"]:
        print(f"\n  ⚠️  存在违规，明细（前 5 条）：")
        for c in check["conflicts"][:5]:
            print(f"    冲突: 设备{c['machine_id']} "
                  f"批次{c.get('batch_id', '?')} "
                  f"N{c['node_a']}(end={c['end_a']:.1f}) 与 "
                  f"N{c['node_b']}(start={c['start_b']:.1f}) "
                  f"重叠 {c['overlap']:.1f} 分钟")
        for v in check["dependency_violations"][:5]:
            print(f"    依赖违规: N{v['node']} start={v['node_start']:.1f} "
                  f"< 前置 N{v['pred']} end={v['pred_end']:.1f} "
                  f"延迟 {v['delay']:.1f}")
    else:
        print(f"\n  ✓ 无设备冲突，无依赖违规，调度方案合法")

    # ---------- 各菜品最后一道工序 & 完成时间差 ----------
    recipe_finish = {}  # rid -> 最后完成时刻
    recipe_last_node = {}  # rid -> 最后一道工序的 nid

    for nid in range(multi.N):
        rid = multi.recipe_of_node[nid]
        e = res.node_end[nid]
        if (rid not in recipe_finish) or (e > recipe_finish[rid] + 1e-9):
            recipe_finish[rid] = e
            recipe_last_node[rid] = nid
        elif abs(e - recipe_finish[rid]) <= 1e-9 and rid not in recipe_last_node:
            recipe_last_node[rid] = nid

    if recipe_finish:
        finishes = list(recipe_finish.values())
        diff = max(finishes) - min(finishes)
        first_rid = min(recipe_finish, key=lambda r: recipe_finish[r])
        last_rid = max(recipe_finish, key=lambda r: recipe_finish[r])

        print(f"\n【各菜品最后一道工序】")
        print(f"  {'菜品':<12}{'最后工序':<18}{'开始':>8}{'完成':>8}")
        print("  " + "-" * 48)
        for rid in sorted(recipe_finish.keys()):
            name = multi.recipe_names[rid]
            nid = recipe_last_node.get(rid)
            last_step = (
                multi.step_names[nid].split("#")[-1]
                if nid is not None else "?"
            )
            s = res.node_start[nid] if nid is not None else 0.0
            e = recipe_finish[rid]
            print(f"  {name:<12}{last_step:<18}{s:>8.1f}{e:>8.1f}")

        print(f"\n【菜品完成时间差】")
        print(f"  最早完成 : {multi.recipe_names[first_rid]} "
              f"@ {recipe_finish[first_rid]:.1f} 分钟")
        print(f"  最晚完成 : {multi.recipe_names[last_rid]} "
              f"@ {recipe_finish[last_rid]:.1f} 分钟")
        print(f"  最后工序最大时间差 : {diff:.1f} 分钟 "
              f"({'✓ 优秀 (<5min)' if diff < 5 else '⚠ 待优化'})")

# ============================================================
# 8. 顶层入口
# ============================================================

def optimize_lexicographic_schedule(
    multi, snapshot=None, max_extra_minutes=0.0,
    stage1_seconds=12.0, stage2_seconds=12.0,
    tick_per_minute=10, batch_limit=3,
    terminal_max_idle=5.0, verbose=True,
    preheat_max_gap_minutes=8.0,
    finish_preparation_max_gap_minutes=15.0,
    hot_keep_minutes=15.0,
    oven_preheat_minutes=None,
    steamer_preheat_minutes=None,
):
    """双目标 CP-SAT + 蒸箱/烤箱设备状态热管理。

    预热时长占用真实设备；每个正式烹饪批次仅允许三种入口：
    已有热态复用 / 原菜谱预热 / 自动预热。任何自动预热都放在该批次开始前，
    在 CP-SAT 中占据设备资源，输出至 res.preheat_records（不是凭空画条）。
    同温同模式且设备保持热态时可复用；工况切换或空闲超过 hot_keep_minutes
    需要重新预热。动态重排只允许重排未开始的工序及自动预热。
    """
    import itertools
    import time
    from collections import defaultdict, deque
    from ortools.sat.python import cp_model

    t0 = time.perf_counter()
    unit = int(tick_per_minute)
    if unit < 1 or hot_keep_minutes < 0:
        raise ValueError("tick_per_minute>=1 且 hot_keep_minutes>=0")

    def T(x):
        return int(round(float(x) * unit))

    def M(x):
        return round(float(x) / unit, 6)

    n = multi.N
    if n == 0:
        return MultiScheduleResult([], [], [], 0.0, 0.0)

    degree = list(multi.indegree)
    todo = deque(i for i in range(n) if degree[i] == 0)
    seen = 0
    while todo:
        u = todo.popleft()
        seen += 1
        for v in multi.successors[u]:
            degree[v] -= 1
            if degree[v] == 0:
                todo.append(v)
    if seen != n:
        raise ValueError("存在循环依赖，不能调度")

    now = T(snapshot.now) if snapshot is not None else 0
    states = snapshot.node_states if snapshot is not None else {}
    frozen = {i for i, st in states.items() if st.status in ("done", "running")}
    dur = [max((T(op[1]) for op in multi.nodes[i]), default=0) for i in range(n)]
    horizon = max(
        now + sum(dur) + sum(max(0, T(st.end) - now) for st in states.values()
                             if st.status == "running") + T(120),
        max((T(st.end) for st in states.values()), default=0) + sum(dur) + T(120),
        T(240),
    )
    model = cp_model.CpModel()
    S = [model.NewIntVar(0, horizon, f"S{i}") for i in range(n)]
    E = [model.NewIntVar(0, horizon, f"E{i}") for i in range(n)]
    for i in range(n):
        if i in frozen:
            st = states[i]
            final_e = st.actual_end if st.status == "done" and st.actual_end is not None else st.end
            model.Add(S[i] == T(st.start))
            model.Add(E[i] == T(final_e))
        else:
            model.Add(S[i] >= now)
            model.Add(E[i] == S[i] + dur[i])
    for a, b in multi.edges:
        model.Add(S[b] >= E[a])

    names = [str(x).split("#")[-1].replace(" ", "") for x in multi.step_names]
    by_recipe = defaultdict(list)
    for i, rid in enumerate(multi.recipe_of_node):
        by_recipe[int(rid)].append(i)

    # 保留已经确认的工艺时间窗，避免仅通过延迟装盘伪造出锅一致性。
    bounded = []
    def add_gap(i, j, upper, kind):
        if upper is None:
            return
        model.Add(S[j] >= E[i])
        model.Add(S[j] <= E[i] + T(upper))
        bounded.append((i, j, float(upper), kind))

    def dag_distance(a, b):
        q = deque([(a, 0)])
        visited = {a}
        while q:
            x, step = q.popleft()
            if x == b:
                return step
            for nxt in multi.successors[x]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, step + 1))
        return None

    if preheat_max_gap_minutes is not None:
        for i in range(n):
            if "预热" not in names[i]:
                continue
            device_set = {int(op[0]) for op in multi.nodes[i]}
            choices = []
            for j in by_recipe[int(multi.recipe_of_node[i])]:
                if j == i or "预热" in names[j]:
                    continue
                if not device_set.intersection(int(op[0]) for op in multi.nodes[j]):
                    continue
                d = dag_distance(i, j)
                if d is not None:
                    choices.append((d, j))
            if choices:
                _, j = min(choices)
                add_gap(i, j, preheat_max_gap_minutes, "recipe_preheat")

    if finish_preparation_max_gap_minutes is not None:
        for rid, ids in by_recipe.items():
            dish = str(multi.recipe_names[rid])
            if "糯米烧麦" in dish:
                patterns = [(lambda s: "包" in s and "烧麦" in s,
                             lambda s: "蒸" in s and "烧麦" in s and "预热" not in s)]
            elif "半熟芝士" in dish:
                patterns = [(lambda s: "注入模具" in s,
                             lambda s: "烘烤" in s or "烤蛋糕" in s)]
            elif "美式薯条" in dish:
                patterns = [(lambda s: "放入薯条" in s,
                             lambda s: "开始烹饪" in s or "烤薯条" in s)]
            elif "轻松一锅蒸" in dish:
                patterns = [(lambda s: "放置" in s and "鱼头" in s,
                             lambda s: s in ("开始烹饪", "蒸制", "正式蒸制", "开始蒸制"))]
            else:
                patterns = []
            for p, q in patterns:
                left = [i for i in ids if p(names[i])]
                right = [i for i in ids if q(names[i])]
                if len(left) == len(right) == 1:
                    add_gap(left[0], right[0], finish_preparation_max_gap_minutes, "freshness")

    def thermal(mid):
        name = str(getattr(multi.machines[mid], "note", "") or "")
        return "蒸箱" if "蒸箱" in name else ("烤箱" if "烤箱" in name else None)

    def heat_time(mid, temp, mode):
        k = thermal(mid)
        if k == "蒸箱":
            if steamer_preheat_minutes is not None:
                return T(steamer_preheat_minutes)
            return T(6 if int(mode) == 13 else 3 if int(mode) == 12 else 4)
        if k == "烤箱":
            if oven_preheat_minutes is not None:
                return T(oven_preheat_minutes)
            v = float(temp or 180)
            return T(4 if v < 180 else 6 if v <= 200 else 8)
        return 0

    def same_state(a, b):
        return tuple(a) == tuple(b)

    # 人工短时参与在动态重排时不能被扩展为整个烹饪节点时长。
    manual_mids = {
        mid for mid, device in enumerate(multi.machines)
        if any(x in str(getattr(device, "note", "") or "")
               for x in ("人工", "手动操作"))
    }

    # 冻结批次：保留真实历史设备占用和热状态。
    fixed_records = []
    fixed_batches = defaultdict(list)
    thermal_history = defaultdict(list)
    for nid in sorted(frozen):
        st = states[nid]
        for mid, bid, rs, re in st.machine_assignments:
            mid = int(mid)
            op = next((op for op in multi.nodes[nid] if int(op[0]) == mid), None)
            temp, heat, mode = tuple(op[2:5]) if op is not None and len(op) >= 5 else (0, 0, 0)
            # 已完成的人工启动操作不能因为本节点仍在蒸/煮而被延长。
            # 只有当前仍在运行的非人工设备工序才可随强制延迟延长。
            end = float(re)
            if (st.status == "running" and mid not in manual_mids
                    and float(re) > float(snapshot.now) + 1e-8):
                end = max(end, float(st.end))
            fixed_records.append((nid, mid, bid, float(rs), end, temp, heat, mode))
            fixed_batches[(mid, bid)].append((T(rs), T(end)))
            if thermal(mid):
                thermal_history[mid].append((T(end), T(rs), (temp, heat, mode)))

    fixed_preheats = []
    if snapshot is not None:
        for p in getattr(snapshot, "preheat_records", ()):
            if float(p["start"]) >= float(snapshot.now) - 1e-8:
                continue                         # 未开始的自动预热重新优化
            p = dict(p)
            fixed_preheats.append(p)
            mid = int(p["machine_id"])
            thermal_history[mid].append((T(p["end"]), T(p["start"]),
                                         (p["temp"], p["heat"], p["mode"])))

    # 物理资源中，普通批次和自动预热都占用同一设备。
    intervals = defaultdict(list)
    for (mid, bid), ranges in fixed_batches.items():
        a = min(x[0] for x in ranges)
        b = max(x[1] for x in ranges)
        if b > a:
            intervals[mid].append(model.NewIntervalVar(a, b - a, b, f"fixed{mid}_{bid}"))
    for idx, p in enumerate(fixed_preheats):
        a, b, mid = T(p["start"]), T(p["end"]), int(p["machine_id"])
        if b > a:
            intervals[mid].append(model.NewIntervalVar(a, b - a, b, f"fixed_heat{idx}"))

    def group_capacity(mid):
        name = str(getattr(multi.machines[mid], "note", "") or "")
        return max(1, int(batch_limit)) if any(x in name for x in ("蒸箱", "烤箱", "冰箱")) else 1

    pending_ops = []
    groups = defaultdict(list)
    for i in range(n):
        if i in frozen:
            continue
        for op in multi.nodes[i]:
            mid, length = int(op[0]), T(op[1])
            if length <= 0:
                continue
            temp, heat, mode = tuple(op[2:5]) if len(op) >= 5 else (0, 0, 0)
            kind = "preheat" if "预热" in names[i] else "normal"
            j = len(pending_ops)
            pending_ops.append((i, mid, length, temp, heat, mode, kind))
            groups[(mid, kind, temp, heat, mode)].append(j)

    batches, cover, thermal_batches = [], defaultdict(list), defaultdict(list)
    for (mid, kind, temp, heat, mode), op_ids in groups.items():
        for count in range(1, min(group_capacity(mid), len(op_ids)) + 1):
            for subset in itertools.combinations(op_ids, count):
                members = [pending_ops[j][0] for j in subset]
                if len(set(members)) != count:
                    continue
                span = max(pending_ops[j][2] for j in subset)
                label = f"b{len(batches)}"
                on = model.NewBoolVar(f"{label}_on")
                bs = model.NewIntVar(now, horizon, f"{label}_s")
                be = model.NewIntVar(now, horizon, f"{label}_e")
                model.Add(be == bs + span).OnlyEnforceIf(on)
                intervals[mid].append(model.NewOptionalIntervalVar(bs, span, be, on, label))
                for j in subset:
                    model.Add(S[pending_ops[j][0]] == bs).OnlyEnforceIf(on)
                    cover[j].append(on)
                b = dict(mid=mid, subset=subset, on=on, s=bs, e=be,
                         kind=kind, state=(temp, heat, mode), span=span,
                         preheat=None, ph_start=None, ph_duration=0)
                batches.append(b)
                if thermal(mid):
                    thermal_batches[mid].append(b)
    for j in range(len(pending_ops)):
        if not cover[j]:
            raise ValueError(f"操作{j}无候选共享批次")
        model.AddExactlyOne(cover[j])

    # 设备温度状态用所选择批次的“紧邻前驱”表示；串行弧由 AddCircuit 选择。
    # 对于单台蒸箱/烤箱，此约束避免多次独立预热和无预热使用。
    hot_window = T(hot_keep_minutes)
    for mid, candidates in thermal_batches.items():
        if int(multi.capacities[mid]) != 1:
            raise ValueError(
                f"{multi.machines[mid].note}物理容量={multi.capacities[mid]}。"
                "设备级热状态目前支持单实例蒸箱/烤箱；请拆分为独立设备实例。"
            )
        prev = max(thermal_history[mid], key=lambda x: (x[1], x[0])) if thermal_history[mid] else None
        prior_end = prev[0] if prev else now
        prior_state = prev[2] if prev else None
        # 故障释放时间也属于新一轮预热/烹饪的起点约束。
        release = now
        if snapshot is not None:
            release = max((T(t) for (machine_id, _iid), t in snapshot.machine_avail.items()
                           if int(machine_id) == mid), default=now)
        first_available = max(now, prior_end, release)

        # 对每个非原生预热批次建立可选的物理自动预热区间。
        for b in candidates:
            if b["kind"] == "preheat":
                continue
            label = f"auto_heat_{mid}_{len(batches)}_{len(str(b['subset']))}"
            req = model.NewBoolVar(f"{label}_needed")
            hs = model.NewIntVar(now, horizon, f"{label}_start")
            hd = heat_time(mid, b["state"][0], b["state"][2])
            model.Add(req <= b["on"])
            model.Add(hs + hd == b["s"]).OnlyEnforceIf(req)
            intervals[mid].append(model.NewOptionalIntervalVar(
                hs, hd, b["s"], req, label
            ))
            b.update(preheat=req, ph_start=hs, ph_duration=hd)

        arcs = []
        for idx, b in enumerate(candidates, 1):
            arcs.append((idx, idx, b["on"].Not()))
            first = model.NewBoolVar(f"th{mid}_first{idx}")
            last = model.NewBoolVar(f"th{mid}_last{idx}")
            arcs.extend([(0, idx, first), (idx, 0, last)])
            model.AddImplication(first, b["on"])
            model.AddImplication(last, b["on"])
            if b["preheat"] is not None:
                req = b["preheat"]
                if prior_state is None or not same_state(prior_state, b["state"]):
                    model.Add(req == 1).OnlyEnforceIf(first)
                else:
                    model.Add(b["s"] <= prior_end + hot_window).OnlyEnforceIf(
                        [first, req.Not()]
                    )
                    model.Add(b["s"] >= prior_end + hot_window + 1).OnlyEnforceIf(
                        [first, req]
                    )
                # 历史事件（可能正在运行）必须先结束才能开始新预热/烹饪。
                model.Add(b["ph_start"] >= first_available).OnlyEnforceIf([first, req])
                model.Add(b["s"] >= first_available).OnlyEnforceIf([first, req.Not()])
            else:
                model.Add(b["s"] >= first_available).OnlyEnforceIf(first)

        for i, a in enumerate(candidates, 1):
            for j, b in enumerate(candidates, 1):
                if i == j:
                    continue
                follows = model.NewBoolVar(f"thermal{mid}_{i}_to_{j}")
                arcs.append((i, j, follows))
                model.AddImplication(follows, a["on"])
                model.AddImplication(follows, b["on"])
                if b["preheat"] is None:
                    model.Add(b["s"] >= a["e"]).OnlyEnforceIf(follows)
                    continue
                req = b["preheat"]
                if not same_state(a["state"], b["state"]):
                    model.Add(req == 1).OnlyEnforceIf(follows)
                else:
                    model.Add(b["s"] <= a["e"] + hot_window).OnlyEnforceIf(
                        [follows, req.Not()]
                    )
                    model.Add(b["s"] >= a["e"] + hot_window + 1).OnlyEnforceIf(
                        [follows, req]
                    )
                model.Add(b["ph_start"] >= a["e"]).OnlyEnforceIf([follows, req])
                model.Add(b["s"] >= a["e"]).OnlyEnforceIf([follows, req.Not()])
        model.AddCircuit(arcs)

    for mid, items in intervals.items():
        model.AddCumulative(items, [1] * len(items), max(1, int(multi.capacities[mid])))

    if snapshot is not None:
        machine_floor = defaultdict(lambda: now)
        for (mid, _iid), when in snapshot.machine_avail.items():
            machine_floor[int(mid)] = max(machine_floor[int(mid)], T(when))
        for p in fixed_preheats:
            mid = int(p["machine_id"])
            if T(p["end"]) > now:
                machine_floor[mid] = max(machine_floor[mid], T(p["end"]))
        for b in batches:
            if thermal(b["mid"]):
                # 设备状态已经在 AddCircuit 的前驱、热态逻辑中约束。
                continue
            model.Add(b["s"] >= machine_floor[b["mid"]]).OnlyEnforceIf(b["on"])

    humans = []
    for i in range(n):
        if not bool(multi.needs_human[i]):
            continue
        if i in frozen:
            st = states[i]
            if not st.machine_assignments:
                continue
            s = T(st.start)
            e = T(st.actual_end if st.status == "done" and st.actual_end is not None else st.end)
            if e > s:
                humans.append(model.NewIntervalVar(s, e-s, e, f"h_fixed{i}"))
        elif dur[i] > 0:
            humans.append(model.NewIntervalVar(S[i], dur[i], E[i], f"h{i}"))
    if humans:
        model.AddNoOverlap(humans)

    if terminal_max_idle is not None:
        for i in range(n):
            if i in frozen or not multi.needs_human[i] or not multi.predecessors[i] or multi.successors[i]:
                continue
            pmax = model.NewIntVar(0, horizon, f"latest_pred{i}")
            model.AddMaxEquality(pmax, [E[p] for p in multi.predecessors[i]])
            model.Add(S[i] <= pmax + T(terminal_max_idle))

    finishes = {}
    for rid, ids in by_recipe.items():
        f = model.NewIntVar(0, horizon, f"finish{rid}")
        model.AddMaxEquality(f, [E[i] for i in ids])
        finishes[rid] = f
    makespan = model.NewIntVar(0, horizon, "makespan")
    latest = model.NewIntVar(0, horizon, "latest")
    earliest = model.NewIntVar(0, horizon, "earliest")
    spread = model.NewIntVar(0, horizon, "spread")
    model.AddMaxEquality(makespan, list(finishes.values()))
    model.AddMaxEquality(latest, list(finishes.values()))
    model.AddMinEquality(earliest, list(finishes.values()))
    model.Add(spread == latest - earliest)

    def solve(seconds):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(0.1, float(seconds))
        solver.parameters.num_search_workers = 8
        solver.parameters.random_seed = 42
        return solver

    model.Minimize(makespan)
    s1 = solve(stage1_seconds)
    status1 = s1.Solve(model)
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"第一阶段无可行解：{s1.StatusName(status1)}；检查预热/冻结/工艺时间窗")
    best = s1.Value(makespan)
    model.Add(makespan <= best + max(0, T(max_extra_minutes)))
    model.Minimize(spread)
    s2 = solve(stage2_seconds)
    status2 = s2.Solve(model)
    selected = s2 if status2 in (cp_model.OPTIMAL, cp_model.FEASIBLE) else s1

    starts = [M(selected.Value(s)) for s in S]
    ends = [M(selected.Value(e)) for e in E]
    records = list(fixed_records)
    heat_records = list(fixed_preheats)
    bid = max((int(r[2]) for r in fixed_records if isinstance(r[2], int)), default=0) + 1
    for b in batches:
        if not selected.BooleanValue(b["on"]):
            continue
        mid = b["mid"]
        for j in b["subset"]:
            nid, _mid, length, temp, heat, mode, _kind = pending_ops[j]
            records.append((nid, mid, bid, starts[nid], starts[nid] + M(length), temp, heat, mode))
        if b["preheat"] is not None and selected.BooleanValue(b["preheat"]):
            heat_records.append({
                "machine_id": mid, "batch_id": f"heat_{bid}",
                "target_batch_id": bid, "start": M(selected.Value(b["ph_start"])),
                "end": M(selected.Value(b["s"])),
                "temp": b["state"][0], "heat": b["state"][1], "mode": b["state"][2],
                "label": "设备自动预热", "recipe_ids": sorted({
                    int(multi.recipe_of_node[pending_ops[j][0]]) for j in b["subset"]
                }),
            })
        bid += 1

    result = MultiScheduleResult(starts, ends, records, max(ends),
                                 (time.perf_counter()-t0)*1000.0)
    result.preheat_records = heat_records
    result.optimize_info = {
        "stage1_status": s1.StatusName(status1),
        "stage2_status": s2.StatusName(status2),
        "stage1_optimal_proven": status1 == cp_model.OPTIMAL,
        "stage2_optimal_proven": status2 == cp_model.OPTIMAL,
        "stage1_makespan": M(best),
        "makespan_cap": M(best + max(0, T(max_extra_minutes))),
        "finish_spread": M(selected.Value(spread)),
        "finish_times": {rid: M(selected.Value(f)) for rid, f in finishes.items()},
        "bounded_gaps": [
            {"predecessor": multi.step_names[i], "successor": multi.step_names[j],
             "max_gap_minutes": limit, "type": kind}
            for i, j, limit, kind in bounded
        ],
        "auto_preheat_count": len(heat_records),
    }
    report = check_multi_schedule(multi, result)
    if report["has_any"]:
        raise AssertionError(f"排程设备/依赖冲突：{report}")
    if verbose:
        print(f"[阶段1] makespan={M(best):.1f}分钟 状态={s1.StatusName(status1)}")
        print(f"[阶段2] 完成时间差={result.optimize_info['finish_spread']:.1f}分钟 "
              f"状态={s2.StatusName(status2)}")
        print(f"[设备级预热] 自动预热{len(heat_records)}次，"
              f"热态复用窗口{hot_keep_minutes:g}分钟")
        for rid, val in sorted(result.optimize_info["finish_times"].items()):
            print(f"  {multi.recipe_names[rid]}: {val:.1f}分钟")
    return result



def run_multi(targets, parse_schedule_text, recipe_to_schedule_text,
              verbose=True, save_dir=None, save_png=None):
    """
    targets: [{"recipe_id","name","ingredients","steps"}, ...]
    parse_schedule_text: 单菜解析函数
    recipe_to_schedule_text: LLM 调用函数
    """
    from greedy_scheduler import greedy_schedule

    inst_list = []
    recipe_infos = []
    single_makespans = []
    llm_total_ms = 0.0
    parse_total_ms = 0.0

    # 提前导入
    from llm_parser import (
        recipe_to_schedule_json, json_to_schedule_text
    )

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    step_names_list = [None] * len(targets)
    ingredients_list = [None] * len(targets)
    inst_list = [None] * len(targets)
    recipe_infos = [None] * len(targets)
    single_makespans = [None] * len(targets)

    # ---------- 定义单个菜谱的解析函数 ----------
    def _parse_one(idx, t):
        """在子线程里独立解析一个菜谱"""
        t0 = time.perf_counter()

        # 1. LLM 调用（耗时主体）
        data_json = recipe_to_schedule_json(
            t["recipe_id"], t["name"],
            t["ingredients"], t["steps"],
            verbose=False,  # ★ 并发时不打印
        )
        llm_ms = (time.perf_counter() - t0) * 1000

        # 2. 提取步骤名
        step_names = []
        for j, s in enumerate(data_json["steps"]):
            nm = s.get("name")
            if not nm or not str(nm).strip():
                nm = f"S{j}"
            step_names.append(str(nm).strip())

        # 3. 提取食材
        ings = data_json.get("ingredients", [])

        # 4. JSON 转调度文本
        text = json_to_schedule_text(data_json)

        # 5. 解析成 Instance
        t1 = time.perf_counter()
        inst = parse_schedule_text(text)
        parse_ms = (time.perf_counter() - t1) * 1000

        # 6. 单菜调度（用于串行基线）
        r_single = greedy_schedule(inst)

        return {
            "idx": idx,
            "name": t["name"],
            "llm_ms": llm_ms,
            "parse_ms": parse_ms,
            "step_names": step_names,
            "ingredients": ings,
            "inst": inst,
            "single_makespan": r_single.makespan,
            "recipe_info": {
                "recipe_id": t["recipe_id"],
                "name": t["name"],
            },
        }

    # ---------- 并发解析 ----------
    max_workers = min(5, len(targets))  # 最多 5 路并发
    print(f"\n[并发] 启动 {max_workers} 路并发解析 "
          f"({len(targets)} 道菜谱)")

    t_parallel_0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_parse_one, i, t): i
            for i, t in enumerate(targets)
        }

        done_count = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  ✗ [{targets[idx]['name']}] 解析失败: {e}")
                raise

            done_count += 1
            print(f"  ✓ [{done_count}/{len(targets)}] "
                  f"{res['name']}  "
                  f"LLM={res['llm_ms']:.0f}ms  "
                  f"解析={res['parse_ms']:.1f}ms")

            # 按索引回填，保证顺序
            step_names_list[idx] = res["step_names"]
            ingredients_list[idx] = res["ingredients"]
            inst_list[idx] = res["inst"]
            recipe_infos[idx] = res["recipe_info"]
            single_makespans[idx] = res["single_makespan"]
            llm_total_ms += res["llm_ms"]
            parse_total_ms += res["parse_ms"]

    t_parallel = time.perf_counter() - t_parallel_0
    print(f"[并发] 全部完成，耗时 {t_parallel:.2f}s ")

    # ---- 合并 + 多菜调度 ----
    print(f"\n[合并] 合并 {len(inst_list)} 道菜到一个多菜实例...")
    multi = merge_instances(
        inst_list, recipe_infos, step_names_list=step_names_list
    )
    print(f"  ✓ N={multi.N}  E={multi.E}  K={multi.K}")

    print(f"\n[调度] 多菜并行堆栈贪心...")

    print("\n[双目标优化] 第一阶段最短总时间，第二阶段同步出锅")

    res = optimize_lexicographic_schedule(
        multi,
        snapshot=None,
        max_extra_minutes=0.0,
        stage1_seconds=12.0,
        stage2_seconds=12.0,
        tick_per_minute=10,
        batch_limit=3,
        terminal_max_idle=5.0,
        verbose=True,
    )

    res = optimize_human_work_streak(
        multi,
        res,
        snapshot=None,
        min_rest_minutes=5.0,
        max_continuous_minutes=None,
        solver_seconds=2.0,
        verbose=True,
    )

    final_delay = res.optimize_info

    final_delay = res.optimize_info

    print(
        f"  ✓ 优化后 makespan={res.makespan:.1f}分钟"
    )
    print(
        f"  ✓ 最大完成时间差="
        f"{final_delay['finish_spread']:.1f}分钟"
    )

    print(f"  ✓ makespan = {res.makespan:.1f} 分钟，"
          f"耗时 {res.algo_time_ms:.3f} ms")

    # ★ 菜品同步对齐
    print(f"\n[对齐] 各菜品完成时间同步...")
    print(f"  ✓ 对齐后 makespan = {res.makespan:.1f} 分钟")

    # ---- 校验 ----
    print(f"\n[校验] 设备冲突 + 依赖关系...")
    check = check_multi_schedule(multi, res)
    print(f"  ✓ 冲突 {len(check['conflicts'])}，"
          f"依赖违规 {len(check['dependency_violations'])}，")

    # ---- 下界 ----
    print(f"\n[下界] 计算理论最优解...")
    lb = compute_lower_bound(multi)
    print(f"  ✓ LB_CP={lb['lb_cp']:.1f}, "
          f"LB_Machine={lb['lb_machine']:.1f}, "
          f"LB={lb['lb']:.1f}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    names = "_".join([n[:4] for n in multi.recipe_names])
    png_file = os.path.join(
        save_dir, f"multi_gantt_{names}_{ts}.png"
    )
    # ---- 甘特图（只保存 PNG，不打印 ASCII）----
    png_file = ""
    if save_png:
        png_file = os.path.join(
            save_dir, f"multi_gantt_{len(multi.recipe_names)}dishes.png"
        )
        try:
            plot_gantt_matplotlib(
                multi, res,
                save_path=png_file,
                show=False,
                figsize=(16, 9),
                dpi=200,
                title=f"多菜并行甘特图  {len(multi.recipe_names)} 道菜  "
                      f"makespan={res.makespan:.1f} 分钟",
            )
        except Exception as e:
            import traceback
            print(f"  ✗ matplotlib 甘特图失败: {e}")
            traceback.print_exc()
            png_file = ""

    # ---- 对比分析（不打印，仅计算 serial 用于 overview）----
    # ---- 对比分析 ----
    serial = sum(single_makespans)

    # 初始调度由 run.py 以 verbose=False 调用，
    # 因此不在这里输出完整分析。
    # 最终报告统一由 run.py 在调度/动态重排结束后输出。
    if verbose:
        print_comparison(
            multi,
            res,
            lb,
            check,
            serial_makespan=serial,
        )

    # ============================================================
    # ★ 关键节点提醒生成
    # ============================================================
    print()
    print("=" * 72)
    print("关键节点提醒（执行时按需推送）")
    print("=" * 72)

    from reminder_generator import generate_reminders

    all_reminders = []
    # 从 0 到 makespan，每 1 分钟扫一次（可根据需要调整步长）
    t_now = 0.0
    step = 1.0
    while t_now <= res.makespan + 0.01:
        reminders = generate_reminders(
            multi, res, now=t_now, lookahead=3.0
        )
        for r in reminders:
            # 去重：同一 (node_id, level) 只保留第一次
            key = (r["node_id"], r["level"])
            if key not in {(x["node_id"], x["level"])
                           for x in all_reminders}:
                all_reminders.append(r)
        t_now += step

    # 按触发时刻排序
    all_reminders.sort(key=lambda x: x["time"])

    if all_reminders:
        # 按级别分类统计
        from collections import Counter
        level_count = Counter(r["level"] for r in all_reminders)
        print(f"\n  共 {len(all_reminders)} 条提醒")
        print(f"    🔴 action  : {level_count.get('action', 0)} 条")
        print(f"    🟡 warning : {level_count.get('warning', 0)} 条")
        print(f"    🔵 info    : {level_count.get('info', 0)} 条")

        # 打印明细（前 30 条）
        print(f"\n  {'触发时刻':<10}{'级别':<10}{'消息'}")
        print("  " + "-" * 68)
        for r in all_reminders[:30]:
            icon = {"action": "🔴", "warning": "🟡",
                    "info": "🔵"}.get(r["level"], "⚪")
            t_trig = r["time"] - r.get("delay_min", 0)
            print(f"  {t_trig:>6.1f}min  {icon} {r['level']:<8}{r['message']}")
        if len(all_reminders) > 30:
            print(f"  ... 还有 {len(all_reminders) - 30} 条")

    print("=" * 72)

    if save_png and save_dir:
        # ---------- 内部格式 ----------
        schedule_json = os.path.join(save_dir, "multi_schedule.json")
        export_schedule_json(
            multi, res,
            save_path=schedule_json,
            lb=lb,
            serial_makespan=sum(single_makespans),
            mode="multi",
        )

    return {
        "multi": multi,
        "res": res,
        "check": check,
        "lb": lb,
        "serial_makespan": serial,
        "single_makespans": single_makespans,
        "llm_total_ms": llm_total_ms,
        "parse_total_ms": parse_total_ms,
        "inst_list": inst_list,
        "recipe_infos": recipe_infos,
        "step_names_list": step_names_list,
        "reminders": all_reminders,
        "ingredients_list": ingredients_list,
    }

# ============================================================
# 9. 多菜 matplotlib 甘特图（保存 PNG）
# ============================================================
def plot_multi_gantt_matplotlib(
    multi,
    res,
    save_path=None,
    show=False,
    figsize=(16, 9),
    dpi=200,
    title=None,
):
    """
    多菜并行甘特图（按菜谱分色）。
    每道菜一个颜色，块上标签为"菜名首字+序号"，图例只显示菜名。
    """
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.cm as cm
    from collections import defaultdict

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    op_records = list(res.op_records)
    makespan = float(res.makespan)
    if not op_records or makespan <= 0:
        print("  (无工序可画)")
        return None

    # ---- Y 轴：设备实例 ----
    used_mid = set()
    for rec in op_records:
        used_mid.add(rec[1])

    y_labels, y_map = [], {}
    y_idx = 0
    for mid in range(multi.K):
        if mid not in used_mid:
            continue
        mname = multi.machines[mid].note or f"m{mid}"
        y_map[mid] = y_idx
        y_labels.append(mname)
        y_idx += 1

    # ---- 每个菜谱一个颜色 ----
    n_recipes = len(multi.recipe_names)
    cmap = cm.get_cmap("tab10", max(1, n_recipes))
    recipe_colors = {i: cmap(i % 10) for i in range(n_recipes)}

    # ---- 菜谱内步骤序号 ----
    step_seq = {}
    counter = defaultdict(int)
    for nid in range(multi.N):
        rid = multi.recipe_of_node[nid]
        step_seq[nid] = counter[rid]
        counter[rid] += 1

    # ---- 画布 ----
    fig, ax = plt.subplots(figsize=figsize)
    x_max = makespan * 1.02

    # ---- 画块 ----
    for rec in op_records:
        nid, mid, bid, s, e = rec[0], rec[1], rec[2], rec[3], rec[4]
        if (mid, bid) not in y_map:
            continue
        y = y_map[(mid, bid)]
        dur = max(float(e) - float(s), 0.01)
        rid = multi.recipe_of_node[nid]
        color = recipe_colors[rid]

        ax.broken_barh(
            [(float(s), dur)],
            (y - 0.4, 0.8),
            facecolors=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=2,
        )

        if dur >= makespan * 0.03:
            seq = step_seq[nid]
            short = multi.recipe_names[rid][:3]
            ax.text(
                float(s) + dur / 2, y,
                f"{short}_{seq}",
                ha="center", va="center",
                fontsize=7, clip_on=True, zorder=3,
            )

    # ---- 坐标轴 ----
    ax.set_yticks(range(y_idx))
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.6, y_idx - 0.4)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("时间（分钟）")
    ax.set_ylabel("设备")
    ax.grid(True, linestyle="--", alpha=0.4, zorder=1)

    if title is None:
        title = f"多菜并行甘特图  makespan = {makespan:.1f} 分钟"
    ax.set_title(title)

    # ---- 图例：只显示菜名 ----
    handles = [
        Patch(facecolor=recipe_colors[i], edgecolor="black",
              label=multi.recipe_names[i])
        for i in range(n_recipes)
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=9,
        ncol=1,
        framealpha=0.9,
    )

    # ---- 保存 ----
    if save_path:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        fig.savefig(save_path, format="png", dpi=dpi, bbox_inches="tight")
        print(f"  ✓ 多菜甘特图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
