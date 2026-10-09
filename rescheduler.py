# rescheduler.py
# -*- coding: utf-8 -*-
"""
动态重排与通知模块

功能：
    1. Snapshot          执行状态快照
    2. classify_nodes    分类：已完成 / 进行中 / 待执行
    3. reschedule        最小扰动增量重排
    4. compare_conflicts 对比重排前后冲突
    5. make_notifications 生成变更通知
"""

import heapq
from dataclasses import dataclass, field
from collections import defaultdict, deque


# ============================================================
# 1. 状态定义
# ============================================================
STATUS_DONE = "done"            # 已完成，时间冻结
STATUS_RUNNING = "running"      # 进行中，不可中断
STATUS_PENDING = "pending"      # 待执行，可重排


@dataclass
class NodeState:
    nid: int
    status: str                 # done / running / pending
    start: float = 0.0          # 原计划开始
    end: float = 0.0            # 原计划结束
    actual_end: float = None    # 实际结束（done 时）
    progress: float = 0.0       # running 时已完成比例 [0,1]
    machine_assignments: list = field(default_factory=list)
    # [(mid, iid, s, e), ...] 原来分配的工序


@dataclass
class Snapshot:
    """执行状态快照"""
    now: float                                  # 当前时刻
    node_states: dict                           # {nid: NodeState}
    machine_avail: dict                         # {(mid,bid): next_avail_time}
    makespan: float


# ============================================================
# 2. 快照：从调度结果 + 当前时间 生成执行状态
# ============================================================
def take_snapshot(multi, res, now,
                  running_progress=None,
                  forced_delays=None,
                  machine_outages=None,
                  cancelled_nodes=None):
    """
    生成执行状态快照。

    参数:
        multi   : MultiInstance
        res     : MultiScheduleResult
        now     : 当前时刻（分钟）

        running_progress : {nid: 进度[0,1]}
            进行中步骤的完成比例，默认按时间线性推算

        forced_delays : {nid: 额外延迟分钟}
            让某些进行中的步骤比原计划晚多久结束
            例：{2: 10.0} 表示 nid=2 额外延迟 10 分钟

        machine_outages : {(machine_id, instance_id): (start, end)}
            设备实例在某时间段不可用
            例：{(3, 0): (20.0, 35.0)} 表示烤箱 20~35 分钟故障
            例：{(3, None): (20.0, 35.0)} 表示整台烤箱所有实例故障

        cancelled_nodes : set/list of nid
            取消的步骤（视为已完成或跳过）
    """
    running_progress = running_progress or {}
    forced_delays = forced_delays or {}
    machine_outages = machine_outages or {}
    cancelled_nodes = set(cancelled_nodes or [])

    node_states = {}

    # 按节点分组工序
    node_ops = defaultdict(list)
    for rec in res.op_records:
        nid, mid, bid, s, e = rec[0], rec[1], rec[2], rec[3], rec[4]
        node_ops[nid].append((mid, bid, s, e))
    for nid in node_ops:
        node_ops[nid].sort(key=lambda x: x[2])

    for nid in range(multi.N):
        # 被取消的节点直接标记为 DONE
        if nid in cancelled_nodes:
            node_states[nid] = NodeState(
                nid, STATUS_DONE, start=0, end=now,
                actual_end=now, progress=1.0,
                machine_assignments=[],
            )
            continue
        node_s = res.node_start[nid]
        node_e = res.node_end[nid]
        ops = node_ops.get(nid, [])

        if node_e <= now:
            # 已完成
            st = NodeState(nid, STATUS_DONE,
                           start=node_s, end=node_e,
                           actual_end=node_e,
                           progress=1.0,
                           machine_assignments=ops)
        elif node_s <= now < node_e:
            # 进行中：可选地推后结束时间
            prog = running_progress.get(
                nid,
                (now - node_s) / max(1e-9, node_e - node_s)
            )
            delay = forced_delays.get(nid, 0.0)
            new_end = node_e + delay
            st = NodeState(nid, STATUS_RUNNING,
                           start=node_s, end=new_end,
                           progress=min(1.0, max(0.0, prog)),
                           machine_assignments=ops)
        else:
            # 待执行
            st = NodeState(nid, STATUS_PENDING,
                           start=node_s, end=node_e,
                           progress=0.0,
                           machine_assignments=ops)
        node_states[nid] = st

    machine_avail = {}
    for k in range(multi.K):
        latest = now
        for nid, st in node_states.items():
            if st.status in (STATUS_DONE, STATUS_RUNNING):
                for mid, bid_, s, e in st.machine_assignments:
                    if mid == k:
                        latest = max(latest, e)
        # 该设备所有实例共享同一个 avail（保守做法）
        cap = max(1, multi.capacities[k])
        for iid in range(cap):
            machine_avail[(k, iid)] = latest

    # ---- 应用设备故障：把故障时段推入可用时间 ----
    for key, (ostart, oend) in machine_outages.items():
        if isinstance(key, tuple):
            mid, bid = key if len(key) == 2 else (key[0], None)
        else:
            mid, bid = key, None

        if bid is not None:
            # 指定实例故障
            avail = machine_avail.get((mid, bid), now)
            if avail < oend and ostart <= oend:
                machine_avail[(mid, bid)] = max(avail, oend)
        else:
            # 整台设备故障
            for k in range(multi.K):
                if k != mid:
                    continue
                for ii in range(max(1, multi.capacities[k])):
                    avail = machine_avail.get((k, ii), now)
                    if avail < oend and ostart <= oend:
                        machine_avail[(k, ii)] = max(avail, oend)

    snapshot = Snapshot(
        now=now,
        node_states=node_states,
        machine_avail=machine_avail,
        makespan=res.makespan,
    )
    # 只冻结已经开始的自动预热；未来自动预热由新优化器重新调度。
    snapshot.preheat_records = [
        dict(p) for p in getattr(res, "preheat_records", ())
        if float(p["start"]) < float(now) - 1e-8
    ]
    for p in snapshot.preheat_records:
        if float(p["end"]) > float(now):
            mid = int(p["machine_id"])
            for iid in range(max(1, int(multi.capacities[mid]))):
                key = (mid, iid)
                machine_avail[key] = max(machine_avail.get(key, now), float(p["end"]))
    return snapshot


# ============================================================
# 3. 增量重排（最小扰动）
# ============================================================
@dataclass
class RescheduleResult:
    node_start: list
    node_end: list
    op_records: list
    makespan: float
    changed_nodes: list     # [(nid, old_start, old_end, new_start, new_end)]
    frozen_nodes: list      # 被冻结的 nid（done + running）
    algo_time_ms: float = 0.0


def _get_node_machine_params(multi, nid, mid):
    """
    从 multi.nodes[nid] 里读该节点对设备 mid 的 temp/heat/mode。
    返回 (temp, heat, mode)，找不到则 (0, 0, 0)
    """
    for item in multi.nodes[nid]:
        if item[0] != mid:
            continue
        if len(item) >= 5:
            return item[2], item[3], item[4]
        return 0, 0, 0
    return 0, 0, 0



def reschedule_minimal_perturbation(
    multi, snapshot, verbose=True, op_index=None,
    respect_original_start=False, max_finish_gap=2.0,
    align_finishes=True, time_step=1.0,
    max_extra_minutes=0.0,
    stage1_seconds=12.0, stage2_seconds=12.0,
    human_rest_minutes=5.0, human_stage_seconds=2.0,
):
    """动态多阶段重排，保留设备自动预热及人工优化安全约束。"""
    import time
    from multi_scheduler import optimize_lexicographic_schedule
    from multi_scheduler import optimize_human_work_streak_safe

    tic = time.perf_counter()
    base = optimize_lexicographic_schedule(
        multi, snapshot=snapshot,
        max_extra_minutes=max_extra_minutes if align_finishes else 0.0,
        stage1_seconds=stage1_seconds,
        stage2_seconds=stage2_seconds,
        verbose=verbose,
    )
    refined = optimize_human_work_streak_safe(
        multi, base, snapshot=snapshot,
        min_rest_minutes=human_rest_minutes,
        solver_seconds=human_stage_seconds,
        verbose=verbose,
    )
    frozen = sorted(i for i, st in snapshot.node_states.items()
                    if st.status in (STATUS_DONE, STATUS_RUNNING))
    changed = []
    for nid, st in snapshot.node_states.items():
        if st.status != STATUS_PENDING:
            continue
        s, e = refined.node_start[nid], refined.node_end[nid]
        if abs(s - st.start) > 1e-6 or abs(e - st.end) > 1e-6:
            changed.append((nid, st.start, st.end, s, e))
    result = RescheduleResult(
        node_start=list(refined.node_start),
        node_end=list(refined.node_end),
        op_records=list(refined.op_records),
        makespan=refined.makespan,
        changed_nodes=changed, frozen_nodes=frozen,
        algo_time_ms=(time.perf_counter() - tic) * 1000,
    )
    result.preheat_records = list(getattr(refined, "preheat_records", []))
    result.optimize_info = dict(getattr(refined, "optimize_info", {}) or {})
    return result


def _is_passive_wait_node(multi, nid):
    """
    判断是否属于被动等待类节点。

    被动等待：
    - 时间必须继续流逝；
    - 但不持续占用人工或主要烹饪设备。
    """

    if not hasattr(multi, "step_names"):
        return False

    name = str(
        multi.step_names[nid]
    ).split("#")[-1]

    keywords = (
        "等待",
        "静置",
        "浸泡",
        "冷却",
        "腌制",
        "醒发",
        "发酵",
    )

    return any(
        k in name
        for k in keywords
    )
# ============================================================
# 4. 冲突对比
# ============================================================
def count_conflicts(res_like):
    """
    res_like: MultiScheduleResult 或 RescheduleResult
    返回 (device_conflicts, dep_violations)
    """
    from collections import defaultdict

    # 4.1 设备冲突
    by_mb = defaultdict(list)
    for rec in res_like.op_records:
        nid, mid, bid, s, e = rec[0], rec[1], rec[2], rec[3], rec[4]
        by_mb[(mid, bid)].append((nid, s, e))

    device_conflicts = []
    for (mid, bid), ops in by_mb.items():
        ops.sort(key=lambda x: x[1])
        for k in range(len(ops) - 1):
            n1, s1, e1 = ops[k]
            n2, s2, e2 = ops[k + 1]
            if e1 > s2 + 1e-6:
                device_conflicts.append({
                    "machine_id": mid, "instance_id": bid,
                    "node_a": n1, "node_b": n2,
                    "overlap": e1 - s2,
                })

    return {
        "device_conflicts": device_conflicts,
        "n_device_conflicts": len(device_conflicts),
    }


def compare_conflicts(orig_res, new_res, multi):
    """按真实物理批次对比重排前后的冲突。"""
    from multi_scheduler import check_multi_schedule

    orig = check_multi_schedule(multi, orig_res)
    new = check_multi_schedule(multi, new_res)

    old_dev = len(orig["conflicts"])
    new_dev = len(new["conflicts"])

    old_dep = len(orig["dependency_violations"])
    new_dep = len(new["dependency_violations"])

    return {
        "orig_device_conflicts": old_dev,
        "new_device_conflicts": new_dev,
        "orig_dep_violations": old_dep,
        "new_dep_violations": new_dep,
        "device_ok": new_dev == 0,
        "dep_ok": new_dep == 0,
        "orig_dep_detail": orig["dependency_violations"],
        "new_dep_detail": new["dependency_violations"],
    }

# ============================================================
# 5. 变更通知
# ============================================================
def make_notifications(orig_res, new_res, multi, snapshot,
                       label_fn=None):
    """
    生成变更通知列表。
    label_fn: 可选，(nid) -> 步骤标签，默认 S{nid}
    """
    if label_fn is None:
        label_fn = lambda nid: f"S{nid}"

    notifications = []
    states = snapshot.node_states

    # ★ 一次性建索引
    nid_machines = defaultdict(set)
    for rec in new_res.op_records:
        nid_machines[rec[0]].add(rec[1])

    for nid in range(multi.N):
        st = states[nid]
        old_s = st.start
        old_e = st.end
        new_s = new_res.node_start[nid]
        new_e = new_res.node_end[nid]

        if st.status == STATUS_DONE:
            continue

        if abs(new_s - old_s) < 1e-6 and abs(new_e - old_e) < 1e-6:
            continue

        delta_s = new_s - old_s
        delta_e = new_e - old_e

        if st.status == STATUS_RUNNING:
            change_type = "进行中-延后" if delta_e > 1e-6 else "进行中-调整"
        elif st.status == STATUS_PENDING:
            if delta_s > 1e-6 and abs(delta_s - delta_e) < 1e-6:
                change_type = "整体延后"
            elif delta_s < -1e-6:
                change_type = "整体提前"
            else:
                change_type = "时间调整"
        else:
            change_type = "未知"

        # 涉及的设备
        # 涉及的设备
        machines = list(set(
            rec[1] for rec in new_res.op_records
            if rec[0] == nid
        ))

        notifications.append({
            "node_id": nid,
            "label": label_fn(nid),
            "status": st.status,
            "change_type": change_type,
            "old_start": old_s,
            "old_end": old_e,
            "new_start": new_s,
            "new_end": new_e,
            "delta_start": delta_s,
            "delta_end": delta_e,
            "machines": machines,
        })

    return notifications


# ============================================================
# 6. 打印
# ============================================================
def print_notifications(notifications, width=78):
    print()
    print("=" * width)
    print("重排变更通知")
    print("=" * width)
    if not notifications:
        print("  （无变更，重排结果与计划一致）")
        print("=" * width)

    print(f"{'步骤':<10}{'状态':<10}{'类型':<12}"
          f"{'原开始':<10}{'新开始':<10}{'Δ':<10}")
    print("-" * width)
    for n in notifications:
        print(f"{n['label']:<10}{n['status']:<10}{n['change_type']:<12}"
              f"{n['old_start']:<10.1f}{n['new_start']:<10.1f}"
              f"{n['delta_start']:+.1f}")
    print("-" * width)
    print(f"共 {len(notifications)} 项变更")
    print("=" * width)


def print_conflict_report(cmp_result, width=78):
    print()
    print("=" * width)
    print("重排质量校验")
    print("=" * width)

    print(f"\n【设备冲突】")
    print(f"  重排前: {cmp_result['orig_device_conflicts']}")
    print(f"  重排后: {cmp_result['new_device_conflicts']}")
    ok = "✓" if cmp_result["device_ok"] else "✗ 新冲突增加！"
    print(f"  结论  : {ok}")

    print(f"\n【依赖违规】")
    print(f"  重排前: {cmp_result['orig_dep_violations']}")
    print(f"  重排后: {cmp_result['new_dep_violations']}")
    ok = "✓" if cmp_result["dep_ok"] else "✗ 存在违规！"
    print(f"  结论  : {ok}")

    if not cmp_result["dep_ok"]:
        print("\n  违规明细（前 5 条）：")
        for v in cmp_result["new_dep_detail"][:5]:
            print(f"    N{v['node']} 早于前置 N{v['pred']} {v['delay']:.1f} 分钟")

    print("=" * width)


# ============================================================
# 7. 顶层入口：完整重排流程
# ============================================================

def dynamic_reschedule(
    multi, orig_res, now,
    running_progress=None, forced_delays=None,
    machine_outages=None, cancelled_nodes=None,
    label_fn=None, verbose=True,
    max_extra_minutes=0.0,
    stage1_seconds=12.0, stage2_seconds=12.0,
):
    """
    动态重排完整流程。

    1. 获取状态快照
    2. 冻结已执行节点
    3. 双目标分层优化
    4. 校验资源和依赖
    5. 生成通知
    """
    import time

    tic = time.perf_counter()

    if verbose:
        print(
            f"\n[动态重排] now={now:.1f}分钟"
        )
        print(
            "[优化模式] 双目标分层优化"
        )

    # 1. 状态快照
    t0 = time.perf_counter()

    snapshot = take_snapshot(
        multi,
        orig_res,
        now,
        running_progress=running_progress,
        forced_delays=forced_delays,
        machine_outages=machine_outages,
        cancelled_nodes=cancelled_nodes,
    )

    snapshot_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    # 2. 执行双目标重排
    t0 = time.perf_counter()

    new_res = reschedule_minimal_perturbation(
        multi,
        snapshot,
        verbose=verbose,
        max_extra_minutes=max_extra_minutes,
        stage1_seconds=stage1_seconds,
        stage2_seconds=stage2_seconds,
    )

    resched_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    # 3. 冲突检查
    t0 = time.perf_counter()

    cmp_result = compare_conflicts(
        orig_res,
        new_res,
        multi,
    )

    compare_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    if (
        not cmp_result["device_ok"]
        or not cmp_result["dep_ok"]
    ):
        raise AssertionError(
            f"双目标重排结果存在冲突: "
            f"{cmp_result}"
        )

    # 不再执行5%时间增加就回退的规则

    # 4. 通知
    t0 = time.perf_counter()

    notifications = make_notifications(
        orig_res,
        new_res,
        multi,
        snapshot,
        label_fn=label_fn,
    )

    notify_ms = (
        time.perf_counter() - t0
    ) * 1000.0

    total_ms = (
        time.perf_counter() - tic
    ) * 1000.0

    if verbose:
        print("\n[重排后校验]")

        print(
            f"  冻结节点: "
            f"{len(new_res.frozen_nodes)}"
        )
        print(
            f"  调整节点: "
            f"{len(new_res.changed_nodes)}"
        )
        print(
            f"  makespan: "
            f"{new_res.makespan:.1f}分钟"
        )
        print(
            f"  设备冲突: "
            f"{cmp_result['new_device_conflicts']}"
        )
        print(
            f"  依赖违规: "
            f"{cmp_result['new_dep_violations']}"
        )

        print_notifications(notifications)

    return {
        "snapshot": snapshot,
        "new_res": new_res,
        "compare": cmp_result,
        "notifications": notifications,
        "optimization": getattr(
            new_res,
            "optimize_info",
            {},
        ),
        "timing": {
            "snapshot_ms": round(snapshot_ms, 3),
            "reschedule_ms": round(resched_ms, 3),
            "compare_ms": round(compare_ms, 3),
            "notify_ms": round(notify_ms, 3),
            "total_ms": round(total_ms, 3),
        },
    }
