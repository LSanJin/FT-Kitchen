# greedy_scheduler.py
# -*- coding: utf-8 -*-

import time
import heapq
from collections import deque
import re
from llm_parser import BATCH_CAPACITY, DEFAULT_BATCH_CAPACITY



def unpack_op(rec):
    """
    把 op_records 的一条记录解包为标准 8 元组。
    返回 (nid, mid, bid, s, e, temp, heat, mode)
    兼容旧 5 元组和新 8 元组。
    """
    n = len(rec)
    if n >= 8:
        return (rec[0], rec[1], rec[2], rec[3],
                rec[4], rec[5], rec[6], rec[7])
    elif n >= 5:
        return (rec[0], rec[1], rec[2], rec[3],
                rec[4], 0, 0, 0)
    else:
        raise ValueError(f"无法识别的 op_record: {rec}")

# ============================================================
# 1. 数据结构
# ============================================================
class Machine:
    __slots__ = ("id", "capacity", "rate", "temp_c",
                 "heat_level", "mode", "duration_min", "note",
                 "mode_detail")
    def __init__(self, id, capacity, rate, temp_c,
                 heat_level, mode, duration_min, note,
                 mode_detail="none"):
        self.id = id
        self.capacity = capacity
        self.rate = rate
        self.temp_c = temp_c
        self.heat_level = heat_level
        self.mode = mode
        self.duration_min = duration_min
        self.note = note
        self.mode_detail = mode_detail


class Instance:
    __slots__ = ("N", "E", "K", "capacities", "edges", "nodes",
                 "machines", "successors", "predecessors",
                 "indegree", "node_duration", "needs_human",
                 "step_names")
    def __init__(self, N, E, K, capacities, edges, nodes, machines,
                 needs_human=None):
        self.N = N
        self.E = E
        self.K = K
        self.capacities = capacities
        self.edges = edges
        self.nodes = nodes
        self.machines = machines
        self.successors = [[] for _ in range(N)]
        self.predecessors = [[] for _ in range(N)]
        self.indegree = [0] * N
        self.node_duration = [0] * N
        self.needs_human = needs_human or [True] * N

    def build(self):
        for frm, to in self.edges:
            self.successors[frm].append(to)
            self.predecessors[to].append(frm)
            self.indegree[to] += 1
        for i, ops in enumerate(self.nodes):
            # ops 现在是 5 元组 (mid, dur, temp, heat, mode)
            # 也可能是 2 元组（兼容旧数据）
            total = 0
            for item in ops:
                if len(item) >= 2:
                    total += item[1]
            self.node_duration[i] = total




def parse_schedule_text(text: str, needs_human=None):
    """
    解析大模型输出的调度数据纯文本。
    容错：自动去除括号、逗号、多余空格，兼容多种分隔符。
    """
    # 清洗：去空行、去 markdown 代码块标记
    lines = []
    for ln in text.strip().split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("```"):
            continue
        lines.append(ln)

    def ints(line: str):
        """从一行里提取所有整数，兼容 (0 2)、0,2、0 2 等写法"""
        return [int(x) for x in re.findall(r"-?\d+", line)]

    idx = 0

    # ---------- 第1行：N E K ----------
    head = ints(lines[idx])
    assert len(head) == 3, f"第1行应为3个整数，实际: {lines[idx]}"
    N, E, K = head
    idx += 1

    # ---------- 第2行：兼容两种格式 ----------
    parts = ints(lines[idx])
    if len(parts) == K + 1 and parts[0] == K:
        capacities = parts[1:]
    elif len(parts) == K:
        capacities = parts
    else:
        raise ValueError(
            f"第2行格式不对：期望 {K} 或 {K+1} 个整数，"
            f"实际 {len(parts)} 个 -> {lines[idx]}"
        )
    idx += 1

    # ---------- 接下来 E 行：依赖边 ----------
    edges = []
    for _ in range(E):
        row = ints(lines[idx])
        assert len(row) == 2, f"边行应为2个整数: {lines[idx]}"
        edges.append((row[0], row[1]))
        idx += 1

    # ---------- 接下来 N 行：节点工序 ----------
    nodes_raw = []
    needs_human_list = []
    for _ in range(N):
        row = list(map(int, lines[idx].split()))
        p = row[0]
        nh = row[1]
        needs_human_list.append(bool(nh))
        ops = []
        for i in range(p):
            mid = row[2 + 2 * i]
            dur = row[3 + 2 * i]
            ops.append((mid, dur))
        nodes_raw.append(ops)
        idx += 1

    # ---------- ★ 接下来 N 行：节点参数行 ----------
    nodes = []
    for sid in range(N):
        p = len(nodes_raw[sid])
        if p == 0:
            nodes.append([])
            # 跳过参数行
            idx += 1
            continue
        row = list(map(int, lines[idx].split()))
        combined = []
        for i in range(p):
            temp = row[4 * i]
            heat = row[4 * i + 1]
            mode = row[4 * i + 2]
            # dur 在节点行已记录，这里忽略
            mid, dur = nodes_raw[sid][i]
            combined.append((mid, dur, temp, heat, mode))
        nodes.append(combined)
        idx += 1

    # ---------- 最后 K 行：设备参数（8 字段，容错） ----------
    machines = []
    for _ in range(K):
        line = lines[idx]
        row = line.split()

        if len(row) == 8:
            # 标准格式
            machines.append(Machine(
                id=int(row[0]),
                capacity=int(row[1]),
                rate=float(row[2]),
                temp_c=int(row[3]),
                heat_level=int(row[4]),
                mode=row[5],
                duration_min=int(row[6]),
                note=row[7],
            ))
        else:
            # 容错：duration_min 和 note 可能粘在一起
            # 格式：id cap rate temp heat mode dur note
            m = re.match(
                r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+"
                r"(\S+)\s+(\d+)\s*(.*?)\s*$",
                line
            )
            if not m:
                raise ValueError(f"设备行无法解析: {line}")
            machines.append(Machine(
                id=int(m.group(1)),
                capacity=int(m.group(2)),
                rate=float(m.group(3)),
                temp_c=int(m.group(4)),
                heat_level=int(m.group(5)),
                mode=m.group(6),
                duration_min=int(m.group(7)),
                note=m.group(8) if m.group(8) else "none",
            ))
        idx += 1

    assert idx == len(lines), \
        f"行数不匹配：解析 {idx} 行，实际 {len(lines)} 行"

    inst = Instance(N, E, K, capacities, edges, nodes, machines)
    inst.needs_human = needs_human_list
    inst.build()
    _validate(inst)
    return inst


def _validate(inst: Instance):
    # ============================================================
    # ★ 1. 自动补全越界设备；100 号固定为"等待"，容量 999
    # ============================================================
    max_mid = 0
    for ops in inst.nodes:
        for item in ops:
            if item[0] > max_mid:
                max_mid = item[0]

    if max_mid >= inst.K:
        need = max_mid + 1

        # machines 补到 need 长，中间用 None 占位
        while len(inst.machines) < need:
            inst.machines.append(None)

        # capacities 补到 need 长，中间用 0 占位
        if hasattr(inst, "capacities"):
            while len(inst.capacities) < need:
                inst.capacities.append(0)

        # ★ 100 号 = 等待设备，容量 999
        if max_mid >= 100:
            if inst.machines[100] is None:
                new_m = None
                try:
                    if inst.machines and inst.machines[0] is not None:
                        import copy
                        new_m = copy.deepcopy(inst.machines[0])
                        if hasattr(new_m, "note"):
                            new_m.note = "等待"
                        if hasattr(new_m, "id"):
                            new_m.id = 100
                        if hasattr(new_m, "capacity"):
                            new_m.capacity = 999
                except Exception:
                    new_m = None
                inst.machines[100] = new_m

            if hasattr(inst, "capacities"):
                inst.capacities[100] = 999

        # 其它扩展槽位（8~99 或 101+）也补个占位设备，避免 None 崩
        for k in range(inst.K, need):
            if inst.machines[k] is None and k != 100:
                new_m = None
                try:
                    if inst.machines and inst.machines[0] is not None:
                        import copy
                        new_m = copy.deepcopy(inst.machines[0])
                        if hasattr(new_m, "note"):
                            new_m.note = f"扩展设备{k}"
                        if hasattr(new_m, "id"):
                            new_m.id = k
                        if hasattr(new_m, "capacity"):
                            new_m.capacity = 1
                except Exception:
                    new_m = None
                inst.machines[k] = new_m

                if hasattr(inst, "capacities") and inst.capacities[k] == 0:
                    inst.capacities[k] = 1

        inst.K = need

    # ============================================================
    # ★ 2. machines[i].id / capacity 与 capacities 一致性
    #    跳过 None 占位
    # ============================================================
    for i, m in enumerate(inst.machines):
        if m is None:
            continue
        if hasattr(m, "id"):
            assert m.id == i, f"设备{i}的 id={m.id} 不一致"
        if hasattr(m, "capacity") and hasattr(inst, "capacities"):
            if i < len(inst.capacities):
                assert m.capacity == inst.capacities[i], (
                    f"设备{i} capacity={m.capacity} 与 "
                    f"capacities[{i}]={inst.capacities[i]} 不一致"
                )

    # ============================================================
    # ★ 3. 节点设备号 / 时长 / 容量校验
    # ============================================================
    # ============================================================
    # ★ 节点校验 + 时长兜底
    # ============================================================
    SHORT_OP_KEYWORDS = (
        "取水", "搅拌", "打蛋", "调味", "刷油",
        "爆香", "撒", "淋", "装盘", "摆盘",
    )

    for i, ops in enumerate(inst.nodes):
        fixed_ops = []

        for item in ops:
            item = list(item)  # 允许修改
            mid, dur = item[0], item[1]

            # ---- 设备号校验 ----
            assert 0 <= mid < inst.K, f"节点{i}机器id越界: {mid}"

            # ---- 容量校验 ----
            cap = inst.capacities[mid] if hasattr(inst, "capacities") else 1
            if cap <= 0:
                note = ""
                if mid < len(inst.machines) and inst.machines[mid] is not None:
                    note = getattr(inst.machines[mid], "note", "") or ""
                raise ValueError(
                    f"节点{i}使用了容量为0的设备{mid}({note})"
                )

            # ---- 时长兜底 ----
            if dur is None or dur <= 0:
                name = (
                    inst.step_names[i].split("#")[-1]
                    if hasattr(inst, "step_names") and i < len(inst.step_names)
                    else ""
                )
                if any(kw in name for kw in SHORT_OP_KEYWORDS):
                    dur = 1
                else:
                    dur = 3
                item[1] = dur
                print(f"  [兜底] 节点{i} '{name}' 时长 0 → {dur} 分钟")

            fixed_ops.append(tuple(item))

        inst.nodes[i] = fixed_ops

    # ---- 重新计算 node_duration ----
    for i, ops in enumerate(inst.nodes):
        total = 0
        for item in ops:
            if len(item) >= 2:
                total += item[1]
        inst.node_duration[i] = total

    # ============================================================
    # ★ 4. 无环检查
    # ============================================================
    indeg = list(inst.indegree)
    q = deque([i for i in range(inst.N) if indeg[i] == 0])
    seen = 0
    while q:
        cur = q.popleft()
        seen += 1
        for nxt in inst.successors[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    assert seen == inst.N, f"存在循环依赖: {seen}/{inst.N}"

# ============================================================
# 3. 堆栈贪心调度
# ============================================================
class ScheduleResult:
    __slots__ = ("node_start", "node_end", "op_records",
                 "makespan", "algo_time_ms", "num_push", "num_pop")
    def __init__(self):
        self.node_start = []
        self.node_end = []
        self.op_records = []      # (node_id, machine_id, instance_id, start, end)
        self.makespan = 0
        self.algo_time_ms = 0.0
        self.num_push = 0
        self.num_pop = 0


# ============================================================
# 批次调度核心
# ============================================================
_batch_counter = [0]

def _new_batch_id():
    _batch_counter[0] += 1
    return _batch_counter[0]


def allocate_batch(
    batches, key, est, duration, cap, machine_capacity=1
):
    """
    为物理设备分配批次。

    key：模式、温度、火力、工序类别
    cap：同一个批次最多容纳的工序数量
    machine_capacity：独立物理设备/工位数量

    相同工况允许跨菜共享，不要求持续时间相同，
    但必须同批开始，各工序按自己的持续时间结束。
    """
    EPS = 1e-7
    est = max(0.0, float(est))
    duration = max(0.0, float(duration))
    cap = max(1, int(cap))
    machine_capacity = max(1, int(machine_capacity))

    def has_room(start, end, exclude=None):
        if end <= start + EPS:
            return True

        events = [(start, 1), (end, -1)]

        for b in batches:
            if b is exclude:
                continue

            bs = float(b["start"])
            be = float(b["end"])

            if be <= start + EPS or bs >= end - EPS:
                continue

            events.append((max(bs, start), 1))
            events.append((min(be, end), -1))

        events.sort(key=lambda t: (t[0], t[1]))

        active = 0
        for _, delta in events:
            active += delta
            if active > machine_capacity:
                return False

        return True

    # 先寻找最早可创建独立批次的时间
    candidate = est

    for _ in range(len(batches) + 2):
        if has_room(candidate, candidate + duration):
            break

        next_ends = [
            float(b["end"])
            for b in batches
            if float(b["end"]) > candidate + EPS
            and float(b["start"]) < candidate + duration - EPS
        ]

        if not next_ends:
            raise RuntimeError("无法找到设备空闲时间")

        candidate = min(next_ends)

    else:
        raise RuntimeError("批次分配未收敛")

    # 查找兼容且尚未开始的共享批次
    compatible = []

    for b in batches:
        if b.get("key") != key:
            continue

        if float(b["start"]) < est - EPS:
            continue

        if int(b.get("used", 1)) >= min(
            int(b.get("cap", 1)), cap
        ):
            continue

        start = float(b["start"])
        extended_end = max(
            float(b["end"]),
            start + duration
        )

        if has_room(start, extended_end, exclude=b):
            compatible.append(
                (start, b, extended_end)
            )

    compatible.sort(
        key=lambda x: (x[0], x[1]["id"])
    )

    # 如果共享不比独立批次开始得更晚，就优先共享
    if compatible and compatible[0][0] <= candidate + EPS:
        start, batch, extended_end = compatible[0]

        batch["used"] = int(batch.get("used", 1)) + 1
        batch["end"] = extended_end

        return (
            start,
            start + duration,
            batch["id"]
        )

    # 创建新批次
    start = candidate
    end = start + duration

    bid = _new_batch_id()

    # 防止多轮动态重排恢复的旧批次ID发生碰撞
    existing_ids = {b["id"] for b in batches}
    while bid in existing_ids:
        bid = _new_batch_id()

    batches.append({
        "key": key,
        "start": start,
        "end": end,
        "used": 1,
        "cap": cap,
        "id": bid,
    })

    return start, end, bid

def greedy_schedule(inst: Instance) -> ScheduleResult:
    """
    堆栈贪心调度。

    模型：
    - 一个节点 = 一个步骤，内部多个工序并行（同时开始、同时结束）
    - 边 = 前后依赖约束
    - 每个设备有容量 capacity（可并行使用份数）
    - 目标：最小化 makespan

    优先级：关键路径长度（rank）降序，再按节点总时长降序
    """
    t0 = time.perf_counter()
    res = ScheduleResult()

    N, K = inst.N, inst.K
    successors = inst.successors
    predecessors = inst.predecessors
    node_duration = inst.node_duration
    nodes = inst.nodes

    # ---------- 3.1 拓扑排序 ----------
    indeg = list(inst.indegree)
    topo = []
    q = deque(i for i in range(N) if indeg[i] == 0)
    while q:
        cur = q.popleft()
        topo.append(cur)
        for nxt in successors[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)

    # ---------- 3.2 逆序计算关键路径 rank ----------
    rank = [0] * N
    for sid in reversed(topo):
        succs = successors[sid]
        if succs:
            rank[sid] = node_duration[sid] + max(rank[s] for s in succs)
        else:
            rank[sid] = node_duration[sid]

    # ---------- 3.3 批次列表（每台设备一个）----------
    machine_batches = [[] for _ in range(K)]

    # 人工资源堆（容量 1）
    human_heap = [(0, 0)]
    heapq.heapify(human_heap)

    # ---------- 3.4 就绪堆 ----------
    ready = []
    for i in range(N):
        if inst.indegree[i] == 0:
            heapq.heappush(ready, (-rank[i], -node_duration[i], i))

    node_start = [0] * N
    node_end = [0] * N
    finish_time = [0] * N
    op_records = []
    indeg = list(inst.indegree)
    scheduled = 0

    # ---------- 3.5 主循环 ----------
    while ready:
        _, _, sid = heapq.heappop(ready)

        est = 0
        for p in predecessors[sid]:
            if finish_time[p] > est:
                est = finish_time[p]

        # ★★★ 预热节点：尽量贴紧后续烹饪
        step_name_sid = (
            inst.step_names[sid].split("#")[-1]
            if hasattr(inst, "step_names") else ""
        )
        if "预热" in step_name_sid:
            succs = successors[sid]
            if succs:
                T_est = max(rank) if rank else 0.0
                succ_earliest = min(T_est - rank[s] for s in succs)
                preheat_dur = node_duration[sid]
                preheat_target = succ_earliest - preheat_dur
                if preheat_target > est:
                    est = preheat_target

        current_time = est

        # ★ 人工资源（预热不占人工）
        needs_human_sid = bool(inst.needs_human[sid])
        if needs_human_sid and "预热" not in step_name_sid:
            h_avail, _ = heapq.heappop(human_heap)
            if h_avail > current_time:
                current_time = h_avail

        # ============================================================
        # ★★★ 节点内工序并行处理
        # ============================================================
        assignments = []
        for item in nodes[sid]:
            mid, dur, temp, heat, mode = item

            key = (0, mode, temp, heat, dur)
            if mid == 7:
                cap = 1
            else:
                cap = BATCH_CAPACITY.get(mid, DEFAULT_BATCH_CAPACITY)
            mc = inst.capacities[mid]

            # 每个工序独立向 allocate_batch 申请
            # 参数 est 用 current_time，但先不记录，等对齐后再记录
            op_s, op_e, bid = allocate_batch(
                machine_batches[mid], key, current_time, dur, cap,
                machine_capacity=mc,
            )
            assignments.append({
                "mid": mid, "bid": bid,
                "dur": dur, "earliest": op_s,
                "temp": temp, "heat": heat, "mode": mode,
            })

        # 所有工序对齐到统一开始时间
        if assignments:
            start_aligned = max(
                max(current_time, a["earliest"]) for a in assignments
            )
            max_dur = max(a["dur"] for a in assignments)
            end_aligned = start_aligned + max_dur

            for a in assignments:
                op_e = start_aligned + a["dur"]
                op_records.append((
                    sid, a["mid"], a["bid"],
                    start_aligned, op_e,
                    a["temp"], a["heat"], a["mode"],
                ))

            # ★ 用 start_aligned 和 end_aligned
            node_start[sid] = start_aligned
            node_end[sid] = end_aligned
            finish_time[sid] = end_aligned
            current_time = end_aligned
        else:
            # ★ 空工序节点：直接用 est
            node_start[sid] = est
            node_end[sid] = est
            finish_time[sid] = est

        # 释放人工资源
        if needs_human_sid and "预热" not in step_name_sid:
            heapq.heappush(human_heap, (current_time, 0))

        scheduled += 1

        for nxt in successors[sid]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                heapq.heappush(
                    ready, (-rank[nxt], -node_duration[nxt], nxt)
                )

    # ============================================================
    # ---------- 3.9 级联修正：确保所有依赖满足 ----------
    # ============================================================
    for _ in range(N):
        changed = False
        for nid in range(N):
            # 计算所有前置的最大结束时间
            pred_end = 0.0
            for p in predecessors[nid]:
                if node_end[p] > pred_end:
                    pred_end = node_end[p]

            if node_start[nid] < pred_end - 1e-6:
                delta = pred_end - node_start[nid]
                # 推后该节点
                node_start[nid] += delta
                node_end[nid] += delta
                finish_time[nid] += delta

                # 推后该节点在 op_records 里的所有工序
                for k, rec in enumerate(op_records):
                    if rec[0] == nid:
                        op_records[k] = (
                            rec[0], rec[1], rec[2],
                            rec[3] + delta, rec[4] + delta,
                            rec[5], rec[6], rec[7],
                        )
                changed = True

        if not changed:
            break

    # ============================================================
    # ---------- 3.10 重建 machine_batches（与 op_records 同步）----------
    # ============================================================
    machine_batches = [[] for _ in range(K)]  # ★ K = inst.K
    for rec in op_records:
        nid, mid, bid = rec[0], rec[1], rec[2]
        s, e = rec[3], rec[4]
        # 从原始节点信息取 mode/temp/heat/dur（用于批次 key）
        temp, heat, mode, dur = 0, 0, 0, int(e - s)
        for item in nodes[nid]:
            if item[0] != mid:
                continue
            dur = item[1]
            if len(item) >= 5:
                temp, heat, mode = item[2], item[3], item[4]
            break

        machine_batches[mid].append({
            "key": (0, mode, temp, heat, dur),  # 单菜 rid=0
            "start": s,
            "end": e,
            "used": 1,
            "cap": 1,
            "id": bid,
        })


    assert scheduled == N, f"仅调度 {scheduled}/{N} 个节点"

    res.node_start = node_start
    res.node_end = node_end
    res.op_records = op_records
    res.makespan = max(finish_time) if finish_time else 0
    res.algo_time_ms = (time.perf_counter() - t0) * 1000.0
    return res

# ============================================================
# 4. 甘特图输出
# ============================================================
def print_gantt(inst: Instance, res: ScheduleResult,
                bar_width: int = 50, show_node_id: bool = True):
    """
    ASCII 甘特图。
    每台设备一行时间轴，字符 '█' 表示占用，'░' 表示空闲。
    多实例设备用不同行区分。
    """
    makespan = res.makespan
    if makespan <= 0:
        print("(makespan=0，无内容)")
        return

    scale = makespan / bar_width  # 1 个字符代表多少分钟

    # 按设备分组 op_records
    by_machine = [[] for _ in range(inst.K)]
    for sid, mid, bid, s, e in res.op_records:
        by_machine[mid].append((sid, bid, s, e))

    # 时间轴刻度
    print()
    print("=" * (20 + bar_width + 4))
    print(f"甘特图  (makespan = {makespan} 分钟, 每字符 ≈ {scale:.2f} 分钟)")
    print("=" * (20 + bar_width + 4))

    # 顶部时间刻度
    ruler = [" "] * bar_width
    num_ticks = 6
    for t in range(num_ticks):
        pos = int(t * (bar_width - 1) / (num_ticks - 1))
        label = str(int(round(t * makespan / (num_ticks - 1))))
        for j, ch in enumerate(label):
            if pos + j < bar_width:
                ruler[pos + j] = ch
    print(" " * 20 + "|" + "".join(ruler) + "|")

    # 每台设备一行
    for k in range(inst.K):
        m = inst.machines[k]
        name = m.note if m.note else f"m{k}"
        label = f"[{k}] {name}"
        label = label[:18].ljust(18)

        ops = by_machine[k]
        if not ops:
            print(f"{label}  |" + " " * bar_width + "|  (空闲)")
            continue

        # 只用一行展示（多实例也合并显示）
        bar = ["░"] * bar_width
        for sid, bid, s, e in ops:
            i0 = int(s / scale)
            i1 = int(round(e / scale))
            i0 = max(0, min(bar_width - 1, i0))
            i1 = max(i0 + 1, min(bar_width, i1))
            for j in range(i0, i1):
                bar[j] = "█"

        print(f"{label}  |" + "".join(bar) + "|")
        # 附加说明：该设备上的工序
        for sid, bid, s, e in sorted(ops, key=lambda x: x[2]):
            tag = f"N{sid}" if show_node_id else ""
            print(f"{'':18}     └─ {tag:<6} 实例{bid}  "
                  f"[{s:>3}, {e:>3}]  {e-s}min")

    print("=" * (20 + bar_width + 4))
    print("图例: █ = 设备被占用   ░ = 空闲")
    print()


def print_time_table(inst: Instance, res: ScheduleResult):
    """按时间排序列出每道工序"""
    print("=" * 70)
    print("详细工序表（按开始时间排序）")
    print("=" * 70)
    print(f"{'节点':<6}{'设备':<14}{'实例':<6}"
          f"{'开始':<8}{'结束':<8}{'时长':<6}")
    print("-" * 70)
    for sid, mid, bid, s, e in sorted(res.op_records, key=lambda x: x[3]):
        mname = inst.machines[mid].note
        print(f"N{sid:<5}{mname:<14}{bid:<6}{s:<8}{e:<8}{e-s:<6}")
    print("-" * 70)
    print(f"makespan = {res.makespan} 分钟")
    print()


def print_summary(inst: Instance, res: ScheduleResult,
                  parse_ms: float, total_ms: float):
    print("=" * 70)
    print("运行摘要")
    print("=" * 70)
    print(f"节点数 N              : {inst.N}")
    print(f"依赖边数 E            : {inst.E}")
    print(f"设备种类 K            : {inst.K}")
    print(f"堆 push 次数          : {res.num_push}")
    print(f"堆 pop  次数          : {res.num_pop}")
    print(f"解析耗时              : {parse_ms:.3f} ms")
    print(f"贪心调度耗时          : {res.algo_time_ms:.3f} ms")
    print(f"总耗时                : {total_ms:.3f} ms")
    print(f"makespan              : {res.makespan} 分钟")
    print("=" * 70)


# ============================================================
# 5. 主入口
# ============================================================
def run(schedule_text: str,
        bar_width: int = 50,
        show_gantt: bool = True,
        show_table: bool = True,
        show_summary: bool = True):
    t_total_0 = time.perf_counter()

    # 解析
    t_parse_0 = time.perf_counter()
    inst = parse_schedule_text(schedule_text)
    parse_ms = (time.perf_counter() - t_parse_0) * 1000.0

    # 调度
    res = greedy_schedule(inst)

    total_ms = (time.perf_counter() - t_total_0) * 1000.0

    # 输出
    if show_gantt:
        print_gantt(inst, res, bar_width=bar_width)
    if show_table:
        print_time_table(inst, res)
    if show_summary:
        print_summary(inst, res, parse_ms, total_ms)

    return inst, res


# ============================================================
# 6. 自测：读取 schedule_output.txt 或内置示例
# ============================================================
SAMPLE_TEXT = """10 9 7
7 1 2 1 1 4 2 1
0 3
1 3
3 4
4 5
5 6
6 7
7 8
8 9
2 6
2 0 5 4 2
1 4 5
2 4 15 8 15
2 0 4 5 4
2 1 2 4 1
4 3 4 5 1 4 1 0 2
2 2 1 3 1
2 2 1 3 1
1 2 15
2 2 1 3 1
0 1 0.05 0 0 cut 5 砧板
1 2 0.10 0 3 fry 6 灶台
2 1 0.08 180 2 bake 16 烤箱_180C
3 1 0.02 0 0 none 0 电饭煲_未用
4 4 0.01 0 0 none 18 碗
5 2 0.06 0 3 fry 6 炒锅
6 1 0.01 4 0 chill 15 冰箱_腌制
"""


if __name__ == "__main__":
    import os

    path = "schedule_output.txt"
    if os.path.exists(path):
        print(f"[main] 从文件读取: {path}")
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        print("[main] 未找到 schedule_output.txt，使用内置示例")
        text = SAMPLE_TEXT

    run(text, bar_width=50,
        show_gantt=True,
        show_table=True,
        show_summary=True)