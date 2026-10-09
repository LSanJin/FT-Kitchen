# schedule_parser.py
# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# ============================================================
# 1. 数据结构定义
# ============================================================
@dataclass
class Machine:
    id: int
    capacity: int
    rate: float
    temp_c: int
    heat_level: int
    mode: str
    duration_min: int
    note: str


@dataclass
class ScheduleInstance:
    """启发式算法输入实例"""
    N: int                                  # 步骤总数
    E: int                                  # 依赖边数
    K: int                                  # 设备种类数
    capacities: List[int]                   # 每台设备的实例数量
    edges: List[Tuple[int, int]]            # 依赖边 (from, to)
    nodes: List[List[Tuple[int, int]]]      # 每个节点的工序 [(machine_id, duration)]
    machines: List[Machine]                 # 设备参数

    # 便捷索引
    successors: Dict[int, List[int]] = field(default_factory=dict)
    predecessors: Dict[int, List[int]] = field(default_factory=dict)
    indegree: Dict[int, int] = field(default_factory=dict)

    def build_graph(self):
        """构建邻接表、入度，供贪心调度器使用"""
        self.successors = {i: [] for i in range(self.N)}
        self.predecessors = {i: [] for i in range(self.N)}
        self.indegree = {i: 0 for i in range(self.N)}

        for frm, to in self.edges:
            self.successors[frm].append(to)
            self.predecessors[to].append(frm)
            self.indegree[to] += 1

    def validate(self):
        """基础校验"""
        assert self.N == len(self.nodes), \
            f"N={self.N} 与 nodes 数 {len(self.nodes)} 不一致"
        assert self.E == len(self.edges), \
            f"E={self.E} 与 edges 数 {len(self.edges)} 不一致"
        assert self.K == len(self.machines), \
            f"K={self.K} 与 machines 数 {len(self.machines)} 不一致"
        assert self.K == len(self.capacities), \
            f"K={self.K} 与 capacities 数 {len(self.capacities)} 不一致"

        for i, m in enumerate(self.machines):
            assert m.id == i, f"machines[{i}].id={m.id} 顺序错误"
            assert m.capacity == self.capacities[i], \
                f"设备 {i} 容量不一致: {m.capacity} vs {self.capacities[i]}"

        # 依赖边范围检查
        for frm, to in self.edges:
            assert 0 <= frm < self.N, f"边起点越界: {frm}"
            assert 0 <= to < self.N, f"边终点越界: {to}"

        # 节点工序的机器 id 检查
        for i, ops in enumerate(self.nodes):
            for item in ops:
                mid = item[0]
                dur = item[1]
                assert 0 <= mid < self.K, f"节点 {i} 机器 id 越界: {mid}"
                assert dur > 0, f"节点 {i} 时长必须为正: {dur}"

        # 无环检查（拓扑排序）
        from collections import deque
        indeg = dict(self.indegree)
        q = deque([i for i in range(self.N) if indeg[i] == 0])
        seen = 0
        while q:
            cur = q.popleft()
            seen += 1
            for nxt in self.successors[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)
        assert seen == self.N, f"存在循环依赖，仅拓扑出 {seen}/{self.N} 个节点"

        return True


# ============================================================
# 2. 解析函数
# ============================================================
def parse_schedule_text(text: str) -> ScheduleInstance:
    """
    解析大模型输出的调度数据纯文本。

    参数:
        text: 大模型输出的完整纯文本

    返回:
        ScheduleInstance
    """
    # 清洗：去空行、去 markdown 代码块标记、去行末空白
    raw_lines = []
    for ln in text.strip().split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("```"):
            continue
        raw_lines.append(ln)

    lines = raw_lines
    idx = 0

    # ---------- 第1行：N E K ----------
    head = lines[idx].split()
    assert len(head) == 3, f"第1行应为3字段，实际: {lines[idx]}"
    N, E, K = map(int, head)
    idx += 1

    # ---------- 第2行：兼容两种格式 ----------
    parts = list(map(int, lines[idx].split()))
    if len(parts) == K + 1 and parts[0] == K:
        # 标准格式：K + K 个容量
        capacities = parts[1:]
    elif len(parts) == K:
        # 简写格式：直接 K 个容量（LLM 偶尔会这样输出）
        capacities = parts
    else:
        raise ValueError(
            f"第2行格式不对：期望 {K} 个容量（或 K+{K} 个），"
            f"实际 {len(parts)} 个 -> {lines[idx]}"
        )
    idx += 1

    # ---------- 接下来 E 行：依赖边 ----------
    edges = []
    for _ in range(E):
        a, b = map(int, lines[idx].split())
        edges.append((a, b))
        idx += 1

    # ---------- 接下来 N 行：节点工序（新格式含 needs_human）----------
    nodes = []
    needs_human = []
    for _ in range(N):
        row = list(map(int, lines[idx].split()))
        p = row[0]
        nh = row[1]
        needs_human.append(bool(nh))
        if p == 0:
            nodes.append([])
        else:
            ops = [(row[2 + 2 * i], row[3 + 2 * i]) for i in range(p)]
            nodes.append(ops)
        idx += 1

    # ---------- 最后 K 行：设备参数（8 字段） ----------
    machines = []
    for _ in range(K):
        row = lines[idx].split()
        assert len(row) == 8, f"设备行应为8字段，实际: {lines[idx]}"
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
        idx += 1

    # ---------- 行数校验 ----------
    assert idx == len(lines), \
        f"行数不匹配：解析 {idx} 行，实际 {len(lines)} 行"

    # ---------- 组装实例 ----------
    inst = ScheduleInstance(
        N=N, E=E, K=K,
        capacities=capacities,
        edges=edges,
        nodes=nodes,
        machines=machines,
    )
    inst.build_graph()
    inst.validate()
    return inst


# ============================================================
# 3. 友好打印（调试用）
# ============================================================
def print_instance(inst: ScheduleInstance):
    print("=" * 60)
    print(f"N={inst.N}  E={inst.E}  K={inst.K}")
    print(f"设备容量: {inst.capacities}")
    print("-" * 60)
    print("依赖边:")
    for frm, to in inst.edges:
        print(f"  {frm} -> {to}")
    print("-" * 60)
    print("节点工序:")
    for i, ops in enumerate(inst.nodes):
        if not ops:
            print(f"  N{i}: 无设备")
            continue
        parts = []
        for item in ops:
            mid = item[0]
            dur = item[1]
            mname = inst.machines[mid].note
            parts.append(f"机器{mid}({mname}) {dur}min")
        print(f"  N{i}: {', '.join(parts)}")
    print("-" * 60)
    print("设备参数:")
    for m in inst.machines:
        print(f"  [{m.id}] {m.note:<12} cap={m.capacity} "
              f"rate={m.rate} temp={m.temp_c}℃ "
              f"heat={m.heat_level} mode={m.mode} "
              f"total={m.duration_min}min")
    print("=" * 60)


# ============================================================
# 4. 自测入口
# ============================================================
if __name__ == "__main__":
    # 方式一：从文件读取（由 llm_parser.py 生成）
    file_path = "schedule_output.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"未找到 {file_path}，请先运行 llm_parser.py"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    inst = parse_schedule_text(text)
    print_instance(inst)

    # 方式二：直接粘贴字符串
    # text = """10 9 10
    # ..."""
    # inst = parse_schedule_text(text)