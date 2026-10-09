# gantt_plot.py
# -*- coding: utf-8 -*-
"""
通用甘特图绘制（单菜 + 多菜自适应）

- 单菜 Instance：每步骤一色，标签 N0 N1 ...
- 多菜 MultiInstance：每菜一色，标签 菜名首字_序号
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False



def plot_gantt_matplotlib(
    inst, res, save_path=None, show=False, figsize=(16, 10), dpi=200,
    title=None, show_labels=True, only_used_machines=True,
    legend_max=30, node_names=None, color_mode="auto",
):
    """一批一块：跨菜共享以斜纹浅底显示，独立设备并行时分层显示。"""
    from collections import defaultdict
    from matplotlib.patches import Patch, Rectangle
    from llm_parser import MODE_DETAIL_NAMES
    from ft_equipment_adapter import machine_display_name
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import os

    records = list(res.op_records)
    auto_heats = list(getattr(res, "preheat_records", ()) or ())

    if not records and not auto_heats:
        print("(无工序可画)")
        return None, None

    is_multi = (
        bool(getattr(inst, "recipe_names", None))
        and hasattr(inst, "recipe_of_node")
    )

    recipe_names = list(inst.recipe_names) if is_multi else ["单菜"]
    n_recipes = len(recipe_names)

    cmap = cm.get_cmap("tab10", max(10, n_recipes))
    recipe_colors = {
        rid: cmap(rid % 10)
        for rid in range(n_recipes)
    }

    makespan = max(
        [float(res.makespan)]
        + [float(r[4]) for r in records]
        + [float(p["end"]) for p in auto_heats]
    )

    # ============================================================
    # 1. 按物理设备和批次分组
    # ============================================================
    grouped = defaultdict(list)

    for rec in records:
        if float(rec[4]) <= float(rec[3]) + 1e-7:
            continue

        mid = int(rec[1])
        bid = rec[2]
        grouped[(mid, bid)].append(rec)

    by_mid = defaultdict(list)

    for (mid, bid), members in grouped.items():
        rids = sorted(set(
            int(inst.recipe_of_node[r[0]]) if is_multi else 0
            for r in members
        ))

        b = {
            "mid": mid,
            "bid": bid,
            "start": min(float(r[3]) for r in members),
            "end": max(float(r[4]) for r in members),
            "members": members,
            "rids": rids,
            "temp": members[0][5] if len(members[0]) >= 8 else 0,
            "mode": members[0][7] if len(members[0]) >= 8 else 0,
        }

        by_mid[mid].append(b)

    # ============================================================
    # 2. 自动预热单独显示
    # ============================================================
    for p in auto_heats:
        mid = int(p["machine_id"])

        by_mid[mid].append({
            "mid": mid,
            "bid": str(p["batch_id"]),
            "start": float(p["start"]),
            "end": float(p["end"]),
            "members": [],
            "rids": list(p.get("recipe_ids") or [0]),
            "temp": p.get("temp", 0),
            "mode": p.get("mode", 0),
            "auto_preheat": True,
        })

    # ============================================================
    # 3. 构建设备行
    #    ID 8 = 准备工序
    #    ID 9 = 其他设备
    # ============================================================
    row_list = []

    for mid in range(inst.K):
        if only_used_machines and mid not in by_mid:
            continue

        batches = sorted(
            by_mid.get(mid, []),
            key=lambda x: (x["start"], x["end"])
        )

        # 容量大于1时，不同并行批次分配子行
        lane_end = []

        for b in batches:
            chosen = next(
                (
                    j for j, end in enumerate(lane_end)
                    if end <= b["start"] + 1e-7
                ),
                None
            )

            if chosen is None:
                chosen = len(lane_end)
                lane_end.append(b["end"])
            else:
                lane_end[chosen] = b["end"]

            b["lane"] = chosen

        width = max(1, len(lane_end))

        # ★ 修改点：统一设备显示名称
        label = machine_display_name(
            inst.machines[mid],
            mid=mid,
            compact=True,
        )

        row_list.append((mid, label, batches, width))

    if not row_list:
        print("(无可绘制设备)")
        return None, None

    # ============================================================
    # 4. 创建画布
    # ============================================================
    total_lanes = sum(row[3] for row in row_list)

    fig_height = min(
        figsize[1],
        max(3.3, total_lanes * 0.9 + 1.8)
    )

    fig, ax = plt.subplots(
        figsize=(figsize[0], fig_height)
    )

    centers = []
    labels = []
    cursor = 0.0
    row_band = 0.83

    # ============================================================
    # 5. 绘制每台设备上的工序
    # ============================================================
    for mid, machine_name, batches, lanes in row_list:
        centers.append(cursor + (lanes - 1) / 2.0)
        labels.append(machine_name)

        # 每个并行子行背景
        for lane in range(lanes):
            y = cursor + lane

            ax.axhspan(
                y - 0.46,
                y + 0.46,
                color="gray",
                alpha=0.025,
                zorder=0
            )

        for b in batches:
            y = cursor + b["lane"]

            s = b["start"]
            e = b["end"]
            length = max(e - s, 0.01)

            shared = len(b["rids"]) > 1

            if b.get("auto_preheat"):
                face = "#FFE4AD"
                edge = "#D97706"
                hatch = "...."
                lw = 1.2

            elif shared:
                face = "#E2EFF9"
                edge = "#287AB3"
                hatch = "////"
                lw = 1.4

            else:
                rid = b["rids"][0]
                face = recipe_colors[rid]
                edge = "black"
                hatch = None
                lw = 0.8

            ax.add_patch(
                Rectangle(
                    (s, y - row_band / 2),
                    length,
                    row_band,
                    facecolor=face,
                    edgecolor=edge,
                    hatch=hatch,
                    linewidth=lw,
                    zorder=2,
                )
            )

            # ====================================================
            # 6. 工序文字标签
            # ====================================================
            if (
                show_labels
                and length >= max(1.5, makespan * 0.006)
            ):
                mode_name = MODE_DETAIL_NAMES.get(
                    b["mode"], ""
                )

                if b.get("auto_preheat"):
                    label = f"自动预热\n{b['temp']}℃"

                elif shared:
                    label = f"共享×{len(b['rids'])}"

                    if mode_name and mode_name != "none":
                        label += f"\n{mode_name}"

                    if b["temp"]:
                        label += f" {b['temp']}℃"

                    if length > makespan * 0.085:
                        label += "\n" + "+".join(
                            recipe_names[r][:5]
                            for r in b["rids"]
                        )

                else:
                    nid = b["members"][0][0]

                    if node_names and nid in node_names:
                        name = node_names[nid]

                    elif hasattr(inst, "step_names"):
                        name = inst.step_names[nid].split("#")[-1]

                    else:
                        name = f"工序{nid}"

                    prefix = (
                        f"{mode_name} {b['temp']}℃\n"
                        if mode_name and b["temp"]
                        else ""
                    )

                    label = prefix + name

                # 窄块竖排显示
                if length < makespan * 0.05:
                    label = "\n".join(
                        label.replace("\n", "")[:10]
                    )
                    font = 6

                else:
                    font = (
                        9 if length >= makespan * 0.13
                        else 7
                    )

                ax.text(
                    s + length / 2,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=font,
                    clip_on=True,
                    zorder=3,
                    color="black",
                )

        cursor += lanes + 0.35

    # ============================================================
    # 7. 设置坐标轴
    # ============================================================
    ax.set_xlim(
        0,
        max(makespan * 1.025, 1.0)
    )

    ax.set_ylim(-0.6, cursor - 0.45)
    ax.invert_yaxis()

    ax.set_yticks(centers)
    ax.set_yticklabels(labels, fontsize=10)

    ax.set_xlabel("时间（分钟）")
    ax.set_ylabel("设备（并行物理工位分层）")

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.35,
        zorder=1,
    )

    ax.set_title(
        title or
        f"多菜并行甘特图  makespan={makespan:.1f}分钟"
    )

    # ============================================================
    # 8. 图例
    # ============================================================
    handles = [
        Patch(
            facecolor=recipe_colors[r],
            edgecolor="black",
            label=recipe_names[r],
        )
        for r in range(min(n_recipes, legend_max))
    ]

    handles.append(
        Patch(
            facecolor="#E2EFF9",
            edgecolor="#287AB3",
            hatch="////",
            label="同批共享（兼容工况）",
        )
    )

    if auto_heats:
        handles.append(
            Patch(
                facecolor="#FFE4AD",
                edgecolor="#D97706",
                hatch="....",
                label="设备自动预热（真实占用）",
            )
        )

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=9,
    )

    # ============================================================
    # 9. 保存
    # ============================================================
    fig.tight_layout()

    if save_path:
        folder = os.path.dirname(save_path)

        if folder:
            os.makedirs(folder, exist_ok=True)

        fig.savefig(
            save_path,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
        )

        print(f"  ✓ 甘特图已保存: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax

