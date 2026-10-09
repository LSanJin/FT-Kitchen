# reminder_generator.py
# -*- coding: utf-8 -*-
"""
实时提醒生成器

在排程执行过程中，根据当前时刻扫描未来窗口内的人工步骤，
生成精准、简洁、可操作的用户提醒。
"""


def generate_reminders(multi, res, now, lookahead=5.0):
    """使用真实人工设备占用区间，生成开始/释放/设备完成提醒。"""
    manual_mids = {
        mid for mid, machine in enumerate(multi.machines)
        if any(k in str(getattr(machine, 'note', '') or '')
               for k in ('人工', '手动操作'))
    }
    manual_spans = {}
    for rec in res.op_records:
        nid, mid, bid, op_start, op_end = rec[:5]
        if int(mid) in manual_mids and float(op_end) > float(op_start) + 1e-7:
            manual_spans.setdefault(int(nid), []).append(
                (float(op_start), float(op_end))
            )

    reminders = []
    now = float(now)
    for nid in range(multi.N):
        start = float(res.node_start[nid])
        end = float(res.node_end[nid])
        rid = int(multi.recipe_of_node[nid])
        recipe_name = multi.recipe_names[rid]
        raw = multi.step_names[nid].split('#')[-1]
        if raw.startswith('S') and raw[1:].isdigit():
            seq = sum(1 for j in range(nid) if multi.recipe_of_node[j] == rid)
            step_name = f'第{seq + 1}步'
        else:
            step_name = raw

        spans = manual_spans.get(nid, [])
        if not spans and bool(multi.needs_human[nid]):
            # 兼容未转换成人工物理设备记录的旧实例。
            spans = [(start, end)] if end > start else []
        manual_end = max((e for _s, e in spans), default=None)
        manual_duration = sum(e - s for s, e in spans)
        has_manual = manual_duration > 1e-7
        delay = start - now

        if has_manual and 0 < delay <= min(3.0, float(lookahead)):
            reminders.append({
                'level': 'warning', 'delay_min': round(delay, 1),
                'time': round(start, 1), 'recipe': recipe_name,
                'step': step_name, 'node_id': nid,
                'message': f'{delay:.0f}分钟后需要{step_name}（{recipe_name}）',
            })
        elif has_manual and -0.5 <= delay <= 0.5:
            suffix = (f'（人工约{manual_duration:g}分钟，随后设备自行运行）'
                      if manual_end is not None and manual_end < end - 1e-7 else '')
            reminders.append({
                'level': 'action', 'delay_min': 0.0,
                'time': round(start, 1), 'recipe': recipe_name,
                'step': step_name, 'node_id': nid,
                'message': f'现在可以开始{step_name}（{recipe_name}）{suffix}',
            })

        # 仅短暂占人工的设备节点：在人工阶段结束时提示释放。
        if (has_manual and manual_end is not None
                and manual_end < end - 1e-7
                and -0.5 <= manual_end - now <= 0.5):
            reminders.append({
                'level': 'info', 'delay_min': 0.0,
                'time': round(manual_end, 1), 'recipe': recipe_name,
                'step': step_name, 'node_id': nid,
                'message': f'{step_name}的人工操作已完成，可处理其他菜品；设备继续运行',
            })

        if end > start + 1e-7 and manual_end != end and 0 < end - now <= 1.0:
            reminders.append({
                'level': 'info', 'delay_min': round(end - now, 1),
                'time': round(end, 1), 'recipe': recipe_name,
                'step': step_name, 'node_id': nid,
                'message': f'{step_name}（{recipe_name}）即将完成，准备下一步',
            })

    reminders.sort(key=lambda x: (x['time'], x['node_id'], x['level']))
    return reminders


def print_reminders(reminders):
    """终端打印提醒"""
    if not reminders:
        print("  （当前无需提醒）")
        return

    icons = {"action": "🔴", "warning": "🟡", "info": "🔵"}
    for r in reminders:
        icon = icons.get(r["level"], "⚪")
        print(f"  {icon} [{r['time']:>5.1f}min] {r['message']}")