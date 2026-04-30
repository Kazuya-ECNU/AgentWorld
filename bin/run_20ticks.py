#!/usr/bin/env python3
"""
20 × 1 Tick 批量运行器

规则:
  1. 每次只跑 1 tick（--count 1），共 20 次
  2. 夜间（21:00-06:00）+6小时；白天（06:00-21:00）+30分钟
  3. 每次输出保存到日志文件
  4. 跑完后打印汇总报告

用法:
  cd ~/Documents/01_Projects/05_AgentWorld
  python3 bin/run_20ticks.py
"""
import sys, os, subprocess, time, json, re
from datetime import datetime

BASE = os.path.expanduser("~/Documents/01_Projects/05_AgentWorld")
LOG_DIR = os.path.join(BASE, "logs", "batch_20tick")
os.makedirs(LOG_DIR, exist_ok=True)

BATCH_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
REPORT_PATH = os.path.join(LOG_DIR, f"summary_{BATCH_ID}.md")

def extract_world_time(log_text: str) -> str:
    """从日志中提取当前世界时间"""
    m = re.search(r'春·第 \d+ 天 \d{2}:\d{2}|夏·第 \d+ 天 \d{2}:\d{2}|秋·第 \d+ 天 \d{2}:\d{2}|冬·第 \d+ 天 \d{2}:\d{2}', log_text)
    return m.group(0) if m else "?"

def extract_trades(log_text: str) -> list:
    """提取库存变化行中的交易"""
    trades = []
    for line in log_text.split('\n'):
        if '→' in line and ('小麦' in line or '蔬菜' in line or '皮毛' in line or '米酒' in line or '工具' in line or '铁矿石' in line):
            if '库存变化' in line:
                continue
            trades.append(line.strip())
    return trades

def extract_npc_table(log_text: str) -> str:
    """提取 NPC 属性变化表格"""
    lines = log_text.split('\n')
    in_table = False
    table = []
    for line in lines:
        if 'NPC' in line and '位置' in line and 'V' in line:
            in_table = True
        if in_table:
            table.append(line)
            if 'LLM 调用明细' in line:
                break
    return '\n'.join(table)

def run_one_tick(tick_index: int) -> dict:
    """执行一次 1 tick，返回结果摘要"""
    print(f"\n{'='*60}")
    print(f"  Tick {tick_index+1}/20 — {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    log_path = os.path.join(LOG_DIR, f"tick{tick_index+1:02d}_{BATCH_ID}.log")
    script = os.path.join(BASE, "bin", "run_tick_report.py")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script, "--count", "1"],
        cwd=BASE,
        capture_output=True, text=True,
        timeout=600
    )
    elapsed = time.time() - t0

    stdout = result.stdout
    stderr = result.stderr

    # 写日志
    with open(log_path, 'w') as f:
        f.write(f"=== Tick {tick_index+1}/20 ===\n")
        f.write(f"Run at: {datetime.now().isoformat()}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n")
        f.write(f"Exit code: {result.returncode}\n")
        f.write(f"{'─'*50}\n")
        f.write(stdout)
        if stderr:
            f.write(f"\n{'─'*50}\nSTDERR:\n{stderr}\n")

    # 解析结果
    world_time = extract_world_time(stdout)
    trades = extract_trades(stdout)
    npc_table = extract_npc_table(stdout)
    llm_count = stdout.count("LLM #")
    has_error = result.returncode != 0 or "Traceback" in stdout or "Error" in stdout

    summary = {
        "tick": tick_index + 1,
        "elapsed": round(elapsed, 1),
        "world_time": world_time,
        "llm_calls": llm_count,
        "trades": trades,
        "has_error": has_error,
        "log_path": log_path,
        "stdout_preview": stdout[:2000] if has_error else stdout[:500],
    }

    # 打印摘要
    status = "✅" if not has_error else "❌"
    print(f"  {status} {elapsed:.1f}s | {world_time} | LLM×{llm_count}")
    if trades:
        for t in trades[:3]:
            print(f"    交易: {t}")
    if has_error:
        print(f"    ❌ 错误！日志: {log_path}")
        print(stdout[:1000])

    return summary


def format_report(all_results: list[dict]):
    """生成最终的 Markdown 报告"""
    lines = []
    lines.append(f"# 批量运行报告 — {BATCH_ID}")
    lines.append(f"\n共 {len(all_results)} tick，{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"\n## 概览")
    lines.append(f"\n| Tick | 时间 | LLM 数 | 耗时 | 结果 |")
    lines.append(f"| ---- | ---- | ------ | ---- | ---- |")

    success = 0
    total_trades = 0
    for r in all_results:
        status = "✅" if not r['has_error'] else "❌"
        lines.append(f"| {r['tick']:>2d} | {r['world_time']} | {r['llm_calls']} | {r['elapsed']:.0f}s | {status} |")
        if not r['has_error']:
            success += 1
        total_trades += len(r['trades'])

    lines.append(f"\n**结果**: {success}/{len(all_results)} 成功 | 总交易: {total_trades} | 总耗时: {sum(r['elapsed'] for r in all_results):.0f}s")

    # 交易明细
    lines.append(f"\n## 交易记录")
    for r in all_results:
        if r['trades']:
            lines.append(f"\n### Tick {r['tick']} ({r['world_time']})")
            for t in r['trades']:
                lines.append(f"- {t}")

    # 时间线
    lines.append(f"\n## 世界时间线")
    for r in all_results:
        lines.append(f"- Tick {r['tick']:>2d}: {r['world_time']}")

    # 错误记录
    errors = [r for r in all_results if r['has_error']]
    if errors:
        lines.append(f"\n## 错误记录")
        for r in errors:
            lines.append(f"\n### Tick {r['tick']}")
            lines.append(f"```\n{r['stdout_preview']}\n```")

    return '\n'.join(lines)


def main():
    print(f"20 × 1 Tick 批量运行")
    print(f"日志目录: {LOG_DIR}")
    print(f"规则: 夜间(21-06) +6h, 白天(06-21) +30min")

    results = []
    for i in range(20):
        summary = run_one_tick(i)
        results.append(summary)
        # 异常快速终止
        if summary['has_error']:
            print(f"\n⚠️  Tick {i+1} 出错，继续下一轮...")

    # 生成报告
    report = format_report(results)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"  运行完成！")
    print(f"  报告: {REPORT_PATH}")
    print(f"  成功: {len([r for r in results if not r['has_error']])}/{len(results)}")
    print(f"{'='*60}")
    print(report)


if __name__ == '__main__':
    main()
