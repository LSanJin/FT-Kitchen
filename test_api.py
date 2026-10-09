import subprocess
import time
import requests
import sys
import os

BASE = "http://localhost:8000"

# ============================================================
# 1. 后台启动 uvicorn
# ============================================================
print("[1/2] 启动 API 服务...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api:app",
     "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd=os.path.dirname(os.path.abspath(__file__)),
)

# 等待服务就绪
ready = False
for i in range(30):
    time.sleep(1)
    try:
        r = requests.get(f"{BASE}/health", timeout=2)
        if r.status_code == 200:
            ready = True
            print(f"  ✓ 服务就绪（等待 {i+1}s）")
            break
    except Exception:
        pass

if not ready:
    print("  ✗ 服务启动失败")
    proc.terminate()
    sys.exit(1)

# ============================================================
# 2. 测试接口
# ============================================================
try:
    # 健康检查
    r = requests.get(f"{BASE}/health")
    print(f"\n[测试] /health: {r.status_code} {r.json()}")

    # 主接口
    recipes = [
        {"id": "5d54bae2a9114174727c8b20", "name": "鸡翅包饭"},
        {"id": "5dad79866601a865649f9cb7", "name": "家常鲈鱼"},
        {"id": "59cc89b47d21be2a79172c98", "name": "豉汁蒸草鱼"},
    ]
    t0 = time.time()
    r = requests.post(f"{BASE}/schedule", json=recipes, timeout=60)
    t1 = time.time()
    print(f"\n[测试] /schedule: {r.status_code}  "
          f"端到端 {t1-t0:.2f}s")
    if r.status_code == 200:
        data = r.json()
        print(f"  顶层字段: {list(data.keys())}")
        print(f"  overview: {data.get('overview')}")

    # 重排
    body = {"recipes": recipes, "now": 20.0}
    t0 = time.time()
    r = requests.post(f"{BASE}/reschedule", json=body, timeout=60)
    t1 = time.time()
    print(f"\n[测试] /reschedule: {r.status_code}  "
          f"端到端 {t1-t0:.2f}s")
    if r.status_code == 200:
        data = r.json()
        print(f"  顶层字段: {list(data.keys())}")
        print(f"  overview: {data.get('overview')}")

finally:
    # 关掉服务
    print("\n[3/3] 关闭服务...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    print("  ✓ 已关闭")