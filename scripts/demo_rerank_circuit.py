"""demo_rerank_circuit.py
演示 Rerank 远程调用的熔断 (circuit breaker) 触发与冷却复位流程。

运行方式:
  1. 确保 .env 中启用 RERANK_ENABLED=true，并设置 RERANK_CB_FAILS=3 (默认) RERANK_CB_COOLDOWN=10 (可临时调小)
  2. python scripts/demo_rerank_circuit.py

脚本流程:
  - 第一步: 将 settings.rerank_endpoint 指向一个不存在的地址 (127.0.0.1:9999/nope) 连续触发调用失败
  - 观察 _CB_STATE 中 fails 计数增长, 直至 >= RERANK_CB_FAILS 时 opened_until 设置 => 熔断打开
  - 熔断打开后再次调用立即抛出 circuit_open (不再真正发起 HTTP)
  - 等待冷却时间结束 (或强制 fast-forward) 后, 切换到一个临时本地 mock server
  - 启动 mock rerank server (Thread + HTTPServer) 返回固定 scores
  - 再次调用, 看到熔断状态恢复 fails=0, opened_until=0, 并获得有效的 normalized scores

注意: 该脚本直接访问内部函数 `_remote_rerank` 和状态 `_CB_STATE` 仅用于演示/调试, 生产中请通过 `rerank_post_score` 统一调用。
"""
from __future__ import annotations
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from graphrag.config.settings import get_settings
from graphrag.retriever import rerank as rr

# 为演示可调小冷却期
settings = get_settings()

# 如果用户想独立控制, 可在运行前导出环境变量再执行
if settings.rerank_cb_cooldown > 15:
    print(f"[INFO] 建议将 RERANK_CB_COOLDOWN 调小 (当前 {settings.rerank_cb_cooldown}) 以加速演示。")

# Step 1: 指向错误端点
settings.rerank_endpoint = "http://127.0.0.1:9999/nope"  # 不存在的端口, 保证失败

# 构造伪候选
candidates = [{"id": f"c{i}", "text": f"This is candidate {i}"} for i in range(5)]
question = "demo question"

print("=== 阶段1: 触发连续失败直到熔断 ===")
while True:
    try:
        rr._remote_rerank(question, candidates)  # type: ignore
    except Exception as e:
        state = rr._CB_STATE
        print(f"调用失败: fails={state['fails']} opened_until={state['opened_until']:.0f} err={e}")
        now = time.time()
        if state['opened_until'] > now:
            print("-> 熔断已打开, 后续在冷却期内的调用将直接 circuit_open")
            break
    time.sleep(0.5)

print("尝试在冷却期内再次调用一次:")
try:
    rr._remote_rerank(question, candidates)  # 立即触发 circuit_open
except Exception as e:
    print("冷却期内快速失败:", e)

# Step 2: 等待冷却期过期
cooldown_left = max(0.0, rr._CB_STATE['opened_until'] - time.time())
print(f"等待冷却剩余 {cooldown_left:.1f}s ...")
while time.time() < rr._CB_STATE['opened_until']:
    time.sleep(0.5)
print("冷却结束, 准备恢复")

# Step 3: 启动 mock server
class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        import json
        length = int(self.headers.get('Content-Length','0'))
        body = self.rfile.read(length)
        try:
            data = json.loads(body or b'{}')
            cands = data.get('candidates') or []
        except Exception:
            cands = []
        scores = [i/(len(cands) or 1) for i,_ in enumerate(cands)]
        resp = json.dumps({"scores": scores}).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length', str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
    def log_message(self, format, *args):  # noqa: A003
        return  # 静默

server_port = 8765
srv = HTTPServer(('127.0.0.1', server_port), MockHandler)
th = threading.Thread(target=srv.serve_forever, daemon=True)
th.start()
print(f"Mock rerank server started at http://127.0.0.1:{server_port}")

# 切换到正确端点
settings.rerank_endpoint = f"http://127.0.0.1:{server_port}/rerank"

print("=== 阶段2: 冷却后恢复调用 ===")
try:
    scores = rr._remote_rerank(question, candidates)
    print("成功恢复, scores=", scores)
    print("当前熔断状态:", rr._CB_STATE)
except Exception as e:
    print("恢复调用异常:", e)

print("演示结束。按需 Ctrl+C 结束 (mock server 守护线程会随进程退出)。")
