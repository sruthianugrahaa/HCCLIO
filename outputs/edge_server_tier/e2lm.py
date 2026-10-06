"""e2lm.py - Edge tier: E2LM probe server (TCP port 9000), plus the probe to the Cloud.

Server (the IoT device probes it before every offloading decision):
  * each probe packet is echoed after a fixed CPU work unit, so when the Edge is
    loaded the IoT device measures a larger delay -> delta_E2LM_edge;
  * the IoT device then sends b"CHAN" and the Edge answers with the simulated
    channel delays it drew for this frame:
        {"wireless_delay_ms": Rayleigh IoT<->Edge delay   (edge_rayleigh_delay.py),
         "backhaul_delay_ms": Gamma Edge<->Cloud delay    (backhaul_delay.py)}
    The IoT device stores both and uses delta_wl + delta_E2LM_edge in its latency gate.

Client: before a cascade the Edge probes the Cloud's E2LM server -> delta_E2LM_cloud.

Wire format (both directions): 4-byte big-endian length, then the payload.

    python e2lm.py serve                 # run only the probe server
    python e2lm.py probe 10.0.17.30      # probe the Cloud from the Edge
"""

from __future__ import annotations

import json
import socket
import socketserver
import statistics
import struct
import threading
import time

HDR = struct.Struct("!I")
CHAN = b"CHAN"


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def _cpu_work(iters: int) -> int:
    acc = 0
    for i in range(iters):
        acc = (acc + i * i) % 1000003
    return acc


# ------------------------------------------------------------------ server
class _ProbeHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                (n,) = HDR.unpack(_recv_exact(sock, HDR.size))
                payload = _recv_exact(sock, n)
                if payload == CHAN:
                    reply = json.dumps(self.server.channel_fn()).encode()
                else:
                    _cpu_work(self.server.work_iters)
                    reply = payload
                sock.sendall(HDR.pack(len(reply)) + reply)
        except (ConnectionError, OSError):
            pass


class E2LMServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int, work_iters: int, channel_fn):
        super().__init__(("0.0.0.0", port), _ProbeHandler)
        self.work_iters = work_iters
        self.channel_fn = channel_fn

    def start_background(self):
        threading.Thread(target=self.serve_forever, name="e2lm", daemon=True).start()
        return self


def make_channel_fn(wireless, backhaul):
    """The per-frame channel sample the Edge sends to the IoT device."""
    lock = threading.Lock()  # the random generators are shared between connections

    def channel_fn() -> dict:
        with lock:
            return {"wireless_delay_ms": wireless.delay_ms(), "backhaul_delay_ms": backhaul.delay_ms()}

    return channel_fn


# ------------------------------------------------------------------ client
def probe_ms(host: str, port: int = 9000, n_probes: int = 3, probe_bytes: int = 1024,
             timeout_s: float = 2.0) -> float:
    """Median probe round trip in ms (+ shared connect time); timeout_s*1000 if unreachable."""
    payload = b"\x00" * probe_bytes
    rtts = []
    t0 = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout_s) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connect_ms = (time.perf_counter() - t0) * 1000.0
            for _ in range(max(1, n_probes)):
                t = time.perf_counter()
                sock.sendall(HDR.pack(len(payload)) + payload)
                (n,) = HDR.unpack(_recv_exact(sock, HDR.size))
                _recv_exact(sock, n)
                rtts.append((time.perf_counter() - t) * 1000.0)
    except (OSError, ConnectionError):
        return timeout_s * 1000.0
    return statistics.median(rtts) + connect_ms / len(rtts)


if __name__ == "__main__":
    import sys

    from backhaul_delay import BackhaulDelay
    from config import load
    from edge_rayleigh_delay import EdgeRayleighDelay

    cfg = load()
    if len(sys.argv) > 2 and sys.argv[1] == "probe":
        print(f"{probe_ms(sys.argv[2], cfg.cloud_e2lm_port):.2f} ms")
    else:
        fn = make_channel_fn(EdgeRayleighDelay(cfg.wireless, cfg.seed), BackhaulDelay(cfg.backhaul, cfg.seed + 1))
        print(f"E2LM server on :{cfg.e2lm_port}")
        E2LMServer(cfg.e2lm_port, cfg.e2lm_work_iters, fn).serve_forever()
