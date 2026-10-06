"""E2LM TCP probe (port 9000): measures the delay to reach a tier and get served.

The server echoes each probe after a fixed CPU work unit, so a busy tier
(background load) answers slower. The client reports the median round trip
in ms over n probes. An unreachable tier reports timeout_s * 1000.

After the probes the client may send the payload b"CHAN": the server answers
with a JSON dict of simulated channel delays for this frame (the Edge returns
its Rayleigh wireless delay and Gamma backhaul delay; the Cloud returns {}).

Run a server:  python -m hcclio.e2lm serve [--port 9000]
Probe a host:  python -m hcclio.e2lm probe 10.0.17.25
"""

from __future__ import annotations

import argparse
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
            raise ConnectionError("E2LM peer closed")
        buf += chunk
    return bytes(buf)


def _work(iters: int) -> int:
    acc = 0
    for i in range(iters):
        acc = (acc + i * i) % 1000003
    return acc


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                (n,) = HDR.unpack(_recv_exact(sock, HDR.size))
                payload = _recv_exact(sock, n)
                if payload == CHAN:
                    fn = self.server.channel_fn
                    reply = json.dumps(fn() if fn else {}).encode()
                    sock.sendall(HDR.pack(len(reply)) + reply)
                    continue
                _work(self.server.work_iters)
                sock.sendall(HDR.pack(len(payload)) + payload)
        except (ConnectionError, OSError):
            pass


class E2LMServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host: str = "0.0.0.0", port: int = 9000, work_iters: int = 20000, channel_fn=None):
        super().__init__((host, port), _Handler)
        self.work_iters = work_iters
        self.channel_fn = channel_fn  # () -> dict of simulated delays, answered on b"CHAN"

    def start_background(self) -> threading.Thread:
        t = threading.Thread(target=self.serve_forever, name="e2lm-server", daemon=True)
        t.start()
        return t


def probe(host: str, port: int = 9000, n_probes: int = 3, probe_bytes: int = 1024,
          timeout_s: float = 2.0) -> float:
    """Median TCP request/response delay in ms (connection setup included once)."""
    return probe_with_channel(host, port, n_probes, probe_bytes, timeout_s, want_channel=False)[0]


def probe_with_channel(host: str, port: int = 9000, n_probes: int = 3, probe_bytes: int = 1024,
                       timeout_s: float = 2.0, want_channel: bool = True) -> tuple[float, dict]:
    """(E2LM delay in ms, simulated channel delays the server sent back or {})."""
    payload = b"\x00" * probe_bytes
    samples = []
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
                samples.append((time.perf_counter() - t) * 1000.0)
            chan = {}
            if want_channel:
                sock.sendall(HDR.pack(len(CHAN)) + CHAN)
                (n,) = HDR.unpack(_recv_exact(sock, HDR.size))
                chan = json.loads(_recv_exact(sock, n) or b"{}")
    except (OSError, ConnectionError, ValueError):
        return timeout_s * 1000.0, {}
    # Count the handshake once, spread over the probes, so a congested path shows up.
    return statistics.median(samples) + connect_ms / len(samples), chan


class E2LMClient:
    def __init__(self, e2lm_cfg: dict):
        self.port = int(e2lm_cfg["port"])
        self.n = int(e2lm_cfg.get("n_probes", 3))
        self.nbytes = int(e2lm_cfg.get("probe_bytes", 1024))
        self.timeout = float(e2lm_cfg.get("timeout_s", 2.0))

    def __call__(self, host: str, port: int | None = None) -> float:
        return probe(host, int(port or self.port), self.n, self.nbytes, self.timeout)

    def with_channel(self, host: str, port: int | None = None) -> tuple[float, dict]:
        return probe_with_channel(host, int(port or self.port), self.n, self.nbytes, self.timeout)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=9000)
    s.add_argument("--work-iters", type=int, default=20000)
    p = sub.add_parser("probe")
    p.add_argument("host")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("-n", type=int, default=3)
    args = ap.parse_args(argv)
    if args.cmd == "serve":
        srv = E2LMServer(args.host, args.port, args.work_iters)
        print(f"E2LM probe server on {args.host}:{args.port}")
        srv.serve_forever()
    else:
        print(f"{probe(args.host, args.port, args.n):.3f} ms")


if __name__ == "__main__":
    main()
