"""e2lm_server.py - Cloud tier: E2LM probe server (TCP port 9000).

The IoT device (direct path) and the Edge (cascade) probe this server to
measure delta_E2LM_cloud. Each probe packet is echoed after a fixed CPU work
unit, so when the laptop is busy the measured delay grows. A b"CHAN" request
gets {} back: the Cloud simulates no wireless link.

Wire format: 4-byte big-endian length, then the payload.
Open TCP 9000 in Windows Defender Firewall.

    python e2lm_server.py        # run only the probe server
"""

from __future__ import annotations

import socket
import socketserver
import struct
import threading

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


class _ProbeHandler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                (n,) = HDR.unpack(_recv_exact(sock, HDR.size))
                payload = _recv_exact(sock, n)
                if payload == CHAN:
                    reply = b"{}"
                else:
                    _cpu_work(self.server.work_iters)
                    reply = payload
                sock.sendall(HDR.pack(len(reply)) + reply)
        except (ConnectionError, OSError):
            pass


class E2LMServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int = 9000, work_iters: int = 20000):
        super().__init__(("0.0.0.0", port), _ProbeHandler)
        self.work_iters = work_iters

    def start_background(self):
        threading.Thread(target=self.serve_forever, name="e2lm", daemon=True).start()
        return self


if __name__ == "__main__":
    from config import load

    cfg = load()
    print(f"E2LM server on :{cfg.e2lm_port}")
    E2LMServer(cfg.e2lm_port, cfg.e2lm_work_iters).serve_forever()
