"""e2lm.py - IoT tier: E2LM delay from the IoT device to the Edge and to the Cloud,
and the simulated channel delays the Edge sends back.

The Edge (10.0.17.25) and the Cloud laptop each run the E2LM probe server on
TCP port 9000 (started automatically by run_edge.py / run_cloud.py). This client
sends n small probes over one TCP connection and returns the median round-trip
time in ms, plus the connection set-up time shared over the probes. The server
does a fixed piece of CPU work per probe, so a busy (loaded) tier answers
slower: that is the "background load" the latency gate sees.

After the probes the IoT device sends b"CHAN" to the Edge, which replies with
the Rayleigh delay it simulated for this frame (the model lives on the Edge):
    {"wireless_delay_ms": delta_wl}
The IoT device stores it with delta_E2LM_edge and uses delta_wl + delta_E2LM_edge
in its latency gate.

If a tier cannot be reached, the delay is reported as timeout_s * 1000
(2000 ms by default), which always fails the 500 ms latency gate.

    python e2lm.py                    # probe Edge and Cloud from config/hcclio.yaml
    python e2lm.py 10.0.17.25 9000    # probe one host
"""

from __future__ import annotations

import json
import socket
import statistics
import struct
import time

HDR = struct.Struct("!I")  # 4-byte length prefix, same framing as the server
CHAN = b"CHAN"             # asks the Edge for this frame's simulated channel delays


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("E2LM server closed the connection")
        buf += chunk
    return bytes(buf)


def e2lm_delay_ms(host: str, port: int = 9000, n_probes: int = 3, probe_bytes: int = 1024,
                  timeout_s: float = 2.0) -> float:
    return e2lm_probe(host, port, n_probes, probe_bytes, timeout_s, want_channel=False)[0]


def e2lm_probe(host: str, port: int = 9000, n_probes: int = 3, probe_bytes: int = 1024,
               timeout_s: float = 2.0, want_channel: bool = True) -> tuple[float, dict]:
    """(E2LM delay in ms, channel delays sent back by the server or {})."""
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
            channel = {}
            if want_channel:
                sock.sendall(HDR.pack(len(CHAN)) + CHAN)
                (n,) = HDR.unpack(_recv_exact(sock, HDR.size))
                channel = json.loads(_recv_exact(sock, n) or b"{}")
    except (OSError, ConnectionError, ValueError):
        return timeout_s * 1000.0, {}
    return statistics.median(rtts) + connect_ms / len(rtts), channel


class E2LM:
    """Probes the Edge and the Cloud (settings from config.py)."""

    def __init__(self, cfg):
        self.edge_addr = (cfg.edge_host, cfg.edge_e2lm_port)
        self.cloud_addr = (cfg.cloud_host, cfg.cloud_e2lm_port)
        self.kw = dict(n_probes=cfg.e2lm_n_probes, probe_bytes=cfg.e2lm_probe_bytes, timeout_s=cfg.e2lm_timeout_s)

    def edge_probe(self) -> tuple[float, dict]:
        """(delta_E2LM_edge, {"wireless_delay_ms": delta_wl} simulated by the Edge)."""
        return e2lm_probe(*self.edge_addr, **self.kw)

    def edge_ms(self) -> float:
        return e2lm_delay_ms(*self.edge_addr, **self.kw)

    def cloud_ms(self) -> float:
        """delta_E2LM_cloud: IoT -> Cloud (used on the direct path when the latency gate fails)."""
        return e2lm_delay_ms(*self.cloud_addr, **self.kw)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        print(f"{e2lm_delay_ms(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 9000):.2f} ms")
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from config import load

        p = E2LM(load())
        ms, chan = p.edge_probe()
        print(f"Edge  {p.edge_addr}:  {ms:.2f} ms   channel from Edge: {chan}")
        print(f"Cloud {p.cloud_addr}: {p.cloud_ms():.2f} ms")
