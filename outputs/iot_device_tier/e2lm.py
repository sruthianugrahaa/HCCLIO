"""e2lm.py - IoT tier: E2LM delay from the IoT device to the Edge and to the Cloud.

The Edge (10.0.17.25) and the Cloud laptop each run the E2LM probe server on
TCP port 9000 (started automatically by run_edge.py / run_cloud.py). This client
sends n small probes over one TCP connection and returns the median round-trip
time in ms, plus the connection set-up time shared over the probes. The server
does a fixed piece of CPU work per probe, so a busy (loaded) tier answers
slower: that is the "background load" the latency gate sees.

If a tier cannot be reached, the delay is reported as timeout_s * 1000
(2000 ms by default), which always fails the 500 ms latency gate.

    python e2lm.py                    # probe Edge and Cloud from config/hcclio.yaml
    python e2lm.py 10.0.17.25 9000    # probe one host
"""

from __future__ import annotations

import socket
import statistics
import struct
import time

HDR = struct.Struct("!I")  # 4-byte length prefix, same framing as the server


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


class E2LM:
    """Probes the Edge and the Cloud using the network section of config/hcclio.yaml."""

    def __init__(self, network_cfg: dict):
        e = network_cfg["e2lm"]
        self.edge = (network_cfg["edge_host"], int(e.get("edge_port") or e["port"]))
        self.cloud = (network_cfg["cloud_host"], int(e.get("cloud_port") or e["port"]))
        self.kw = dict(n_probes=int(e.get("n_probes", 3)), probe_bytes=int(e.get("probe_bytes", 1024)),
                       timeout_s=float(e.get("timeout_s", 2.0)))

    def edge_ms(self) -> float:
        """delta_E2LM_edge: IoT -> Edge."""
        return e2lm_delay_ms(*self.edge, **self.kw)

    def cloud_ms(self) -> float:
        """delta_E2LM_cloud: IoT -> Cloud (used on the direct path when the latency gate fails)."""
        return e2lm_delay_ms(*self.cloud, **self.kw)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        print(f"{e2lm_delay_ms(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 9000):.2f} ms")
    else:
        import yaml

        cfg = yaml.safe_load(open(Path(__file__).resolve().parents[2] / "config" / "hcclio.yaml"))
        p = E2LM(cfg["network"])
        print(f"Edge  {p.edge}:  {p.edge_ms():.2f} ms")
        print(f"Cloud {p.cloud}: {p.cloud_ms():.2f} ms")
