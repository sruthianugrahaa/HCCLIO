"""MQTT + msgpack transport (and an in-process stand-in for tests)."""

from __future__ import annotations

import queue
import threading
import uuid
from collections import defaultdict
from typing import Callable

import msgpack
import numpy as np

Handler = Callable[[str, dict], None]


# ---------------------------------------------------------------- codec
def _default(obj):
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {"__nd__": True, "dtype": arr.dtype.str, "shape": list(arr.shape), "data": arr.tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot msgpack {type(obj)}")


def _hook(obj):
    if obj.get("__nd__"):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj


def pack(msg: dict) -> bytes:
    return msgpack.packb(msg, default=_default, use_bin_type=True)


def unpack(data: bytes) -> dict:
    return msgpack.unpackb(data, object_hook=_hook, raw=False)


# ------------------------------------------------------------ topics
def topics(prefix: str) -> dict:
    return {
        "edge_request": f"{prefix}/edge/request",
        "cloud_request": f"{prefix}/cloud/request",
        "iot_response": f"{prefix}/iot/{{client}}/response",
    }


# --------------------------------------------------------- transports
class Transport:
    def publish(self, topic: str, msg: dict) -> int:
        """Publish msg; returns the encoded size in bytes."""
        raise NotImplementedError

    def subscribe(self, topic: str, handler: Handler) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MqttTransport(Transport):
    def __init__(self, host: str, port: int = 1883, keepalive: int = 60, qos: int = 1,
                 client_id: str | None = None):
        import paho.mqtt.client as mqtt

        self.qos = qos
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._connected = threading.Event()
        cid = client_id or f"hcclio-{uuid.uuid4().hex[:8]}"
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
        except AttributeError:  # paho-mqtt 1.x
            self.client = mqtt.Client(client_id=cid)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.max_inflight_messages_set(100)
        self.client.connect(host, port, keepalive)
        self.client.loop_start()
        if not self._connected.wait(10):
            raise ConnectionError(f"MQTT broker {host}:{port} not reachable")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        for t in self._handlers:
            client.subscribe(t, qos=self.qos)
        self._connected.set()

    def _on_message(self, client, userdata, m):
        msg = unpack(m.payload)
        for h in self._handlers.get(m.topic, []):
            # Handlers may block (inference), so never run them on paho's network thread.
            threading.Thread(target=h, args=(m.topic, msg), daemon=True).start()

    def publish(self, topic, msg):
        data = pack(msg)
        self.client.publish(topic, data, qos=self.qos)
        return len(data)

    def subscribe(self, topic, handler):
        self._handlers[topic].append(handler)
        self.client.subscribe(topic, qos=self.qos)

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()


class InProcBroker:
    """Minimal in-process pub/sub so all tiers can run in one Python process."""

    def __init__(self):
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            topic, data = self._q.get()
            for h in list(self._handlers.get(topic, [])):
                threading.Thread(target=h, args=(topic, unpack(data)), daemon=True).start()

    def publish(self, topic, data: bytes):
        self._q.put((topic, data))

    def subscribe(self, topic, handler):
        self._handlers[topic].append(handler)


class InProcTransport(Transport):
    def __init__(self, broker: InProcBroker):
        self.broker = broker

    def publish(self, topic, msg):
        data = pack(msg)  # round-trip through msgpack like the real path
        self.broker.publish(topic, data)
        return len(data)

    def subscribe(self, topic, handler):
        self.broker.subscribe(topic, handler)


def make_transport(cfg, client_id: str | None = None, broker: InProcBroker | None = None) -> Transport:
    if broker is not None:
        return InProcTransport(broker)
    m = cfg["network"]["mqtt"]
    return MqttTransport(m["broker_host"], int(m["broker_port"]), int(m["keepalive_s"]), int(m["qos"]), client_id)


class ResponseWaiter:
    """Matches responses on the IoT reply topic to outstanding frame ids."""

    def __init__(self):
        self._events: dict[int, threading.Event] = {}
        self._results: dict[int, dict] = {}
        self._lock = threading.Lock()

    def expect(self, frame_id: int) -> None:
        with self._lock:
            self._events[frame_id] = threading.Event()

    def on_message(self, topic: str, msg: dict) -> None:
        fid = msg.get("frame_id")
        with self._lock:
            ev = self._events.get(fid)
            if ev is None:
                return
            self._results[fid] = msg
        ev.set()

    def wait(self, frame_id: int, timeout: float) -> dict | None:
        ev = self._events[frame_id]
        ok = ev.wait(timeout)
        with self._lock:
            self._events.pop(frame_id, None)
            return self._results.pop(frame_id, None) if ok else None
