"""mqtt_client.py - IoT tier MQTT client: send a frame, wait for its answer.

Where MQTT sits in HCCLIO
  * The broker is Mosquitto on the Edge machine (10.0.17.25:1883). It only routes
    messages; the IoT device, the Edge and the Cloud are all MQTT clients of it.
  * The IoT device publishes to
        hcclio/edge/request    (latency gate passed)   JPEG + p_i + timings
        hcclio/cloud/request   (latency gate failed)   JPEG + p_i + timings
    and receives the final answer on its own topic
        hcclio/iot/pi5-hcclio/response
    from whichever tier answered (Edge or Cloud).
  * Payloads are msgpack dicts; p_i travels as raw float32 bytes.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # outputs/ (holds common/)

from common.messages import pack, unpack  # noqa: E402  (msgpack + numpy codec)


class MqttClient:
    def __init__(self, host: str, port: int, client_id: str, reply_topic: str, qos: int = 1):
        import paho.mqtt.client as mqtt

        self.qos = qos
        self.reply_topic = reply_topic
        self._pending: dict[int, threading.Event] = {}
        self._answers: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._connected = threading.Event()
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
        except AttributeError:  # paho-mqtt 1.x
            self.client = mqtt.Client(client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(host, port, keepalive=60)
        self.client.loop_start()
        if not self._connected.wait(10):
            raise ConnectionError(f"cannot reach the MQTT broker at {host}:{port}")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        client.subscribe(self.reply_topic, qos=self.qos)
        self._connected.set()

    def _on_message(self, client, userdata, message):
        answer = unpack(message.payload)
        with self._lock:
            ev = self._pending.get(answer.get("frame_id"))
            if ev is None:
                return  # a late answer for a frame that already timed out
            self._answers[answer["frame_id"]] = answer
        ev.set()

    def request(self, topic: str, msg: dict, timeout_s: float) -> dict | None:
        """Publish msg (must carry frame_id) and block until its answer arrives, or None on timeout."""
        fid = msg["frame_id"]
        ev = threading.Event()
        with self._lock:
            self._pending[fid] = ev
        self.client.publish(topic, pack({**msg, "reply_topic": self.reply_topic}), qos=self.qos)
        ok = ev.wait(timeout_s)
        with self._lock:
            self._pending.pop(fid, None)
            return self._answers.pop(fid, None) if ok else None

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
