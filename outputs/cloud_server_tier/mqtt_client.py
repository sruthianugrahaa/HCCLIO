"""mqtt_client.py - Cloud tier MQTT client.

Where MQTT sits in HCCLIO
  * The broker is Mosquitto, a separate program running on the Edge machine
    (10.0.17.25:1883). It only routes messages; it runs no HCCLIO code. The
    laptop connects to it as a client, so it needs no open inbound MQTT port.
  * The IoT device, the Edge program and the Cloud program are all MQTT
    *clients* of that broker. Nobody connects to anybody else directly.
  * Topics:
        hcclio/edge/request             IoT  -> Edge   JPEG + p_i + timings
        hcclio/cloud/request            Edge -> Cloud  (cascade)  or  IoT -> Cloud (direct)
        hcclio/iot/<client>/response    Edge or Cloud -> IoT       final answer + timings
  * Payloads are msgpack dicts; softmax vectors travel as raw float32 bytes.

This client subscribes to hcclio/cloud/request, hands each message to a callback
on its own thread (inference can take a while), and publishes the final answers.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # outputs/ (holds common/)

from common.messages import pack, unpack  # noqa: E402  (msgpack + numpy codec)


class MqttClient:
    def __init__(self, host: str, port: int, client_id: str, qos: int = 1):
        import paho.mqtt.client as mqtt

        self.qos = qos
        self._handlers: dict[str, Callable[[dict], None]] = {}
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
        for topic in self._handlers:  # re-subscribe after a reconnect
            client.subscribe(topic, qos=self.qos)
        self._connected.set()

    def _on_message(self, client, userdata, message):
        handler = self._handlers.get(message.topic)
        if handler:
            threading.Thread(target=handler, args=(unpack(message.payload),), daemon=True).start()

    def subscribe(self, topic: str, handler: Callable[[dict], None]) -> None:
        self._handlers[topic] = handler
        self.client.subscribe(topic, qos=self.qos)

    def publish(self, topic: str, msg: dict) -> None:
        self.client.publish(topic, pack(msg), qos=self.qos)

    def close(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()
