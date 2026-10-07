"""messages.py - MQTT topic names and the msgpack message format shared by all tiers.

Topics (broker: Mosquitto on the Edge)
    <prefix>/edge/request            IoT  -> Edge
    <prefix>/cloud/request           Edge -> Cloud (cascade) or IoT -> Cloud (direct)
    <prefix>/iot/<client>/response   Edge or Cloud -> IoT

Messages are dicts encoded with msgpack; numpy arrays (softmax vectors) travel
as raw bytes with their dtype and shape.
"""

from __future__ import annotations

import msgpack
import numpy as np


def topics(prefix: str) -> dict:
    return {
        "edge_request": f"{prefix}/edge/request",
        "cloud_request": f"{prefix}/cloud/request",
        "iot_response": f"{prefix}/iot/{{client}}/response",
    }


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
