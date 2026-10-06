# HCCLIO project: folder structure and why each part exists

Branch `claude/hcclio-testbed-coawxa` (PR #1). The whole repository is cloned on all three machines.
Each machine runs only its own tier folder. Everything else is shared.

```
HCCLIO/
├── README.md                      setup + run instructions for all three machines
├── requirements.txt               pip packages (numpy, msgpack, paho-mqtt, pyyaml, pillow, torch, timm, …)
├── pyproject.toml                 makes `hcclio/` installable; pytest settings
├── .gitignore                     keeps CSV logs, reports and caches out of git
│
├── config/
│   └── hcclio.yaml                ★ ONE config for all tiers
│
├── outputs/
│   ├── iot_device_tier/           ★ runs on the Raspberry Pi 5
│   │   ├── main.py                Algorithm 1, IoT side (gates + offloading decision)
│   │   ├── config.py              IoT settings read from hcclio.yaml
│   │   ├── load_dataset.py        frames from ~/testbed/dataset/imagenet_1000 (camera replacement)
│   │   ├── inference.py           ViT-Small/16 → p_i, c_i, y_i, time
│   │   ├── e2lm.py                E2LM client: delay to Edge / Cloud + Rayleigh delay from Edge
│   │   ├── mqtt_client.py         sends a frame, waits for its answer
│   │   ├── qoe.py                 Q_x per frame + μ model
│   │   ├── results.py             one CSV row per frame → outputs/logs/qoe_coclio.csv
│   │   ├── download_dataset.py    builds the 1000-frame dataset (run once)
│   │   └── run_iot.py             same algorithm via the library (used for benchmarks)
│   │
│   ├── edge_server_tier/          ★ runs on the Ubuntu Edge server (10.0.17.25, with Mosquitto)
│   │   ├── main.py                receives frames, ViT-Base, ensemble, answer or cascade
│   │   ├── config.py              Edge settings read from hcclio.yaml
│   │   ├── models.py              loads ViT-Base/16
│   │   ├── inference.py           ViT-Base + 0.44/0.56 ensemble + 0.80 gate
│   │   ├── e2lm.py                E2LM probe server :9000 (+ probe to the Cloud)
│   │   ├── edge_rayleigh_delay.py Rayleigh IoT↔Edge delay, sent to the IoT device
│   │   ├── mqtt_client.py         MQTT client of the local broker
│   │   └── run_edge.py            old name, just starts main.py
│   │
│   ├── cloud_server_tier/         ★ runs on the Windows laptop
│   │   ├── main.py                receives frames, backhaul delay, ViT-Large, final answer
│   │   ├── config.py              Cloud settings read from hcclio.yaml
│   │   ├── model.py               loads ViT-Large/16
│   │   ├── inference.py           ViT-Large + 3- or 2-model ensemble, Cloud / Fallback-IoT
│   │   ├── e2lm_server.py         E2LM probe server :9000
│   │   ├── backhaul_delay.py      Gamma Edge↔Cloud backhaul delay
│   │   ├── mqtt_client.py         MQTT client of the broker on the Edge
│   │   └── run_cloud.py           old name, just starts main.py
│   │
│   ├── benchmark_strategies/      ★ comparison baselines, run on the Pi
│   │   ├── local_only/run.py              ViT-Small only, never offloads
│   │   ├── edge_only/run.py               every frame → Edge, no local ViT, no gates
│   │   ├── cloud_only/run.py              every frame → Cloud, no local ViT
│   │   └── distributed_benchmark/run.py   DCI (Zhang et al.): confidence gate only
│   │
│   ├── run_mu_sweep.py            Monte-Carlo QoE vs μ_i / μ_ES / μ_CS on the logged CSVs
│   ├── make_report.py             accuracy / QoE / latency / tier-share table per strategy
│   ├── logs/                      per-frame CSVs (qoe_coclio.csv, qoe_edge_only.csv, …)
│   └── reports/                   summary.csv/.md, mu_sweep_*.csv/.png (the plots)
│
├── hcclio/                        shared library (imported by the tiers, benchmarks, tests)
│   ├── config.py                  loads + validates hcclio.yaml (weights sum to 1, direct weights)
│   ├── classes.py                 the 30 cobot classes ↔ ImageNet-1k indices
│   ├── transport.py               msgpack encoding of messages, MQTT topic names
│   ├── channel.py                 Rayleigh + Gamma models (library copy)
│   ├── e2lm.py                    E2LM client/server (library copy)
│   ├── models.py                  timm ViT wrapper + simulated "stub" ViTs
│   ├── dataset_loader.py          dataset reader + synthetic test images
│   ├── qoe.py                     ensemble, QoE, CSV column list
│   ├── iot.py / servers.py        all five strategies in one place (for the benchmarks)
│   ├── mu_sweep.py, report.py     the analysis behind run_mu_sweep.py / make_report.py
│   └── cli.py                     command-line wrappers
│
├── docs/
│   ├── mu_model.md                how μ_i, μ_ES, μ_CS are defined and swept
│   └── folder_structure.md        this file
│
├── scripts/
│   └── loopback.py                runs all three tiers on ONE computer (dry run, no hardware)
│
└── tests/
    └── test_hcclio.py             23 automatic checks (pytest)
```

Not in git, but needed on the machines:
- `~/testbed/dataset/imagenet_1000/` on the Pi holds `manifest.csv` and `images/*.jpg`, created by `download_dataset.py`.
- Mosquitto is installed on the Edge with `apt`. It's the MQTT broker, a separate program.

---

## Why it is organised this way

**1. One folder per tier, one file per step.** Each machine has a different job: the Pi does local inference and makes the decision, the Edge does ViT-B and the first ensemble, and the laptop does ViT-L and the final answer. Putting each job in its own folder means you copy or run exactly one folder per machine. Within a folder, each step of Algorithm 1 is its own file, so a reviewer can map paper equation → file, e.g. Rayleigh delay → `edge_rayleigh_delay.py`, QoE → `qoe.py`.

**2. One shared config file (`config/hcclio.yaml`).** τ_conf, τ_lat, the weights and the IP addresses must be identical on all three machines. If each tier had its own constants, one typo would make the tiers disagree silently. Each tier's `config.py` just reads its own part of this one file. To change an experiment, edit only the YAML and copy it to every machine.

**3. Each simulated delay lives where its link ends.**
- The Rayleigh IoT↔Edge delay is drawn on the Edge. It reaches the Pi in the E2LM reply, so the Pi has both values it needs for the 500 ms gate (δ_wl + δ_E2LM) before it offloads.
- The Gamma backhaul delay is drawn on the Cloud, for every frame that arrives there, whether it came from the Edge or directly from the Pi.
- The E2LM delays are measured, not simulated. That's why each server tier runs a probe server: a busy machine answers slower, which is the background load the gate should react to.

**4. MQTT through one broker.** Mosquitto on the Edge is the only server; the Pi, the Edge program and the laptop are all clients. This means the laptop never needs an open MQTT port, and nobody needs anyone else's address except the broker's and the E2LM probe servers'. Each tier has its own `mqtt_client.py` so you can see exactly what it sends and receives:

| Topic | From → to | Content |
|---|---|---|
| `hcclio/edge/request` | Pi → Edge | JPEG + p_i + timings |
| `hcclio/cloud/request` | Edge → Cloud (cascade), or Pi → Cloud (latency gate failed) | JPEG + p_i (+ p_ES) + timings |
| `hcclio/iot/pi5-hcclio/response` | Edge or Cloud → Pi | final answer + all timings |

**5. The Pi writes the CSV.** Every answer returns to the Pi with all the timings collected along the way. So one machine computes T^E2E and Q_x and writes one complete row per frame, and there are no clocks on different machines to synchronise.

**6. Benchmarks are separate folders with one script each.** Each baseline gets its own log file (`qoe_edge_only.csv` …), so the comparison plots read straight from `outputs/logs/`. They use the same Edge and Cloud servers, which only need to be started once.

**7. Analysis runs offline on the logs.** `run_mu_sweep.py` and `make_report.py` only read the CSVs, so they run on any computer after you copy `outputs/logs/`. You can re-plot without re-running the hardware.

**8. `hcclio/` library, `tests/`, `scripts/loopback.py`.** The library holds the pieces several places need: the msgpack encoding, the class list and the config validation. It also holds an all-strategies version of the algorithm that the benchmark scripts use. The tests and the loopback script run the whole system on one computer with simulated ViTs, which is how the code was checked without your hardware.

**Duplication to be aware of.** The Rayleigh, Gamma, E2LM and ensemble code exists both in the tier folders (readable, one file per step) and in `hcclio/` (used by the benchmarks). A test checks that the two Rayleigh models give identical delays. If you change a formula, change it in both places, or tell me and I'll make the tier files the only copy.

---

## Run order

1. **Edge:** start Mosquitto, then `python outputs/edge_server_tier/main.py`.
2. **Laptop:** `python outputs/cloud_server_tier/main.py`.
3. **Pi:** `python outputs/iot_device_tier/main.py`, then the four `benchmark_strategies/*/run.py`.
4. **Any computer:** `python outputs/make_report.py` and `python outputs/run_mu_sweep.py --tier i|ES|CS`.
