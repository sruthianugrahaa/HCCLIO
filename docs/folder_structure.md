# HCCLIO folder structure

Each machine stores only the shared `outputs/common/` folder (about 60 KB) plus the folder for its own tier. Nothing else is needed at run time; the ViT weights are downloaded by timm into its own cache on each machine, so the repository stays small.

## What each machine stores

```
Raspberry Pi 4 (IoT)                 Ubuntu laptop (Edge) 10.0.17.25   Windows laptop (Cloud)
~/hcclio/outputs/                    ~/hcclio/outputs/                 C:\hcclio\outputs\
├── common/                          ├── common/                       ├── common/
├── iot_device_tier/                 └── edge_server_tier/             ├── cloud_server_tier/
├── benchmark_strategies/                                              ├── run_mu_sweep.py
└── logs/        (CSVs written here)                                   ├── make_report.py
                                                                       ├── logs/     (copy of the Pi's CSVs)
                                                                       └── reports/  (tables + plots)
```

The only rule: `common/` must sit next to the tier folder, inside the same `outputs/` folder. Each tier's files look for `../common/` and nothing else.

Two ways to put the folders on a machine:

```bash
# A. copy from a clone or zip of the repo (Pi example)
cp -r outputs/common outputs/iot_device_tier outputs/benchmark_strategies outputs/logs ~/hcclio/outputs/

# B. sparse git checkout: only those folders are downloaded
git clone --filter=blob:none --sparse https://github.com/sruthianugrahaa/HCCLIO && cd HCCLIO
git sparse-checkout set outputs/common outputs/iot_device_tier outputs/benchmark_strategies outputs/logs   # Pi
git sparse-checkout set outputs/common outputs/edge_server_tier                                           # Edge
git sparse-checkout set outputs/common outputs/cloud_server_tier outputs/logs outputs/reports             # laptop
```

Option B lets you `git pull` updates later and still only get your machine's folders.

## Full repository (on GitHub, not on any one machine)

```
HCCLIO/
├── README.md
├── requirements-dev.txt           development only (tests + dry run on one computer)
├── pyproject.toml                 pytest settings
├── outputs/
│   ├── common/                    → ALL THREE MACHINES (identical copy everywhere)
│   │   ├── hcclio.yaml            every parameter: gates, weights, IPs, channel models, QoE, sweep
│   │   ├── settings.py            loads + checks hcclio.yaml (weights sum to 1, direct-to-Cloud weights)
│   │   ├── messages.py            MQTT topic names + msgpack encoding of messages
│   │   ├── classes.py             the 30 cobot ImageNet classes
│   │   └── stub_vit.py            simulated ViT for dry runs (--backend stub)
│   ├── iot_device_tier/           → PI ONLY
│   │   ├── main.py                Algorithm 1: confidence gate 0.80, latency gate 500 ms, offloading;
│   │   │                          --strategy hcclio | local_only | edge_only | cloud_only | dci
│   │   ├── config.py              IoT settings from common/hcclio.yaml
│   │   ├── load_dataset.py        reads ~/testbed/dataset/imagenet_1000 (replaces the camera)
│   │   ├── inference.py           ViT-Small/16 → p_i, c_i, y_i, time
│   │   ├── e2lm.py                E2LM delay to Edge / Cloud; receives the Edge's Rayleigh delay
│   │   ├── mqtt_client.py         sends frames, waits for the answer
│   │   ├── qoe.py                 Q_x per frame, μ_i / μ_ES / μ_CS helpers
│   │   ├── results.py             one CSV row per frame (28 columns)
│   │   ├── download_dataset.py    builds the 1000-frame dataset
│   │   └── requirements.txt
│   ├── benchmark_strategies/      → PI ONLY (each run.py = main.py --strategy …)
│   │   ├── local_only/run.py
│   │   ├── edge_only/run.py
│   │   ├── cloud_only/run.py
│   │   └── distributed_benchmark/run.py    DCI: confidence gate only, no latency gate
│   ├── edge_server_tier/          → EDGE ONLY
│   │   ├── main.py                receive, ViT-Base, ensemble 0.44/0.56, answer or cascade
│   │   ├── config.py
│   │   ├── models.py              loads ViT-Base/16
│   │   ├── inference.py           ViT-Base + ensemble + confidence gate
│   │   ├── e2lm.py                E2LM probe server :9000, answers with this frame's Rayleigh delay
│   │   ├── edge_rayleigh_delay.py IoT ↔ Edge Rayleigh delay, sent to the Pi
│   │   ├── mqtt_client.py
│   │   └── requirements.txt
│   ├── cloud_server_tier/         → LAPTOP ONLY
│   │   ├── main.py                receive, add backhaul delay, ViT-Large, final answer
│   │   ├── config.py
│   │   ├── model.py               loads ViT-Large/16
│   │   ├── inference.py           ViT-Large + ensemble 0.28/0.34/0.38 + Cloud / Fallback-IoT
│   │   ├── e2lm_server.py         E2LM probe server :9000
│   │   ├── backhaul_delay.py      Edge ↔ Cloud Gamma backhaul delay
│   │   ├── mqtt_client.py
│   │   └── requirements.txt       also matplotlib for the plots
│   ├── run_mu_sweep.py            → LAPTOP: QoE vs μ_i / μ_ES / μ_CS from the logs
│   ├── make_report.py             → LAPTOP: accuracy / QoE / latency / tier-share table
│   ├── logs/                      qoe_coclio.csv + qoe_<benchmark>.csv (written on the Pi)
│   └── reports/                   summary.{csv,md}, mu_sweep_<tier>.{csv,png}
├── docs/                          this file, mu_model.md
├── scripts/loopback.py            development only: all tiers on one computer, simulated ViTs
└── tests/test_hcclio.py           development only: pytest suite
```

Not in the repository:
- `~/testbed/dataset/imagenet_1000/` on the Pi holds `manifest.csv` and `images/*.jpg`, created by `download_dataset.py`.
- Mosquitto is installed on the Edge with `apt`. It's the MQTT broker, a separate program.
- ViT weights live in each machine's timm / Hugging Face cache, and each machine downloads only its own model (Small on the Pi, Base on the Edge, Large on the laptop).

---

## Why it is organised this way

**1. One folder per machine, nothing duplicated.** Each tier folder contains only its own step files and imports nothing from the other tiers, so a machine never needs another tier's code. The Pi does not get the Rayleigh or backhaul code, the Edge does not get the dataset or analysis scripts, and so on.

**2. A small `common/` folder for what must be identical.** τ_conf, τ_lat, the weights, the IP addresses and the message format must agree on all three machines. If each tier had its own copy, one typo would make the tiers disagree silently. So `common/` holds just those: the YAML, its loader, the topic names and msgpack encoding, and the class list. To change an experiment, edit `common/hcclio.yaml` and copy that one file to every machine.

**3. Each simulated delay lives where its link ends.**
- The Rayleigh IoT↔Edge delay is drawn on the Edge. It reaches the Pi in the E2LM reply, so the Pi has both values it needs for the 500 ms gate (δ_wl + δ_E2LM) before it offloads.
- The Gamma backhaul delay is drawn on the Cloud, for every frame that arrives there, whether it came from the Edge or directly from the Pi.
- The E2LM delays are measured, not simulated. That's why each server tier runs a probe server: a busy machine answers slower, which is the background load the gate should react to.

**4. MQTT through one broker.** Mosquitto on the Edge is the only server; the Pi, the Edge program and the laptop are all clients. The laptop never needs an open MQTT port, and nobody needs anyone else's address except the broker's and the E2LM probe servers'.

| Topic | From → to | Content |
|---|---|---|
| `hcclio/edge/request` | Pi → Edge | JPEG + p_i + timings |
| `hcclio/cloud/request` | Edge → Cloud (cascade), or Pi → Cloud (latency gate failed) | JPEG + p_i (+ p_ES) + timings |
| `hcclio/iot/pi4-hcclio/response` | Edge or Cloud → Pi | final answer + all timings |

**5. The Pi writes the CSV.** Every answer returns to the Pi with all the timings collected along the way. So one machine computes T^E2E and Q_x and writes one complete row per frame, and there are no clocks on different machines to synchronise.

**6. Benchmarks reuse the Pi's `main.py`.** Each `benchmark_strategies/*/run.py` just runs `iot_device_tier/main.py` with a different `--strategy`, so the benchmarks and HCCLIO share exactly the same code and only the decision rule differs. Each one writes its own log (`qoe_edge_only.csv` …), and they use the same Edge and Cloud servers, which only need to be started once.

**7. Analysis runs on the laptop from the logs.** `run_mu_sweep.py` and `make_report.py` only read the CSVs, so after you copy the Pi's `outputs/logs/*.csv` to the laptop you can re-plot as often as you like without re-running the hardware.

**8. `tests/` and `scripts/loopback.py` stay on GitHub.** They run the whole system on one computer with simulated ViTs, which is how the code was checked without your hardware. None of the three machines needs them.

---

## Run order

1. **Edge:** start Mosquitto, then `python outputs/edge_server_tier/main.py`.
2. **Laptop:** `python outputs/cloud_server_tier/main.py`.
3. **Pi:** `python outputs/iot_device_tier/main.py`, then the four `benchmark_strategies/*/run.py`.
4. **Laptop:** copy the Pi's `outputs/logs/*.csv` into its `outputs/logs/`, then `python outputs/make_report.py` and `python outputs/run_mu_sweep.py --tier i|ES|CS`.
