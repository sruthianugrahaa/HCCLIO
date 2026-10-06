# HCCLIO — Collaborative Confidence–Latency Inference Offloading

Three-tier testbed (IoT → Edge → Cloud) for cobot-assisted IoT. Each frame is classified on the
IoT device and offloaded only when a **confidence gate** fails, routed by a **latency gate**, and
re-decided at each higher tier by a **weighted collaborative ensemble**. Every frame is logged with
its end-to-end latency and Quality of Experience so a Monte-Carlo sweep of QoE vs processing rate
μ can be run on the recorded data.

| Tier  | Machine                      | Model (timm, `augreg_in21k`)        | Acc. A_x |
|-------|------------------------------|-------------------------------------|----------|
| IoT   | Raspberry Pi 4 `mypi4.local` | ViT-Small/16 → p_i, c_i, y_i        | 0.75     |
| Edge  | Ubuntu laptop `10.0.17.25` + Mosquitto | ViT-Base/16 → p_ES, c_ES, y_ES   | 0.80     |
| Cloud | Windows laptop               | ViT-Large/16 → p_CS, c_CS, y_CS     | 0.85     |

Transport is MQTT (broker on the Edge) carrying msgpack messages with the JPEG and the softmax
vectors. Each server tier also runs the **E2LM** TCP probe server on port 9000.

## Layout

Each machine holds only `outputs/common/` (60 KB, the same on all three) plus its own tier folder.

```
outputs/
  common/                   COPY TO ALL THREE MACHINES (identical everywhere)
    hcclio.yaml             every parameter: gates, weights, IPs, channel models, QoE, sweep
    settings.py             loads + validates hcclio.yaml (weights sum to 1, direct-to-Cloud weights)
    messages.py             MQTT topic names + msgpack encoding of the messages
    classes.py              the 30 cobot ImageNet classes
    stub_vit.py             simulated ViT for dry runs (--backend stub)
  iot_device_tier/          PI 4 ONLY, one file per step:
    main.py                 Algorithm 1 on the Pi: confidence gate, latency gate, offloading;
                            --strategy hcclio|local_only|edge_only|cloud_only|dci
    config.py               IoT settings from common/hcclio.yaml
    load_dataset.py         frames from ~/testbed/dataset/imagenet_1000 (replaces the camera)
    inference.py            ViT-Small/16 -> p_i, c_i, y_i, time
    e2lm.py                 E2LM delay IoT -> Edge / Cloud; receives the Edge's Rayleigh delay
    mqtt_client.py          publishes frames, waits for the answer on the IoT reply topic
    qoe.py                  Q_x per frame + mu_i / mu_ES / mu_CS helpers (docs/mu_model.md)
    results.py              per-frame CSV (outputs/logs/qoe_coclio.csv)
    download_dataset.py     builds the 1000-frame dataset
    requirements.txt
  benchmark_strategies/     PI 4 ONLY: one-line wrappers that run iot_device_tier/main.py --strategy ...
    edge_only/run.py  cloud_only/run.py  local_only/run.py  distributed_benchmark/run.py (DCI)
  edge_server_tier/         EDGE (UBUNTU LAPTOP) ONLY:
    main.py                 receives frames, ViT-Base, ensemble, answer or cascade to the Cloud
    config.py               Edge settings from common/hcclio.yaml
    models.py               loads ViT-Base/16
    inference.py            ViT-Base + 0.44/0.56 ensemble + confidence gate
    e2lm.py                 E2LM probe server :9000 (answers with this frame's Rayleigh delay)
    edge_rayleigh_delay.py  simulated IoT <-> Edge Rayleigh delay (sent to the IoT device)
    mqtt_client.py          MQTT client of the Mosquitto broker running on this machine
    requirements.txt
  cloud_server_tier/        LAPTOP (WINDOWS) ONLY:
    main.py                 receives frames, adds the backhaul delay, ViT-Large, final answer
    config.py               Cloud settings from common/hcclio.yaml
    model.py                loads ViT-Large/16
    inference.py            ViT-Large + weighted ensemble + Cloud / Fallback-IoT decision
    e2lm_server.py          E2LM probe server :9000
    backhaul_delay.py       simulated Edge <-> Cloud Gamma backhaul delay
    mqtt_client.py          MQTT client of the broker on the Edge
    requirements.txt        also matplotlib for the plots
  run_mu_sweep.py           LAPTOP: Monte-Carlo QoE vs mu_i / mu_ES / mu_CS on the logged CSVs
  make_report.py            LAPTOP: accuracy / QoE / latency / tier-share table
  logs/                     qoe_coclio.csv + one qoe_<benchmark>.csv per benchmark (written on the Pi)
  reports/                  summary.{csv,md}, mu_sweep_<tier>.{csv,png} (written on the laptop)
scripts/loopback.py         development only: all tiers on one computer with simulated ViTs
tests/                      development only: pytest suite
docs/runbook.md             step-by-step: from scratch to results on the three machines
```

## What goes on which machine

| Machine | Copy these folders (keep them side by side under one `outputs/`) |
|---|---|
| Raspberry Pi 4 | `common/`, `iot_device_tier/`, `benchmark_strategies/`, `logs/` |
| Ubuntu laptop (Edge) | `common/`, `edge_server_tier/` |
| Windows laptop | `common/`, `cloud_server_tier/`, `run_mu_sweep.py`, `make_report.py`, `logs/`, `reports/` |

Each tier finds `common/` as its sibling folder, so the only rule is: `common/` and the tier folder
must sit in the same parent folder. For example on the Pi, from a clone or a zip of the repo:

```bash
mkdir -p ~/hcclio/outputs && cd HCCLIO/outputs
cp -r common iot_device_tier benchmark_strategies logs ~/hcclio/outputs/
```

or, without copying by hand, a sparse git checkout of just those folders:

```bash
git clone --filter=blob:none --sparse https://github.com/sruthianugrahaa/HCCLIO && cd HCCLIO
git sparse-checkout set outputs/common outputs/iot_device_tier outputs/benchmark_strategies outputs/logs
# Edge:   git sparse-checkout set outputs/common outputs/edge_server_tier
# Laptop: git sparse-checkout set outputs/common outputs/cloud_server_tier outputs/logs outputs/reports
#         (run_mu_sweep.py and make_report.py are files in outputs/ and are always checked out)
```

When you change a parameter, edit `common/hcclio.yaml` and copy that one file to all three machines.

## Setup

On each machine (Python ≥ 3.9) install only that tier's packages:

```bash
pip install -r outputs/iot_device_tier/requirements.txt     # Pi 4: 64-bit OS + CPU torch first (see the file)
pip install -r outputs/edge_server_tier/requirements.txt    # Edge
pip install -r outputs/cloud_server_tier/requirements.txt   # laptop
```

Set at least `network.cloud_host` (the laptop's IP) in `outputs/common/hcclio.yaml`.

**Edge (Ubuntu laptop)**: Mosquitto 2.x only listens on localhost by default:

```bash
sudo apt install mosquitto
printf 'listener 1883 0.0.0.0\nallow_anonymous true\n' | sudo tee /etc/mosquitto/conf.d/hcclio.conf
sudo systemctl restart mosquitto
sudo ufw allow 1883/tcp && sudo ufw allow 9000/tcp     # only if ufw is enabled
```

Because the Edge is a laptop, keep it on mains power and stop it suspending when the lid closes
(`HandleLidSwitch=ignore` in `/etc/systemd/logind.conf`), or the broker disappears mid-run.

**Pi 4**: use the 64-bit Raspberry Pi OS (`uname -m` must print `aarch64`); PyTorch has no wheels
for the 32-bit OS. Before the first run, time ViT-Small on the Pi and put the suggested value in
`qoe.t_i_ms` (see T_i^comp below):

```bash
python outputs/iot_device_tier/inference.py ~/testbed/dataset/imagenet_1000/images/<any>.jpg
```

**Cloud (Windows)**: allow inbound TCP 9000 (E2LM) in Windows Defender Firewall.

**Dataset (Pi)**: 1000 frames from the 30 cobot classes (`common/classes.py`), balanced 33–34 per
class, written to `~/testbed/dataset/imagenet_1000/` with a `manifest.csv`:

```bash
huggingface-cli login     # ILSVRC/imagenet-1k is gated: accept its licence on huggingface.co first
python outputs/iot_device_tier/download_dataset.py --source hf
# or from a local ImageNet split laid out by synset:
python outputs/iot_device_tier/download_dataset.py --source dir --dir /data/imagenet/val
```

## Running

Start the servers, then the IoT device:

```bash
# Edge  (Ubuntu laptop, 10.0.17.25)
python outputs/edge_server_tier/main.py
# Cloud (Windows laptop)
python outputs/cloud_server_tier/main.py
# IoT   (Pi 4)
python outputs/iot_device_tier/main.py                            # HCCLIO -> outputs/logs/qoe_coclio.csv
python outputs/benchmark_strategies/local_only/run.py             # = main.py --strategy local_only
python outputs/benchmark_strategies/edge_only/run.py
python outputs/benchmark_strategies/cloud_only/run.py
python outputs/benchmark_strategies/distributed_benchmark/run.py  # DCI -> qoe_distributed_benchmark.csv
# laptop, after copying the Pi's outputs/logs/*.csv into the laptop's outputs/logs/
python outputs/make_report.py
python outputs/run_mu_sweep.py --tier ES  # QoE vs mu_ES -> outputs/reports/mu_sweep_ES.{csv,png}
```

`--limit N` runs the first N frames; `--backend stub` swaps the ViTs for seeded simulated
classifiers (no torch needed). `python outputs/iot_device_tier/e2lm.py 10.0.17.25` checks a probe server.

Dry run of the whole pipeline on one computer (needs a local mosquitto on 127.0.0.1:1883):

```bash
pip install -r requirements-dev.txt
python scripts/loopback.py --frames 100
python -m pytest
```

## Algorithm (as implemented)

For each frame x, the IoT runs ViT-Small → (p_i, c_i, y_i).

1. **c_i ≥ τ_conf (0.80)** → tier `IoT`, result y_i.
2. Otherwise probe the Edge: it returns the measured δ_E2LM,edge and the Rayleigh delay δ_wl it drew for this frame.
3. **δ_wl + δ_E2LM,edge ≤ τ_lat (500 ms)** → send JPEG + p_i to the Edge. ViT-Base gives p_ES;
   p_ens = 0.44 p_i + 0.56 p_ES. If max p_ens ≥ τ_conf → tier `Edge`. Otherwise the Edge probes
   the Cloud (E2LM → δ_E2LM,cloud) and cascades JPEG + p_i + p_ES; the Cloud draws the Gamma backhaul delay.
   ViT-Large gives p_CS; p_ens = 0.28 p_i + 0.34 p_ES + 0.38 p_CS. If max ≥ τ_conf → `Cloud`,
   else `Fallback-IoT` with y_i.
4. **Latency gate fails** → straight to the Cloud (Rayleigh hop to the Edge AP + Gamma backhaul +
   δ_E2LM,cloud probed from the IoT). p_ens = w_i p_i + w_CS p_CS with the cloud weights
   renormalised to sum to 1 (0.424 / 0.576); same Cloud / Fallback-IoT decision.

**E2E latency** is the sum of the logged components: measured inference times at each tier, the
measured E2LM probe delays, and the simulated Rayleigh and backhaul delays.

**QoE** (Ernest et al.): Q_x = (1 − T_x^E2E / T_i^comp) · A_x, and `Discard` with Q_x = 0 when
T_x^E2E ≥ T_i^comp. A_x is the pretrained accuracy of the tier that answered (Fallback-IoT uses A_i).

## Channel models

**IoT ↔ Edge (Rayleigh).** |h|² ~ Exp(1), SNR = P_t|h|² / (N_0 B PL), R = B log₂(1+SNR),
delay = 1 Mb / R. PL is 3GPP UMi-like, LOS/NLOS drawn per frame with
P_LOS(d) = exp(d·ln(1−ρ_c)/d_c). f_c = 3.5 GHz, B = 20 MHz, d = 10 m, P_t = 23 dBm,
N_0 = −174 dBm/Hz.

**Edge ↔ Cloud (Gamma backhaul).** T_bh ~ Gamma(m_bh, θ_bh) with
m_bh = ⌊(1 + 1.28·M_BS/M_GW)·k_1 + (h−1)·k_2⌋ and θ_bh = a + K_size·k_3.

## Placeholder values (not given in the spec — tune in `outputs/common/hcclio.yaml`)

| Parameter | Value | Note |
|---|---|---|
| ρ_c, d_c | 0.5, 18 m | P_LOS(10 m) ≈ 0.68 |
| noise figure | 0 dB | spec SNR has no NF term |
| M_BS/M_GW, h | 4, 3 hops | |
| k_1, k_2 | 1.5, 1.0 | → m_bh = 11 |
| a, k_3 | 0.2 ms, 1.5e-6 ms/bit | K_size = 1 Mb → θ_bh = 1.7 ms, mean T_bh ≈ 18.7 ms |
| T_i^comp | 1000 ms (`qoe.t_i_mode: fixed`) | replace with the Pi 4 measurement, see below |
| cloud_host | 10.0.51.160 | Windows laptop |

**T_i^comp.** Every offloaded frame's E2E latency already contains the IoT's own inference time,
so using the per-frame measured IoT time as T_i^comp would discard every frame. T_i^comp is
therefore one reference per run: a fixed value (`t_i_mode: fixed`), or `t_i_mode: calibrated`,
which sets it to `calib_scale` × the mean ViT-Small time over the first `calib_frames` frames on the
Pi. It is logged in every row (`T_i_ms`) and `run_mu_sweep.py --t-i-ms` re-scores the logs with
another value. Prefer one fixed value measured once on the Pi 4 (`inference.py` above prints it):
`calibrated` re-measures in every run, so each strategy would be scored against a slightly
different T_i^comp. The 1000 ms placeholder must stay well above the Pi 4's ViT-Small time, because
every offloaded frame's E2E latency includes that local inference.

**Model heads.** The default checkpoints are `*_patch16_224.augreg_in21k_ft_in1k`: ImageNet-21k
pre-training with the ImageNet-1k head, so all three softmax vectors share the dataset's 1000-class
label space and can be ensembled. The raw `augreg_in21k` checkpoints have a 21,843-class head that
does not match ImageNet-1k indices. `models.restrict_to_dataset_classes: true` renormalises the
softmax over the 30 cobot classes only.

**DCI.** `dci.use_ensemble: true` keeps the same weighted ensembles, so the only difference from
HCCLIO is the missing latency gate. Set it to `false` for each tier to decide on its own model.

## μ sweep

`run_mu_sweep.py --tier i|ES|CS` keeps each frame's logged route and prediction and replaces that
tier's inference time with an M/M/1 system time Exp(mean 1/μ) by default, where μ is the service rate
1/μ_x and μ_x = C_x K_task / f_x. `--service-model mm1` adds M/M/1 queueing,
Exp(mean 1/(μ − P_x λ)) with P_x the share of frames reaching the tier (unstable queue → Discard),
and `det` uses a fixed 1/μ. See `docs/mu_model.md`. E2E latency
and Q_x are recomputed per trial; the output is mean Q_x with a 95 % CI, discard rate and mean E2E
per μ and per strategy.

## CSV columns

`frame_id, timestamp, image_file, ground_truth_idx, ground_truth_name, tier, route, path,
prediction_idx, class_name, correct, iot_confidence, aggregated_conf, local_inference_ms,
wireless_delay_ms, E2LM_edge_ms, edge_inference_ms, backhaul_delay_ms, E2LM_cloud_ms,
cloud_inference_ms, e2e_latency_ms, A_x, Q_x, tau_conf, tau_lat_ms, T_i_ms`

`tier` ∈ {IoT, Edge, Cloud, Fallback-IoT, Discard, Timeout}; `route` records each gate outcome
(e.g. `iot_conf_fail|lat_pass|edge_conf_fail|cloud_conf_pass`); `path` the hops
(e.g. `IoT->Edge->Cloud`). Components a frame did not use are left empty.
