# HCCLIO — Collaborative Confidence–Latency Inference Offloading

Three-tier testbed (IoT → Edge → Cloud) for cobot-assisted IoT. Each frame is classified on the
IoT device and offloaded only when a **confidence gate** fails, routed by a **latency gate**, and
re-decided at each higher tier by a **weighted collaborative ensemble**. Every frame is logged with
its end-to-end latency and Quality of Experience so a Monte-Carlo sweep of QoE vs processing rate
μ can be run on the recorded data.

| Tier  | Machine                      | Model (timm, `augreg_in21k`)        | Acc. A_x |
|-------|------------------------------|-------------------------------------|----------|
| IoT   | Raspberry Pi 5 `mypi4.local` | ViT-Small/16 → p_i, c_i, y_i        | 0.75     |
| Edge  | Ubuntu `10.0.17.25` + Mosquitto | ViT-Base/16 → p_ES, c_ES, y_ES   | 0.80     |
| Cloud | Windows laptop               | ViT-Large/16 → p_CS, c_CS, y_CS     | 0.85     |

Transport is MQTT (broker on the Edge) carrying msgpack messages with the JPEG and the softmax
vectors. Each server tier also runs the **E2LM** TCP probe server on port 9000.

## Layout

```
config/hcclio.yaml          every parameter, shared by all tiers
hcclio/                     library: channel models, E2LM, transport, models, tiers, QoE, sweep
outputs/
  iot_device_tier/          IoT tier, one file per step:
    main.py                 Algorithm 1 on the Pi: confidence gate, latency gate, offloading
    load_dataset.py         frames from ~/testbed/dataset/imagenet_1000 (replaces the camera)
    inference.py            ViT-Small/16 -> p_i, c_i, y_i, time
    edge_rayleigh_delay.py  simulated IoT -> Edge Rayleigh delay
    e2lm.py                 E2LM delay IoT -> Edge and IoT -> Cloud (TCP :9000)
    qoe.py                  Q_x per frame + M/M/1 mu_i / mu_ES / mu_CS model (docs/mu_model.md)
    results.py              per-frame CSV (outputs/logs/qoe_coclio.csv)
    download_dataset.py     builds the 1000-frame dataset
    run_iot.py              same algorithm via the library; also used by the benchmarks
  edge_server_tier/         run_edge.py
  cloud_server_tier/        run_cloud.py
  benchmark_strategies/
    edge_only/run.py        every frame -> Edge, no local ViT, no gates
    cloud_only/run.py       every frame -> Cloud, no local ViT
    local_only/run.py       ViT-Small only, no offload
    distributed_benchmark/run.py   DCI (Zhang et al.): confidence gate only, no latency gate
  run_mu_sweep.py           Monte-Carlo over mu_ES on the logged CSVs
  make_report.py            accuracy / QoE / latency / tier-share table
  logs/                     qoe_coclio.csv and one qoe_<benchmark>.csv per benchmark
  reports/                  summary.{csv,md}, mu_sweep_ES.{csv,png}
scripts/loopback.py         all tiers on one machine, for checking the pipeline
tests/                      pytest suite
```

## Setup

On every machine (Python ≥ 3.9), clone the repo and:

```bash
pip install -r requirements.txt     # on the Pi: pip install torch --index-url https://download.pytorch.org/whl/cpu first
```

Edit `config/hcclio.yaml` once (at least `network.cloud_host`) and copy the same file everywhere.

**Edge (Ubuntu)** — Mosquitto 2.x only listens on localhost by default:

```bash
sudo apt install mosquitto
printf 'listener 1883 0.0.0.0\nallow_anonymous true\n' | sudo tee /etc/mosquitto/conf.d/hcclio.conf
sudo systemctl restart mosquitto
```

**Cloud (Windows)** — allow inbound TCP 9000 (E2LM) in Windows Defender Firewall.

**Dataset (IoT)** — 1000 frames from the 30 cobot classes (`hcclio/classes.py`), balanced 33–34 per
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
# Edge  (10.0.17.25)
python outputs/edge_server_tier/run_edge.py
# Cloud (laptop)
python outputs/cloud_server_tier/run_cloud.py
# IoT   (Pi 5)
python outputs/iot_device_tier/main.py                            # HCCLIO -> outputs/logs/qoe_coclio.csv
python outputs/benchmark_strategies/local_only/run.py
python outputs/benchmark_strategies/edge_only/run.py
python outputs/benchmark_strategies/cloud_only/run.py
python outputs/benchmark_strategies/distributed_benchmark/run.py
# anywhere, after copying outputs/logs/
python outputs/make_report.py
python outputs/run_mu_sweep.py            # QoE vs mu_ES -> outputs/reports/mu_sweep_ES.{csv,png}
```

`--limit N` runs the first N frames; `--backend stub` swaps the ViTs for seeded simulated
classifiers (no torch needed). `python -m hcclio.e2lm probe 10.0.17.25` checks a probe server.

Dry run of the whole pipeline on one machine (in-process broker, or a local mosquitto):

```bash
python scripts/loopback.py --limit 200
python scripts/loopback.py --mqtt 127.0.0.1:1883
python -m pytest
```

## Algorithm (as implemented)

For each frame x, the IoT runs ViT-Small → (p_i, c_i, y_i).

1. **c_i ≥ τ_conf (0.80)** → tier `IoT`, result y_i.
2. Otherwise probe the Edge (E2LM → δ_E2LM,edge) and sample the Rayleigh uplink delay δ_wl.
3. **δ_wl + δ_E2LM,edge ≤ τ_lat (500 ms)** → send JPEG + p_i to the Edge. ViT-Base gives p_ES;
   p_ens = 0.44 p_i + 0.56 p_ES. If max p_ens ≥ τ_conf → tier `Edge`. Otherwise the Edge probes
   the Cloud (E2LM → δ_E2LM,cloud), samples the Gamma backhaul, and cascades JPEG + p_i + p_ES.
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

## Placeholder values (not given in the spec — tune in `config/hcclio.yaml`)

| Parameter | Value | Note |
|---|---|---|
| ρ_c, d_c | 0.5, 18 m | P_LOS(10 m) ≈ 0.68 |
| noise figure | 0 dB | spec SNR has no NF term |
| M_BS/M_GW, h | 4, 3 hops | |
| k_1, k_2 | 1.5, 1.0 | → m_bh = 11 |
| a, k_3 | 0.2 ms, 1.5e-6 ms/bit | K_size = 1 Mb → θ_bh = 1.7 ms, mean T_bh ≈ 18.7 ms |
| T_i^comp | 1000 ms (`qoe.t_i_mode: fixed`) | see below |
| cloud_host | 10.0.17.30 | set the laptop's IP |

**T_i^comp.** Every offloaded frame's E2E latency already contains the IoT's own inference time,
so using the per-frame measured IoT time as T_i^comp would discard every frame. T_i^comp is
therefore one reference per run: a fixed value (`t_i_mode: fixed`), or `t_i_mode: calibrated`,
which sets it to `calib_scale` × the mean ViT-Small time over the first `calib_frames` frames on the
Pi. It is logged in every row (`T_i_ms`) and `run_mu_sweep.py --t-i-ms` re-scores the logs with
another value.

**Model heads.** The default checkpoints are `*_patch16_224.augreg_in21k_ft_in1k`: ImageNet-21k
pre-training with the ImageNet-1k head, so all three softmax vectors share the dataset's 1000-class
label space and can be ensembled. The raw `augreg_in21k` checkpoints have a 21,843-class head that
does not match ImageNet-1k indices. `models.restrict_to_dataset_classes: true` renormalises the
softmax over the 30 cobot classes only.

**DCI.** `dci.use_ensemble: true` keeps the same weighted ensembles, so the only difference from
HCCLIO is the missing latency gate. Set it to `false` for each tier to decide on its own model.

## μ sweep

`run_mu_sweep.py --tier i|ES|CS` keeps each frame's logged route and prediction and replaces that
tier's inference time with an M/M/1 system time Exp(mean 1/(μ − P_x λ)), where μ is the service rate
1/μ_x, μ_x = C_x K_task / f_x, and P_x is the share of frames reaching the tier (unstable queue →
Discard). `--service-model exp|det` are alternatives. See `docs/mu_model.md`. E2E latency
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
