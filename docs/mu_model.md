# Processing rates μ_i, μ_ES, μ_CS in HCCLIO

This note carries the cobot-tier M/M/1 model from your earlier work over to all three HCCLIO tiers.
In the formulas, x stands for i (IoT, Pi 5), ES (Edge) or CS (Cloud).

## 1. The model

| Quantity | Formula | In HCCLIO |
|---|---|---|
| Mean service (inference) time | μ_x = C_x K_task / f_x  [s] | the time one ViT forward pass takes on tier x |
| Computation complexity | C_x [ops/bit] | ViT multiply-accumulates per image ÷ K_task: ViT-S 4.6 G, ViT-B 17.6 G, ViT-L 61.6 G |
| Task size | K_task [bit] | 1 Mb (the same payload as the Rayleigh link) |
| Computational rate | f_x [ops/s] | the effective speed of the Pi 5, the Edge server or the laptop |
| Arrivals at tier x | P_x λ | λ = frames/s generated at the IoT; P_i = 1, while P_ES and P_CS are the shares of frames that the gates send to the Edge or the Cloud |
| Mean system time | τ̄_sys,x = (μ_x⁻¹ − P_x λ)⁻¹ | queueing + service; the queue is stable only if μ_x⁻¹ > P_x λ |
| Tier delay | τ_x = τ_tx,x + τ_sys,x,  τ_sys,x ~ Exp(mean τ̄_sys,x) | τ_tx: Rayleigh delay (IoT→ES), plus the Gamma backhaul delay (ES→CS) |

In your paper μ_x is a time, so 1/μ_x is the service rate in frames/s. The sweep script's x-axis is that
rate, 1/μ_x, and its CSV also lists `mean_service_time_ms` = 1000/rate so you can plot against μ_x itself.

## 2. Getting μ_x and f_x from the testbed

1. Run the testbed. The CSV logs the measured `local_inference_ms`, `edge_inference_ms` and
   `cloud_inference_ms`.
2. Take the mean measured time of each tier as μ_x. Then f_x = C_x K_task / μ_x, which works out to
   `qoe.rate_from_measurement(tier, mean_ms)` in `outputs/iot_device_tier/qoe.py`. This gives the real
   operating point of each device.
3. Read P_ES and P_CS from the `path` column: the share of rows that contain "Edge" or "Cloud".
   `run_mu_sweep.py` computes them automatically.

Example from `python outputs/iot_device_tier/qoe.py`: if ViT-S takes 180 ms on the Pi, then μ_i = 0.18 s,
f_i ≈ 2.6·10¹⁰ ops/s and 1/μ_i ≈ 5.6 frames/s. At λ = 1 frame/s, τ̄_sys,i = 220 ms. At λ = 5 frames/s,
τ̄_sys,i = 1.8 s, so the Pi is close to saturation.

## 3. Sweeping μ (QoE vs μ_x plots)

`run_mu_sweep.py --tier i|ES|CS` keeps every logged frame's routing decision and prediction. On each
Monte-Carlo trial, for every frame that used tier x, it swaps the measured inference time for a draw of
τ_sys,x ~ Exp(mean 1/(1/μ_x − P_x λ)). It then recomputes T_x^E2E and Q_x (Discard if T_x^E2E ≥ T_i^comp)
and averages Q_x over frames and trials.

```bash
python outputs/run_mu_sweep.py --tier ES --arrival-rate 1 --mu 2 4 6 8 10 15 20 30 40 60 80 100
python outputs/run_mu_sweep.py --tier i
python outputs/run_mu_sweep.py --tier CS
```

Rates in the list at or below P_x λ make the queue unstable, and those frames count as Discard (Q = 0).
The measured operating point of each tier is 1000 / mean measured ms, so the list should include rates
around that value.

## 4. What the sweep does not change

The gates use confidences, which μ does not affect, so routing stays as logged. The latency gate uses the
Rayleigh and E2LM delays, which are measured or sampled before offloading, so it is not re-evaluated
either. To make the gate depend on μ_ES as well, add τ̄_sys,ES to the gate condition
(δ_wl + δ_E2LM + τ̄_sys,ES ≤ τ_lat). That is a one-line change in `main.py`. Tell me if you want it.
