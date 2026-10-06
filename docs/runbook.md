# HCCLIO runbook: from scratch to results

Three machines on the same network:

| Tier | Machine | Address used in the config |
|---|---|---|
| IoT | Raspberry Pi 4 | `mypi4.local` |
| Edge | Ubuntu laptop (runs the MQTT broker) | `10.0.17.25` |
| Cloud | Windows laptop | its IP, which you look up in step 0 |

Do the steps in order. Each machine needs internet the first time, to install packages and download its own ViT model.

---

## Step 0: find the IP addresses

- **Ubuntu laptop (Edge):** it is `10.0.17.25` when plugged into the H3C testbed L2 switch, and `10.0.51.158` on Wi-Fi. Run the experiments on the switch: Wi-Fi changes the measured E2LM delays from run to run, and the Pi and the Windows laptop can only reach the Edge on an address of the network they are on. So plug all three machines into the switch, or else put all three on Wi-Fi and set `edge_host` and `broker_host` to `10.0.51.158`.
- **Windows laptop:** run `ipconfig` in PowerShell and note the "IPv4 Address" of the Wi-Fi or Ethernet adapter, for example `10.0.17.30`.
- **Pi 4:** run `hostname -I`.
- **Check:** from each machine, `ping` the other two. All three must reach each other.

---

## Step 1: get the code onto each machine (only its own folders)

The code is on the branch `claude/hcclio-testbed-coawxa` of https://github.com/sruthianugrahaa/HCCLIO. The repository is private, so git asks for your GitHub username and a personal access token.

**Pi 4:**
```bash
git clone -b claude/hcclio-testbed-coawxa --filter=blob:none --sparse https://github.com/sruthianugrahaa/HCCLIO ~/HCCLIO
cd ~/HCCLIO
git sparse-checkout set outputs/common outputs/iot_device_tier outputs/benchmark_strategies outputs/logs
```

**Ubuntu laptop:**
```bash
git clone -b claude/hcclio-testbed-coawxa --filter=blob:none --sparse https://github.com/sruthianugrahaa/HCCLIO ~/HCCLIO
cd ~/HCCLIO
git sparse-checkout set outputs/common outputs/edge_server_tier
```

**Windows laptop (PowerShell):**
```powershell
git clone -b claude/hcclio-testbed-coawxa --filter=blob:none --sparse https://github.com/sruthianugrahaa/HCCLIO C:\HCCLIO
cd C:\HCCLIO
git sparse-checkout set outputs/common outputs/cloud_server_tier outputs/logs outputs/reports
```

`run_mu_sweep.py` and `make_report.py` sit directly in `outputs/`, so they always come along. Every command below is run from the `HCCLIO` folder.

---

## Step 2: edit the one config file (on one machine, then copy it to all three)

Open `outputs/common/hcclio.yaml` and set:

```yaml
network:
  iot_host: mypi4.local
  edge_host: 10.0.17.25          # Ubuntu laptop IP from step 0
  cloud_host: 10.0.17.30         # Windows laptop IP from step 0   <-- must change
  mqtt:
    broker_host: 10.0.17.25      # same as edge_host
```

Leave `qoe.t_i_ms` for now; you set it in step 5. Every machine must have the identical file. After any change, copy it, for example from the Pi:

```bash
scp outputs/common/hcclio.yaml <user>@10.0.17.25:~/HCCLIO/outputs/common/
```

and on Windows, from PowerShell:

```powershell
scp <user>@mypi4.local:~/HCCLIO/outputs/common/hcclio.yaml C:\HCCLIO\outputs\common\
```

---

## Step 3: install (once per machine)

**Ubuntu laptop (Edge):**
```bash
sudo apt install -y mosquitto python3-pip python3-venv
printf 'listener 1883 0.0.0.0\nallow_anonymous true\n' | sudo tee /etc/mosquitto/conf.d/hcclio.conf
sudo systemctl restart mosquitto
sudo ufw allow 1883/tcp; sudo ufw allow 9000/tcp       # only matters if ufw is enabled
python3 -m venv ~/hcclio-env && source ~/hcclio-env/bin/activate
pip install -r outputs/edge_server_tier/requirements.txt
```

Keep the laptop plugged in, and stop it sleeping when the lid closes: set `HandleLidSwitch=ignore` in `/etc/systemd/logind.conf`, then run `sudo systemctl restart systemd-logind`.

**Windows laptop (Cloud):** install Python 3.10 or newer from python.org (tick "Add to PATH"), then in PowerShell:
```powershell
py -m venv C:\hcclio-env; C:\hcclio-env\Scripts\activate
pip install -r outputs\cloud_server_tier\requirements.txt
```

Then allow inbound TCP 9000. In Windows Defender Firewall go to Advanced settings, then Inbound Rules, then New Rule, then Port, TCP 9000, Allow. Or from an admin PowerShell:
```powershell
New-NetFirewallRule -DisplayName "HCCLIO E2LM" -Direction Inbound -Protocol TCP -LocalPort 9000 -Action Allow
```

**Pi 4 (IoT):** it needs the **64-bit** Raspberry Pi OS. `uname -m` must print `aarch64`; if it prints `armv7l`, reflash with the 64-bit image.
```bash
sudo apt install -y python3-pip python3-venv
python3 -m venv ~/hcclio-env && source ~/hcclio-env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r outputs/iot_device_tier/requirements.txt
```

**Check each tier's settings.** On each machine, run its own `config.py` and confirm the IPs:
```bash
python outputs/iot_device_tier/config.py      # Pi
python outputs/edge_server_tier/config.py     # Ubuntu
python outputs/cloud_server_tier/config.py    # Windows
```

---

## Step 4: build the dataset (Pi, once)

This makes 1000 frames from the 30 cobot classes in `~/testbed/dataset/imagenet_1000/`.

**Option A, from Hugging Face.** ImageNet is gated: log in to huggingface.co, open the `ILSVRC/imagenet-1k` page and accept the licence first. Then:
```bash
huggingface-cli login            # paste a read token from huggingface.co/settings/tokens
python outputs/iot_device_tier/download_dataset.py --source hf
```

**Option B, from an ImageNet val folder you already have** (one sub-folder per synset):
```bash
python outputs/iot_device_tier/download_dataset.py --source dir --dir /path/to/imagenet/val
```

**Check:** `head ~/testbed/dataset/imagenet_1000/manifest.csv` shows rows, and `ls ~/testbed/dataset/imagenet_1000/images | wc -l` prints 1000.

---

## Step 5: measure the Pi 4 and set T_i^comp (Pi, once)

```bash
python outputs/iot_device_tier/inference.py ~/testbed/dataset/imagenet_1000/images/$(ls ~/testbed/dataset/imagenet_1000/images | head -1)
```

This prints the mean ViT-Small time and a `suggested qoe.t_i_ms`. Put that number into `qoe.t_i_ms` in `hcclio.yaml`, then copy the file to the other two machines (step 2).

---

## Step 6: start the servers (every session, in this order)

1. **Ubuntu laptop:** check the broker with `systemctl status mosquitto` (it should say active), then:
   ```bash
   source ~/hcclio-env/bin/activate
   python outputs/edge_server_tier/main.py
   ```
   Wait for the `Edge ready` line. The first start downloads ViT-Base.
2. **Windows laptop:**
   ```powershell
   C:\hcclio-env\Scripts\activate
   python outputs\cloud_server_tier\main.py
   ```
   Wait for the `Cloud ready` line. The first start downloads ViT-Large.
3. **Pi: quick connectivity check.** Both probes should print a few ms; 2000 ms means unreachable, which points to the firewall or a wrong IP.
   ```bash
   python outputs/iot_device_tier/e2lm.py 10.0.17.25
   python outputs/iot_device_tier/e2lm.py <windows-ip>
   ```
4. **Pi: smoke test on 20 frames:**
   ```bash
   python outputs/iot_device_tier/main.py --limit 20
   ```
   Check that `outputs/logs/qoe_coclio.csv` has 20 rows, and that the Edge and Cloud windows logged some frames. The full run in step 7 overwrites it.

Leave the two server windows open for all of step 7.

---

## Step 7: run all five strategies (Pi)

Each command processes the 1000 frames and writes its own CSV in `outputs/logs/`:

```bash
python outputs/iot_device_tier/main.py                            # HCCLIO     -> qoe_coclio.csv
python outputs/benchmark_strategies/local_only/run.py             # Local only -> qoe_local_only.csv
python outputs/benchmark_strategies/edge_only/run.py              # Edge only  -> qoe_edge_only.csv
python outputs/benchmark_strategies/cloud_only/run.py             # Cloud only -> qoe_cloud_only.csv
python outputs/benchmark_strategies/distributed_benchmark/run.py  # DCI        -> qoe_distributed_benchmark.csv
```

Each run overwrites its own CSV, so if one is interrupted, just run that command again.

---

## Step 8: tables and plots (Windows laptop)

Copy the five CSVs from the Pi:
```powershell
scp <pi-user>@mypi4.local:~/HCCLIO/outputs/logs/*.csv C:\HCCLIO\outputs\logs\
```

Then:
```powershell
python outputs\make_report.py                  # -> outputs\reports\summary.csv and summary.md
python outputs\run_mu_sweep.py --tier ES       # QoE vs mu_ES -> outputs\reports\mu_sweep_ES.csv / .png
python outputs\run_mu_sweep.py --tier CS       # QoE vs mu_CS
python outputs\run_mu_sweep.py --tier i        # QoE vs mu_i
```

`summary.md` has accuracy, mean QoE, mean and p95 end-to-end latency, and the share of frames answered at each tier, for every strategy. The `mu_sweep_*.png` files compare all five strategies.

For the four paper plots: send their specifications and they will be added as one script that reads these same CSVs, so the hardware does not need to run again.

---

## If something goes wrong

| Symptom | Likely cause |
|---|---|
| Many `Timeout` rows | The Edge or Cloud `main.py` isn't running, or `broker_host` is wrong |
| E2LM probe prints 2000 ms | Port 9000 is blocked by a firewall, or the IP is wrong |
| Almost every offloaded frame is `Discard` | `qoe.t_i_ms` is too small; redo step 5 |
| `pip install torch` fails on the Pi | 32-bit OS; reflash the 64-bit image |
| Pi can't connect to the broker | Mosquitto is listening only on localhost; redo the `listener 1883 0.0.0.0` line in step 3 |
