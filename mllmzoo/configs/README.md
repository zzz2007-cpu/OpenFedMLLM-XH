# MLLM Config Layout

`mllmzoo/configs` has been reorganized by model family:

- `minicpm/`
- `qwen2vl/`

Each experiment is a single standalone config file in the style:

- `minicpmv-crisismmid-FedAvg.py`
- `minicpmv-crisismmid-FedProx-quick.py`
- `qwen2vl-crisismmid-FedNova.py`
- `qwen2vl-crisismmid-Scaffold-quick.py`
- `...`

All files can be launched directly:

```bash
python mllmzoo/configs/minicpm/minicpmv-crisismmid-FedAvg.py
python mllmzoo/configs/minicpm/minicpmv-crisismmid-FedAvg-quick.py
python mllmzoo/configs/qwen2vl/qwen2vl-crisismmid-FedProx.py
python mllmzoo/configs/qwen2vl/qwen2vl-crisismmid-FedProx-quick.py
```

MiniCPM CrisisMMD also has a unified launcher with runtime-selectable algorithm and Dirichlet alpha:

```bash
python mllmzoo/configs/minicpm/run_minicpmv_crisismmid.py --algorithm fedavg --alpha 0.5
python mllmzoo/configs/minicpm/run_minicpmv_crisismmid.py --algorithm fedprox --alpha 0.1
python mllmzoo/configs/minicpm/run_minicpmv_crisismmid.py --algorithm fednova --alpha 1.0 --num-rounds 10 --dry-run
```
