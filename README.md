# AX-DevSecOps: simulator, experiment driver and results

Code and data for the paper **"An Agile-Driven DevSecOps Framework for Explainable Anomaly
Detection in IoT Edge Devices"** by Omar Isam Al-Mrayat, Malik Jawarneh and We'am Talafha
(Amman Arab University). Every number in Section V of the paper is reproduced by the commands below.

## Contents
| File | Purpose |
|---|---|
| `sim_iot_flows.py` | Seeded simulator of IoT gateway flow records (CICFlowMeter-style columns) |
| `axd_experiments.py` | Experiment driver: model zoo, TreeSHAP metrics, sprint loop, quality gates, seeded defects |
| `summarize.py` | Turns the raw CSVs into Tables II–V and the threshold-sensitivity table |
| `supp_srcport_check.py` | Supplementary check: RF/XGB with and without source port |
| `results/` | All outputs used in the paper (raw and summarized) |

## Reproduce
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python axd_experiments.py --sim-seeds 7,8,9,10,11 --out results   # about 70 min on 2 CPU cores
python summarize.py results
python supp_srcport_check.py results
```
Run on the real CIC IoT-DIAD 2024 flow files (not part of the paper's results):
```bash
python axd_experiments.py --data /path/to/AD_Flow-based-features --out results_real
```

## Results → paper
| File | Paper |
|---|---|
| `results/table2_zoo.csv` | Table II |
| `results/table3_explanations.csv` | Table III |
| `results/table4_sprints.csv` | Table IV |
| `results/table5_defects.csv` | Table V |
| `results/sensitivity.csv` | Section V-E (thresholds ±10%) |
| `results/supp_srcport_check.csv` | Section V-D (RF source-port check) |
| `results/*_raw.csv` | Per data set / per run values |
| `results/run_info.json` | Library versions, arguments, data-set sizes |

Latency values depend on the machine that runs the script. The results describe the simulator, not real IoT traffic.

## License
MIT (see `LICENSE`).
