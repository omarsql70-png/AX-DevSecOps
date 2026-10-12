#!/usr/bin/env python3
"""
sim_iot_flows.py: seeded simulator of IoT gateway flow records (CICFlowMeter-style columns).

WHAT THIS IS
  A controlled stand-in for real traffic. It is used to exercise the AX-DevSecOps
  pipeline (leakage-aware splits, shortcut audit, quality gates, seeded defects).
  It does NOT estimate how well any detector works on real IoT traffic.

DESIGN (all parameters are set by the authors; seed 7 was used during development)
  * 8 classes: Benign, DDoS, DoS, Recon, BruteForce, Spoofing, Web, Mirai.
  * Every class is generated as S capture sessions that follow each other in time.
  * Each session draws its own offsets for a few behavioural features (session effect),
    so flows from the same session resemble each other. A random split lets sessions
    leak across train/test; a chronological split does not.
  * Each attack class runs from one attacker host. Some tools take sequential source ports,
    and that host's port window advances from session to session (sequential ephemeral
    allocation over a long capture). Source port therefore tracks capture time within a
    class: an identifier-like shortcut that helps a random split and misleads a
    chronological one. Benign traffic uses random ephemeral ports.
  * Header lengths add 12 bytes of TCP options on a random share of packets, so they are
    not exact multiples of packet counts.
  * Class distributions overlap on purpose (DoS vs DDoS, Web vs Benign, Spoofing vs Benign).

Usage: python sim_iot_flows.py --out sim_iot_flows.csv --n 45000 --seed 7
"""
import argparse
import numpy as np
import pandas as pd

CLASSES = ["Benign", "DDoS", "DoS", "Recon", "BruteForce", "Spoofing", "Web", "Mirai"]
SHARE = [0.40, 0.14, 0.10, 0.09, 0.08, 0.07, 0.06, 0.06]
# per-class parameters: log-duration mean/sd, fwd pkts rate, bwd/fwd ratio, fwd/bwd payload mean,
# SYN rate, RST prob, PSH rate, UDP prob, port set, sequential-source-port prob
P = {
    "Benign":     dict(ld=(-0.2, 1.6), fw=9,  br=0.9, lf=(4.6, 0.9), lb=(5.2, 1.0), syn=1.0, rst=0.05, psh=3.0, udp=0.35, ports=[80, 443, 1883, 8883, 5683, 123, 53], w=[.22, .25, .18, .07, .1, .08, .1], seq=0.0),
    "DDoS":       dict(ld=(-2.6, 1.2), fw=16, br=0.12, lf=(3.4, 0.9), lb=(2.8, 1.0), syn=4.5, rst=0.28, psh=0.8, udp=0.25, ports=[80, 443], w=[.6, .4], seq=0.3),
    "DoS":        dict(ld=(-1.3, 1.3), fw=9, br=0.3, lf=(4.2, 0.9), lb=(3.6, 1.0), syn=2.4, rst=0.18, psh=1.6, udp=0.2, ports=[80, 8080, 443], w=[.6, .2, .2], seq=0.9),
    "Recon":      dict(ld=(-3.0, 0.9), fw=2,  br=0.35, lf=(1.6, 0.9), lb=(1.4, 1.0), syn=1.2, rst=0.55, psh=0.1, udp=0.1, ports=list(range(1, 1025, 41)), w=None, seq=0.9),
    "BruteForce": dict(ld=(0.2, 1.0),  fw=12, br=0.95, lf=(4.0, 0.6), lb=(4.2, 0.7), syn=1.1, rst=0.1, psh=4.0, udp=0.02, ports=[22, 23, 2323, 21], w=[.4, .3, .2, .1], seq=0.7),
    "Spoofing":   dict(ld=(-1.4, 1.3), fw=4,  br=0.5, lf=(3.6, 0.9), lb=(4.0, 1.0), syn=0.6, rst=0.06, psh=1.0, udp=0.7, ports=[53, 67, 80, 443, 123], w=[.35, .15, .15, .15, .2], seq=0.9),
    "Web":        dict(ld=(0.2, 1.4), fw=13, br=1.0, lf=(5.6, 0.8), lb=(6.1, 0.9), syn=1.0, rst=0.05, psh=6.0, udp=0.01, ports=[80, 8080, 443], w=[.55, .25, .2], seq=0.9),
    "Mirai":      dict(ld=(-1.0, 1.1), fw=5,  br=0.3, lf=(3.2, 0.7), lb=(2.8, 0.8), syn=1.6, rst=0.3, psh=0.8, udp=0.15, ports=[23, 2323, 48101, 80], w=[.4, .3, .2, .1], seq=0.9),
}
FEATURES = [
    "Dst Port", "Protocol", "Flow Duration", "Total Fwd Packets", "Total Bwd Packets",
    "Total Length Fwd Packets", "Total Length Bwd Packets", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Mean", "Bwd IAT Mean",
    "SYN Flag Count", "ACK Flag Count", "RST Flag Count", "PSH Flag Count", "FIN Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Init Fwd Win Bytes", "Init Bwd Win Bytes",
    "Active Mean", "Idle Mean", "Fwd Header Length", "Bwd Header Length",
]


def make(n_total=45000, sessions=12, sess_sd=0.50, seed=7, noise=0.6):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2024-03-01 00:00:00")
    # schedule sessions on a shared timeline (class-interleaved)
    plan = [(c, s) for c in CLASSES for s in range(sessions)]
    rng.shuffle(plan)
    slot = {}
    for i, (c, s) in enumerate(plan):
        slot[(c, s)] = t0 + pd.Timedelta(hours=6 * i)
    # keep sessions of a class in increasing order of session id
    for c in CLASSES:
        starts = sorted(slot[(c, s)] for s in range(sessions))
        for s in range(sessions):
            slot[(c, s)] = starts[s]
    for ci, c in enumerate(CLASSES):
        p = P[c]
        n_c = int(round(n_total * SHARE[ci]))
        # each attack class runs from one attacker host whose ephemeral source ports advance
        # monotonically from session to session (sequential port allocation on a long capture)
        port_start = int(rng.integers(32768, 36000)); port_step = int(rng.integers(1800, 2200))
        sizes = rng.multinomial(n_c, rng.dirichlet(np.full(sessions, 6.0)))
        for s in range(sessions):
            m = sizes[s]
            if m == 0:
                continue
            off = rng.normal(0, sess_sd, size=6)  # session effect on duration, rate, payload f/b, syn, win
            dur = np.exp(rng.normal(p["ld"][0] + off[0], p["ld"][1] * noise, m))
            fwd = 1 + rng.poisson(p["fw"] * np.exp(off[1]), m)
            bwd = rng.poisson(np.maximum(fwd * p["br"], 0.05) * np.exp(rng.normal(0, 0.25, m)))
            lf = np.exp(rng.normal(p["lf"][0] + off[2], p["lf"][1] * noise, m))
            lb = np.exp(rng.normal(p["lb"][0] + off[3], p["lb"][1] * noise, m)) * (bwd > 0)
            tf, tb = fwd * lf, bwd * lb
            pk = fwd + bwd
            iat_mean = dur / np.maximum(pk - 1, 1) * np.exp(rng.normal(0, 0.15, m))
            iat_std = iat_mean * np.exp(rng.normal(-0.3, 0.6, m))
            iat_max = iat_mean * (1 + np.exp(rng.normal(0.4, 0.7, m)))
            iat_min = iat_mean * np.exp(rng.normal(-1.6, 0.9, m))
            syn = rng.poisson(p["syn"] * np.exp(off[4]), m)
            ack = np.maximum(pk - syn - rng.poisson(1.0, m), 0)
            rst = (rng.random(m) < p["rst"]).astype(int) + rng.poisson(0.1 * p["rst"], m)
            psh = rng.poisson(p["psh"], m)
            fin = (rng.random(m) < 0.5 * (1 - p["rst"])).astype(int)
            proto = np.where(rng.random(m) < p["udp"], 17, 6)
            w = None if p["w"] is None else np.array(p["w"]) / np.sum(p["w"])
            dport = rng.choice(p["ports"], m, p=w)
            if c == "Benign" or c == "Web":
                dport = np.where(rng.random(m) < 0.08, rng.integers(1024, 65535, m), dport)
            wins = np.array([0, 1024, 5840, 8192, 29200, 64240])
            wprob = np.array([.1, .1, .2, .2, .2, .2]) if c != "DDoS" else np.array([.3, .3, .1, .1, .1, .1])
            iw = rng.choice(wins, m, p=wprob / wprob.sum()) * np.exp(off[5] * 0.3)
            bw = rng.choice(wins, m, p=wprob[::-1] / wprob.sum()) * (bwd > 0)
            # source port: sequential inside a session for some tools, random otherwise
            base = port_start + s * port_step
            seq = rng.random(m) < p["seq"]
            sp = np.where(seq, base + rng.integers(0, 300, m), rng.integers(32768, 61000, m))
            ts = slot[(c, s)] + pd.to_timedelta(np.sort(rng.uniform(0, 5.5 * 3600, m)), unit="s")
            df = pd.DataFrame({
                "Flow ID": [f"10.0.{ci}.{s}-{i}" for i in range(m)],
                "Src IP": [f"10.0.{ci}.{int(rng.integers(2, 250))}" for _ in range(m)],
                "Src Port": sp.astype(float),
                "Timestamp": ts,
                "Dst Port": dport.astype(float), "Protocol": proto.astype(float),
                "Flow Duration": dur, "Total Fwd Packets": fwd, "Total Bwd Packets": bwd,
                "Total Length Fwd Packets": tf, "Total Length Bwd Packets": tb,
                "Fwd Packet Length Mean": lf, "Bwd Packet Length Mean": lb,
                "Fwd Packet Length Std": lf * np.exp(rng.normal(-1.0, 0.5, m)),
                "Bwd Packet Length Std": lb * np.exp(rng.normal(-1.0, 0.5, m)),
                "Flow Bytes/s": (tf + tb) / np.maximum(dur, 1e-6), "Flow Packets/s": pk / np.maximum(dur, 1e-6),
                "Flow IAT Mean": iat_mean, "Flow IAT Std": iat_std, "Flow IAT Max": iat_max, "Flow IAT Min": iat_min,
                "Fwd IAT Mean": iat_mean * np.exp(rng.normal(0.2, 0.2, m)), "Bwd IAT Mean": iat_mean * np.exp(rng.normal(0.3, 0.3, m)),
                "SYN Flag Count": syn, "ACK Flag Count": ack, "RST Flag Count": rst, "PSH Flag Count": psh, "FIN Flag Count": fin,
                "Down/Up Ratio": bwd / np.maximum(fwd, 1), "Average Packet Size": (tf + tb) / np.maximum(pk, 1),
                "Init Fwd Win Bytes": iw, "Init Bwd Win Bytes": bw,
                "Active Mean": dur * np.exp(rng.normal(-0.7, 0.4, m)), "Idle Mean": dur * np.exp(rng.normal(-1.8, 0.8, m)),
                # 20-byte TCP/IP header plus 12 bytes of options on a session-dependent share of packets
                "Fwd Header Length": fwd * 20.0 + 12.0 * rng.binomial(fwd, rng.uniform(0.2, 0.8)),
                "Bwd Header Length": bwd * 20.0 + 12.0 * rng.binomial(bwd, rng.uniform(0.2, 0.8)),
                "Label": c,
            })
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values("Timestamp", kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sim_iot_flows.csv"); ap.add_argument("--n", type=int, default=45000)
    ap.add_argument("--sessions", type=int, default=12); ap.add_argument("--sess-sd", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--noise", type=float, default=0.6)
    a = ap.parse_args()
    d = make(a.n, a.sessions, a.sess_sd, a.seed, a.noise)
    d.to_csv(a.out, index=False)
    print(d.Label.value_counts().to_string()); print(len(d), "flows ->", a.out)
