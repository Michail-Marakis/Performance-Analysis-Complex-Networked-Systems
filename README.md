# Federated Learning Performance Analysis (Flower + PyTorch)

This project implements and analyzes a Federated Learning system using the **Flower Framework** and **PyTorch**, comparing Full Fine-Tuning (NoLoRA) and Low-Rank Adaptation (LoRA) under both Local and Distributed settings.

---

## Overview

Federated Learning enables multiple clients to collaboratively train a global model without sharing raw data.

- Model: TinyBERT  
- Dataset: Rotten Tomatoes sentiment dataset  
- Aggregation: FedAvg  

---

## System Architecture

                          Flower Server
                                │
                                ▼
                       Global TinyBERT Model
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
        Client 1            Client 2            Client N
            │                   │                   │
            ▼                   ▼                   ▼
     Local Dataset      Local Dataset      Local Dataset
            └────────── Updated Weights ───────────┘
                                │
                                ▼
                       FedAvg Aggregation

---

## Federated Learning Workflow

1. Initialize global model
2. Distribute weights to clients
3. Local training on each client
4. Send updates back to server
5. Apply FedAvg aggregation
6. Repeat for multiple rounds

---

## Code Structure

```text
data/
└── Rotten Tomatoes dataset

src/
├── lora/
│   ├── pyproject.toml
│   └── my_awesome_app/
│       ├── server.py
│       ├── client.py
│       ├── strategy.py
│       └── task.py
│
└── nolora/
    ├── pyproject.toml
    ├── my_awesome_app/
    │   ├── server.py
    │   ├── client.py
    │   ├── strategy.py
    │   └── task.py
    │
    └── example_results/
        └── Demonstration of results obtained with the selected hyperparameters
```

### Files

- `server.py` → federation orchestration, aggregation, and evaluation
- `client.py` → local training and metrics collection
- `task.py` → TinyBERT pipeline, dataset handling, and training logic
- `strategy.py` → custom FedAvg strategy, logging, and performance tracking
- `pyproject.toml` → experiment configuration and main hyperparameter settings

### Dataset

The **Rotten Tomatoes** sentiment analysis dataset is stored in the `data/` directory and is shared by both implementations (`lora` and `nolora`).

### Results

The `nolora/example_results/` directory contains a demonstration of the results obtained using the selected hyperparameters.

---

## NoLoRA vs LoRA

### NoLoRA (Full Fine-Tuning)

- All model parameters are trained

Pros:
- Higher flexibility
- Potentially higher accuracy

Cons:
- High communication cost
- High memory usage
- Slower training

---

### LoRA (Low-Rank Adaptation)

W = W₀ + BA

- W₀: frozen pretrained weights  
- A, B: trainable low-rank matrices  

Pros:
- Extremely low communication overhead
- Faster training
- Better scalability

Cons:
- Slight accuracy trade-off

---

## Experimental Configurations

Different configurations are used for Local vs Distributed setups due to hardware constraints.

---

### Local Setup (Simulation)

- Clients: 10  
- num-partitions: 10  
- batch size: 16  
- local epochs: 10  
- Hardware: RTX 4060 (8GB VRAM)  
- Mode: Flower simulation  

Goal:
- Maximum parallelism
- Pure training performance (no network effects)
- Scalability evaluation

---

### Distributed Setup (Real Deployment)

- Clients: 2  
- num-partitions: 2  
- batch size: 8 (reduced due to hardware limitations)  
- Server: GTX 1650 (4GB VRAM)  
- Clients: RTX 4060 (8GB VRAM)  
- Communication: ngrok tunnel  

Goal:
- Real-world federated learning behavior
- Network latency and bottleneck analysis
- Hardware-constrained training

---

## Key Differences

| Setting      | Clients | Partitions | Batch Size | Goal |
|-------------|--------|------------|------------|------|
| Local       | 10     | 10         | 16         | Simulation / performance benchmark |
| Distributed | 2      | 2         | 8          | Real-world deployment |

---

## Results Summary

The primary objective of this project is **not to maximize model accuracy**, but to analyze the communication bottlenecks of Federated Learning systems under realistic hardware and network constraints.

All values below correspond to the **final federation round (Round 10)**.

### Local Setup (Simulation)

| Method | Accuracy | Data Transmitted | Training Time |
|----------|----------|------------------|---------------|
| Full Fine-Tuning | 0.57 | 167.32 MB | 4.37 min |
| LoRA | 0.55 | 0.63 MB | 3.12 min |

**Observation**

LoRA reduced transmitted data by approximately **99.62%** (167.32 MB → 0.63 MB) while sacrificing only **3.5% accuracy** (0.57 → 0.55). Additionally, total training time was reduced by approximately **29%**. :contentReference[oaicite:0]{index=0}

---

### Distributed Setup (Real Deployment)

| Method | Accuracy | Data Transmitted | Training Time |
|----------|----------|------------------|---------------|
| Full Fine-Tuning | 0.63 | 340.14 MB | 14.46 min |
| LoRA | 0.59 | 1.42 MB | 10.58 min |

**Observation**

LoRA reduced transmitted data by approximately **99.58%** (340.14 MB → 1.42 MB) while sacrificing only **6.3% accuracy** (0.63 → 0.59). Total training time was reduced by approximately **26.8%**. :contentReference[oaicite:1]{index=1}

---

### Main Conclusion

This project focuses primarily on **communication efficiency rather than absolute model accuracy**.

The results show that:

- LoRA consistently reduces communication overhead by approximately **99.6%**
- Accuracy remains relatively close to Full Fine-Tuning
- Distributed deployments introduce substantial network and synchronization delays
- Communication becomes the dominant bottleneck as systems scale
- Parameter-efficient techniques such as LoRA are essential for practical Federated Learning deployments

Therefore, the most important metric in this study is **bandwidth consumption and system scalability**, not achieving the highest possible classification accuracy on the Rotten Tomatoes dataset.

---

## Network Model

The system is modeled as an M/G/1 queue:
- M: stochastic arrivals  
- G: general service time  
- 1: single aggregation server  

---

## Little’s Law

N = λT

- N: number of pending updates  
- λ: arrival rate  
- T: time in system  

---

## Quantum Federated Learning (QFL)

This project also briefly references **Quantum Federated Learning (QFL)** as an emerging research direction.

Quantum Federated Learning extends traditional federated learning by integrating **quantum computing principles** into the learning pipeline.

- Combines quantum computing with federated learning for distributed training  
- Uses quantum communication (e.g., teleportation) instead of classical transmission  
- Motivated by potential speedups, reduced communication costs, and stronger security  
- Limited today by quantum hardware constraints, noise, and fidelity loss in long-distance networks  
- Represents a future vision of large-scale, secure, and efficient distributed learning systems  

---

## Key Findings

- LoRA reduces communication cost by >99%
- Distributed setups introduce significant network latency
- Communication is the main bottleneck in FL systems
- Hardware constraints strongly affect batch size and scalability
- Simulation results differ significantly from real-world deployment
- QFL represents a promising future research direction for FL systems

---

## Reference

For more detailed theoretical background, experimental results, and extended analysis, see:

`presentation.pdf`

(Contains full Queueing theory, Little's Law, QFL discussion, and detailed experimental results)
