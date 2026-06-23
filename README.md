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

data/
└── Rotten Tomatoes dataset
src/
├── lora/
│   └── my_awesome_app/
│       ├── server.py
│       ├── client.py
│       ├── strategy.py
│       └── task.py
│
└── nolora/
    ├── my_awesome_app/
    │   ├── server.py
    │   ├── client.py
    │   ├── strategy.py
    │   └── task.py
    │
    └── example_results/
        └── Demonstration of results obtained with the selected hyperparameters
        
- `server.py` → orginisation, aggregation, evaluation  
- `client.py` → local training and metrics  
- `task.py` → TinyBERT pipeline, dataset handling, training logic  
- `strategy.py` → custom FedAvg strategy, logging, performance tracking  

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
