"""my-awesome-app: A Flower / PyTorch app."""

import json
from typing import List, Tuple

import torch
from datasets import load_dataset
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader

from my_awesome_app.strategy import CustomFedAvg
from my_awesome_app.task import Net, load_data, get_weights, set_weights, test, get_tokenizer


def get_evaluate_fn(testloader, device):
    """Return a callback that evaluates the global model on a centralized dataset."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model using the provided centralized test set."""
        net = Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)

        return loss, {"cen_accuracy": accuracy}

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from an evaluation round using a weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    return {"accuracy": sum(accuracies) / total_examples}


def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from a training fit round."""
    b_values = []
    for _, m in metrics:
        my_metric_str = m["my_metric"]
        my_metric = json.loads(my_metric_str)
        b_values.append(my_metric["b"])
    return {"max_b": max(b_values)}


def on_fit_config(server_round: int) -> Metrics:
    """Configure training parameters and adjust the learning rate dynamically based on the round."""
    lr = 1e-5
    if server_round > 2:
        lr = 5e-6
    return {"lr": lr}



#RAM-OPTIMIZED GLOBAL LOGIC (COMMENTED OUT)

# print("[INFO] [SERVER] Globally initializing centralized evaluation dataset (rotten_tomatoes)...")
# raw_testset = load_dataset("rotten_tomatoes", verbose=False)["test"]
# tokenizer = get_tokenizer()
#
# def tokenize_function(examples):
#      return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
#
# print("[INFO] [SERVER] Mapping tokenization across centralized test sequences...")
# tokenized_testset = raw_testset.map(tokenize_function, batched=True, verbose=False)
# tokenized_testset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
#
# testloader = DataLoader(tokenized_testset, batch_size=16, num_workers=2, pin_memory=True)
# print("[INFO] [SERVER] Centralized DataLoader cached globally and ready for execution.")
#
# def server_fn(context: Context):
#      """Create and configure the components for the ServerApp using global dataset cache."""
#      print("[INFO] [SERVER] Initializing ServerApp configuration...")
#     
#      if context.run_config:
#          num_rounds = context.run_config.get("num-server-rounds", 5)
#          fraction_fit = context.run_config.get("fraction-fit", 1.0)
#      else:
#          num_rounds = 5
#          fraction_fit = 1.0
#     
#      device = "cuda" if torch.cuda.is_available() else "cpu"
#      init_net = Net().to(device)
#      ndarrays = get_weights(init_net)
#      parameters = ndarrays_to_parameters(ndarrays)
#      del init_net
#      if torch.cuda.is_available():
#          torch.cuda.empty_cache()
#     
#      strategy = CustomFedAvg(
#          fraction_fit=fraction_fit,
#          fraction_evaluate=0.0,
#          min_available_clients=2,
#          initial_parameters=parameters,
#          evaluate_metrics_aggregation=weighted_average,
#          fit_metrics_aggregation=handle_fit_metrics,
#          on_fit_config_fn=on_fit_config,
#          evaluate_fn=get_evaluate_fn(testloader, device=device),
#      )
#      config = ServerConfig(num_rounds=num_rounds)
#      return ServerAppComponents(strategy=strategy, config=config)


#correct fed logic
def server_fn(context: Context):
    """Create and configure the components for the ServerApp."""
    print("[INFO] [SERVER] Initializing ServerApp configuration...")

    if context.run_config:
        num_rounds = context.run_config.get("num-server-rounds", 5)
        fraction_fit = context.run_config.get("fraction-fit", 1.0)
        print(f"[INFO] [SERVER] Run configuration loaded from context. Rounds: {num_rounds}, Fraction Fit: {fraction_fit}")
    else:
        num_rounds = 5
        fraction_fit = 1.0
        print("[WARNING] [SERVER] run_config was empty. Deploying fail-safe default values (Rounds: 5, Fraction Fit: 1.0).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] [SERVER] Target evaluation device set to: {device}")

    init_net = Net().to(device)
    ndarrays = get_weights(init_net)
    parameters = ndarrays_to_parameters(ndarrays)
    
    del init_net
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[INFO] [SERVER] Loading centralized validation dataset (rotten_tomatoes)...")
    testset = load_dataset("rotten_tomatoes", verbose=False)["test"]

    tokenizer = get_tokenizer()
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    print("[INFO] [SERVER] Tokenizing centralized test sequence dataset...")
    tokenized_testset = testset.map(tokenize_function, batched=True, verbose=False)
    tokenized_testset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


    testloader = DataLoader(tokenized_testset, batch_size=16, num_workers=2, pin_memory=True)
    print("[INFO] [SERVER] Centralized DataLoader is ready.")

    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0, 
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation=weighted_average, 
        fit_metrics_aggregation=handle_fit_metrics,  
        on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device=device),
    )
    config = ServerConfig(num_rounds=num_rounds)

    print("[INFO] [SERVER] ServerApp components initialized successfully. Handing over to runtime engine.")
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)