"""my-awesome-app: A Flower / PyTorch app."""

import json
import time
from datetime import datetime
import torch
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .task import Net, set_weights


class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy that logs round latency, cumulative transfer size, and benchmarks locally."""

    def __init__(self, *args, **kwargs):
        """Initialize the custom strategy and setup production-ready metrics logging."""
        super().__init__(*args, **kwargs)
        self.results_to_save = {}
        self.cumulative_data_transferred_mb = 0.0
        
        print("[SERVER] [INFO] Strategy initialized in offline benchmarking mode.")
        print("[SERVER] [INFO] Tracking communication costs.")

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate local model updates, compute network transfer volume, and measure round latency."""
        start_time = time.time()

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        round_duration = time.time() - start_time

        if parameters_aggregated is not None:
            ndarrays = parameters_to_ndarrays(parameters_aggregated)
            total_bytes = sum(arr.nbytes for arr in ndarrays)
            total_mb = total_bytes / (1024 * 1024)
            
            self.cumulative_data_transferred_mb += total_mb

    
            print(f"[SERVER] [INFO] [ROUND {server_round}] Aggregation completed in {round_duration:.2f} seconds.")
            print(f"[SERVER] [METRIC] [ROUND {server_round}] Current round payload: {total_mb:.2f} MB")
            print(f"[SERVER] [METRIC] [ROUND {server_round}] Cumulative network traffic: {self.cumulative_data_transferred_mb:.2f} MB")

        
            model = Net()
            set_weights(model, ndarrays)
            checkpoint_path = f"global_model_round_{server_round}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[SERVER] [INFO] [ROUND {server_round}] Global model checkpoint saved to: {checkpoint_path}")

        return parameters_aggregated, metrics_aggregated

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate the global model parameters and export server-side metrics to JSON."""
        eval_res = super().evaluate(server_round, parameters)
        if eval_res is None:
            return None

        loss, metrics = eval_res

        my_results = {
            "loss": loss, 
            "round_duration_seconds": None,
            "cumulative_mb": self.cumulative_data_transferred_mb,
            **metrics
        }
        self.results_to_save[server_round] = my_results

        print(f"[SERVER] [METRIC] [EVAL_ROUND {server_round}] Centralized loss: {loss:.5f}")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"[SERVER] [METRIC] [EVAL_ROUND {server_round}] {metric_name}: {value*100:.2f}%")
            else:
                print(f"[SERVER] [METRIC] [EVAL_ROUND {server_round}] {metric_name}: {value}")

    
        output_filename = "results.json"
        with open(output_filename, "w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        return loss, metrics