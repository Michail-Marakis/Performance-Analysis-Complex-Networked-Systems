"""my-awesome-app: A Flower / PyTorch app."""

import json
from random import random

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import ConfigRecord, Context

from my_awesome_app.task import Net, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    """Flower client implementation for local dataset training and evaluation."""

    def __init__(self, net, trainloader, valloader, local_epochs, context: Context):
        """Initialize the client, configure execution device, and setup state tracking."""
        self.client_state = context.state
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        if "fit_metrics" not in self.client_state.config_records:
            self.client_state.config_records["fit_metrics"] = ConfigRecord()

    def fit(self, parameters, config):
        """Train the local model parameters using the configurations provided by the ServerApp."""
        set_weights(self.net, parameters)

        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            config["lr"],
            self.device,
        )

        fit_metrics = self.client_state.config_records["fit_metrics"]
        if "train_loss_hist" not in fit_metrics:
            fit_metrics["train_loss_hist"] = [train_loss]
        else:
            fit_metrics["train_loss_hist"].append(train_loss)

        complex_metric = {"a": 123, "b": random(), "mylist": [1, 2, 3, 4]}
        complex_metric_str = json.dumps(complex_metric)

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {
                "train_loss": train_loss,
                "my_metric": complex_metric_str,
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate the received global model parameters using the local validation set."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Instantiate and return a Flower Client instance associated with a specific data partition."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    print(f"[INFO] [CLIENT] Spawning virtual client instance for Partition ID: {partition_id}")

    net = Net()
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    print(f"[INFO] [CLIENT] Successfully mounted execution state for Partition ID: {partition_id}")
    return FlowerClient(net, trainloader, valloader, local_epochs, context).to_client()


app = ClientApp(client_fn=client_fn)