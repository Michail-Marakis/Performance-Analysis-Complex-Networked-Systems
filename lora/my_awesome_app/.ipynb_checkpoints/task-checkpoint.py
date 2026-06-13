"""my-awesome-app: A Flower / PyTorch app."""

from collections import OrderedDict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader

#MODEL_NAME = "distilbert-base-uncased"
MODEL_NAME = "prajjwal1/bert-tiny"

def Net():
    """Load the base DistilBERT model from Hugging Face for binary classification."""
    print(f"[INFO] [TASK] Initializing global model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    print("[INFO] [TASK] Model loaded successfully.")
    return model

def get_tokenizer():
    """Return the DistilBERT tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)

fds = None

def load_data(partition_id: int, num_partitions: int):
    """Load the local text partition for the client using a Dirichlet distribution."""
    global fds
    print(f"[INFO] [Client {partition_id}] Preparing local data loaders...")
    tokenizer = get_tokenizer()

    if fds is None:
        print("[INFO] [TASK] Initializing FederatedDataset via DirichletPartitioner (alpha=1.0)...")
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions, partition_by="label", alpha=1.0
        )
        fds = FederatedDataset(
            dataset="rotten_tomatoes",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    print(f"[INFO] [Client {partition_id}] Executing sequence tokenization...")
    train_set = partition_train_test["train"].map(tokenize_function, batched=True, verbose=False)
    test_set = partition_train_test["test"].map(tokenize_function, batched=True, verbose=False)

    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    trainloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(test_set, batch_size=16, num_workers=2, pin_memory=True)

    
    print(f"[INFO] [Client {partition_id}] Data loaders are fully prepared.")
    return trainloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the entire DistilBERT model (Baseline Full FL)."""
    print(f"[INFO] [TASK] Starting local training on target device: {device} for {epochs} epoch(s)")
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    net.train()

    running_loss = 0.0
    total_batches = len(trainloader)

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(trainloader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = net(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  └─ [Epoch {epoch+1}/{epochs}] Step {batch_idx}/{total_batches} - Current Loss: {loss.item():.4f}")

    avg_trainloss = running_loss / (len(trainloader) * epochs)
    print(f"[INFO] [TASK] Local training complete. Average Training Loss: {avg_trainloss:.4f}")
    return avg_trainloss


def test(net, testloader, device):
    """Evaluate the DistilBERT model and return loss and accuracy."""
    print("[INFO] [TASK] Starting global model evaluation on centralized/validation set...")
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()

    with torch.no_grad():
        for batch in testloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = net(input_ids=input_ids, attention_mask=attention_mask)
            loss += criterion(outputs.logits, labels).item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    print(f"[METRIC] Evaluation Complete -> Loss: {loss:.4f} | Accuracy: {accuracy*100:.2f}%")
    return loss, accuracy


def get_weights(net):
    """Extract ALL model parameters (Full FL Baseline)."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    """Load the received parameters into the model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({
        k: torch.from_numpy(v).to(val.device) 
        for (k, val), (_, v) in zip(net.state_dict().items(), params_dict)
    })
    net.load_state_dict(state_dict, strict=True)
