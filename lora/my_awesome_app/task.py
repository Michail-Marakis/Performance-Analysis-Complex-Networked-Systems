import os
from collections import OrderedDict
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

MODEL_NAME = r"C:\Users\lenia\Desktop\numenv\bert-tiny"
RAW_DATA_DIR = r"C:\Users\lenia\Desktop\numenv\nolora\data\rotten_tomatoes_raw"

def Net():
    """Load BERT model, apply Safety Bypass, and wrap it with LoRA configuration."""
    print(f"[INFO] [TASK] Initializing global model with LoRA: {MODEL_NAME}")
    

    config = BertConfig.from_pretrained(MODEL_NAME, num_labels=2, local_files_only=True)
    base_model = BertForSequenceClassification(config)
    

    weights_path = os.path.join(MODEL_NAME, "pytorch_model.bin")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        base_model.load_state_dict(state_dict, strict=False)
    else:
        raise OSError(f"Δεν βρέθηκε το αρχείο βαρών στο: {weights_path}")
    

    lora_config = LoraConfig(
        r=16,                                   
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    
    return model

def get_tokenizer():
    """Return the BERT tokenizer."""
    return BertTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

def get_weights(net):
    """Extract ONLY the trainable LoRA parameters to save network bandwidth."""
    lora_state_dict = get_peft_model_state_dict(net)
    return [val.cpu().numpy() for _, val in lora_state_dict.items()]

def set_weights(net, parameters):
    """Load the received LoRA parameters into the model."""
    lora_keys = get_peft_model_state_dict(net).keys()
    params_dict = zip(lora_keys, parameters)
    
    state_dict = OrderedDict({
        k: torch.from_numpy(v) for k, v in params_dict
    })

    set_peft_model_state_dict(net, state_dict)

def load_data(partition_id: int, num_partitions: int):
    """Load and manually partition the local text data for the client."""
    print(f"[INFO] [Client {partition_id}] Preparing local data loaders from local Parquet storage...")
    tokenizer = get_tokenizer()

    train_parquet_path = os.path.join(RAW_DATA_DIR, "train.parquet")
    local_dataset = load_dataset("parquet", data_files={"train": train_parquet_path})
    full_train = local_dataset["train"]
    
    partition_size = len(full_train) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size
    
    client_train_full = full_train.select(range(start_idx, end_idx))
    split_data = client_train_full.train_test_split(test_size=0.2, seed=42)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    train_set = split_data["train"].map(tokenize_function, batched=True)
    test_set = split_data["test"].map(tokenize_function, batched=True)

    train_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_set.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    trainloader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    testloader = DataLoader(test_set, batch_size=16, num_workers=0, pin_memory=True)

    return trainloader, testloader

def train(net, trainloader, epochs, lr, device):
    """Train the LoRA layers of the BERT model."""
    print(f"[INFO] [TASK] Starting local LoRA training on device: {device} for {epochs} epoch(s)")
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
    return avg_trainloss

def test(net, testloader, device):
    """Evaluate the LoRA enhanced BERT model."""
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