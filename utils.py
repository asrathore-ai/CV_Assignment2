import os 
import torch 

from pathlib import Path 
from dotenv import load_dotenv

from torch.utils.data import random_split, TensorDataset
from torchvision.transforms import v2

from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

load_dotenv()

def get_q1_data():
    data_root= Path(os.getenv("DATA_ROOT"))
    data_root = data_root / "Q1"

    train_labels = torch.load(data_root / "train_labels.pt")
    test_labels = torch.load(data_root / "test_labels.pt")

    labels = torch.cat((train_labels, test_labels))
    unique = torch.unique(labels)

    return {
        "num_classes": unique.shape[0],
        "train":{
            "x": torch.load(data_root / "train_data.pt"),
            "y": train_labels
        },
        "test":{
            "x": torch.load(data_root / "test_data.pt"),
            "y": test_labels
        }
    }

def get_datasets(device, resize=None, dtype=torch.float32):
    
    data = get_q1_data()
    transforms = []

    if resize is not None:
        assert isinstance(resize, tuple), f"Expected resize to be a tuple, got: {type(resize)}"
        assert len(resize) == 2, f"Expected resize shapes to be 2D, got {len(resize)}"      
        transforms.append(v2.Resize(size=resize))

    MEAN = torch.tensor([0.485 , 0.456 , 0.406]).to(dtype).to(device)
    STD = torch.tensor([0.229 , 0.224 , 0.225]).to(dtype).to(device)
    transforms.append(v2.Normalize(mean=MEAN, std=STD))
    final_tform = v2.Compose(transforms)

    # Train dataset
    imgs = data["train"]["x"].to(dtype).to(device)
    imgs = final_tform(imgs)

    labels = data["train"]["y"].to(device).long()
    
    full_dataset = TensorDataset(imgs, labels)
    splits = random_split(full_dataset, [0.8, 0.2])
    
    train_dataset = splits[0]
    val_dataset = splits[1]

    # Test dataset 
    imgs = data["test"]["x"].to(dtype).to(device)
    imgs = final_tform(imgs)
    
    labels = data["test"]["y"].to(device).long()
    test_dataset = TensorDataset(imgs, labels)

    return train_dataset, val_dataset, test_dataset, data["num_classes"]

class Metrics:
    
    def __init__(self, split):
        self.title=split
        self.total_count=0
        self.correct_count=0
        self.total_loss=0
        self.batch_count=0

    def update(self, loss, logits, y):
        assert isinstance(loss, torch.Tensor), "Expect loss to be torch.tensor"
        
        self.total_count += y.shape[0]
        self.batch_count += 1

        self.total_loss += loss.item()

        y_hat = torch.argmax(logits, dim=1)    
        correct = torch.sum(y_hat == y)
        self.correct_count += correct.item()

    def get_values(self):
        accuracy = (self.correct_count / self.total_count)*100
        loss = self.total_loss  / self.batch_count
        return accuracy, loss 

def print_dataset_details(console, train, val, test, num_classes, batch_img):

    data_table = Table(title="Loading Data Q1.1", show_header=True, header_style="bold magenta")
    data_table.add_column("Category", style="dim")
    data_table.add_column("Value")

    data_table.add_row("Train Samples", f"{len(train):,}") 
    data_table.add_row("Val Samples", f"{len(val):,}")
    data_table.add_row("Test Samples", f"{len(test):,}")
    data_table.add_row("Num Classes", str(num_classes))
    data_table.add_row("Img Shape", str(list(batch_img.shape[1:])))

    console.print(data_table)

def print_model_details(console, batch_size, device, model, model_weights, criterion, epochs, optimizer, scheduler):
    train_table = Table(title="Training Params")
    train_table.add_row("Batch Size", str(batch_size))
    train_table.add_row("Device", str(device).upper(), style="green")
    train_table.add_row("Model", str(model))
    train_table.add_row("Weights Type", str(model_weights))
    train_table.add_row("Loss", str(criterion))
    train_table.add_row("Epochs", str(epochs))
    train_table.add_row("Optimizer", str(optimizer))
    train_table.add_row("Scheduler", str(scheduler))
    console.print(train_table)

def print_metrics(console, e, train_accuracy, train_loss, val_accuracy, val_loss, lr):

    epoch_table = Table(title=f"Metrics for Epoch {e}", show_header=False)
    epoch_table.add_row("Learning Rate", f"{lr}")
    
    epoch_table.add_section()
    epoch_table.add_row(Text("Train Metrics", justify="center", style="bold yellow"), style="on blue")
    epoch_table.add_row("Accuracy", f"{train_accuracy:.6f}")
    epoch_table.add_row("Loss", f"{train_loss:.6f}")

    epoch_table.add_section()
    epoch_table.add_row(Text("Validation Metrics", justify="center", style="bold yellow"), style="on blue")
    epoch_table.add_row("Accuracy", f"{val_accuracy:.6f}")
    epoch_table.add_row("Loss", f"{val_loss:.6f}")
    console.print(epoch_table)