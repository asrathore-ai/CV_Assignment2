import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from rich.table import Table
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from tqdm import tqdm 

from utils import get_datasets, print_metrics, Metrics, print_dataset_details, print_model_details


console = Console()

device=torch.device("cuda") if torch.cuda.is_available() else "cpu"
batch_size=128
train, val, test, num_classes = get_datasets(device=torch.device("cpu"), resize=(224, 224))


train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

batch_img, _ = next(iter(train_loader))
print_dataset_details(console, train, val, test, num_classes, batch_img)

from torchvision.models import resnet18, ResNet18_Weights

model_weights=ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=model_weights)
model.fc=nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)
criterion=nn.CrossEntropyLoss(reduction="sum")

epochs=100
lr = 5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler=MultiStepLR(optimizer, [30,60], gamma=0.1)

print_model_details(console, batch_size, device, model, model_weights, criterion, epochs, optimizer, scheduler)

import wandb 
wandb.init(
    project="CV_Assignment_2",
    name="resnet18_imagenet_imgsize_224",
    config={
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "architecture": "ResNet18",
        "optimizer": "Adam",
        "scheduler": "MultiStepLR"
    }
)

    
def step(x, y, model, loss_fn, metrics):
    # Data is already on cuda
    logits = model(x)
    loss = loss_fn(logits, y)
    metrics.update(loss, logits, y)
    return loss 

    

for e in range(epochs):
    
    #Train 
    train_metrics=Metrics("Train")
    model.train()
    for x,y in tqdm(train_loader, desc=f"Training Loop : {e}"):
        x = x.to(device)
        y = y.to(device)
        loss = step(x,y,model,criterion,train_metrics)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    #Val
    val_metrics=Metrics("Val")
    model.eval()
    with torch.no_grad():
        for x,y in tqdm(val_loader, desc=f"Validation Loop : {e}"):
            x = x.to(device)
            y = y.to(device)
            loss = step(x,y,model,criterion,val_metrics)

    train_accuracy, train_loss = train_metrics.get_values()
    val_accuracy, val_loss = val_metrics.get_values()
    lr = optimizer.param_groups[0]['lr']

    print_metrics(console, e, train_accuracy, train_loss, val_accuracy, val_loss, lr)

    wandb.log({
        "epoch": e,
        "train/loss": train_loss,
        "train/acc": train_accuracy,
        "val/loss": val_loss,
        "val/acc": val_accuracy,
        "lr": lr
    })

    
#Test
model.eval()
test_metrics=Metrics("Test")
with torch.no_grad():
    for x,y in tqdm(test_loader):
        x = x.to(device)
        y = y.to(device)
        loss = step(x,y,model,criterion,test_metrics)
test_accuracy, test_loss = test_metrics.get_values()

print(f"Final Accuracy = {test_accuracy} | Test Loss = {test_loss}")

wandb.run.summary["test_accuracy"] = test_accuracy
wandb.run.summary["test_loss"] = test_loss
wandb.finish()
