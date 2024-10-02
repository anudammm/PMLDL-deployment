import sys
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import os



class CNNClassificationModel(nn.Module):
    def __init__(self,):
        super(CNNClassificationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),

            nn.Linear(7 * 7 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        pass

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        print("working on GPU")
    else:
        print("working on CPU")
    script_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(script_dir, "..", "datasets", "mnist_train.csv")
    train = pd.read_csv(dataset_path)
    split = 0.8
    le = len(train)
    validation = train[int(le*split):]
    train = train[:int(le*split)]

    batch_size = 100
    num_epochs = 15
    target = train["label"]
    features = train.drop(columns = "label")/255
    trainDataLoader = DataLoader(torch.utils.data.TensorDataset(torch.Tensor(features.values),torch.Tensor(target.values)),batch_size=batch_size, shuffle = False)
    target = validation["label"]
    features = validation.drop(columns = "label")/255
    testDataLoader = DataLoader(torch.utils.data.TensorDataset(torch.Tensor(features.values),torch.Tensor(target.values)),batch_size=batch_size, shuffle = False)



    from torch.optim import Adam

    model = CNNClassificationModel().to(device)
    optimizer = Adam(model.parameters(),lr = 5e-4)
    crossEntropy = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        train_loop = tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), desc=f"Epoch {epoch}")
        model.train()
        train_loss = 0.0
        total_train = 0
        correct_train = 0

        # Training loop
        for i, (images, labels) in train_loop:
            images = images.view(batch_size, 1, 28, 28).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = crossEntropy(outputs, labels.long())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate training accuracy
            predicted = torch.max(outputs.data, 1)[1]
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update tqdm with loss
            train_loop.set_postfix({"loss": loss.item()})

        # Log training accuracy
        train_acc = correct_train / total_train
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Accuracy: {train_acc * 100:.2f}%")

        if writer:
            writer.add_scalar("Loss/train", train_loss / len(trainDataLoader), epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            val_loop = tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), desc="Val")
            for i, (images, labels) in val_loop:  # Use the correct validation loader
                images = images.view(batch_size, 1, 28, 28).to(device)
                labels = labels.to(device)
                outputs = model(images)

                loss = crossEntropy(outputs, labels.long())
                val_loss += loss.item()

                predicted = torch.max(outputs.data, 1)[1]
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                # Update tqdm with validation accuracy
                val_loop.set_postfix({"acc": (correct_val / total_val) * 100})

        # Log validation accuracy
        val_acc = correct_val / total_val
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_acc * 100:.2f}%")

        if writer:
            writer.add_scalar("Loss/val", val_loss / len(testDataLoader), epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

    torch.save(model.state_dict(), "best.pt")