import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


model = torchvision.models.resnet18()
# model.features.conv0 = nn.Conv2d(
#     in_channels=1,  # Change input channels from 3 (RGB) to 1 (Grayscale)
#     out_channels=64,  # Keep this the same as the original model
#     kernel_size=(7, 7),
#     stride=(2, 2),
#     padding=(3, 3),
#     bias=False
# )
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
)

# model.features[0][0] = nn.Conv2d(
#     in_channels=1,  # Change input channels from 3 (RGB) to 1 (Grayscale)
#     out_channels=32,
#     kernel_size=(3, 3),
#     stride=(2, 2),
#     padding=(1, 1),
#     bias=False
# )
# model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))


# Create the model & hyperparameters
# model = ResNet(ResBox)

## Load data

# label_map = {str(i): i for i in range(10)}
# label_map.update({chr(65 + i): 10 + i for i in range(26)})
# label_map.update({chr(97 + i): 36 + i for i in range(26)})


class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations (image paths and labels).
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample (image).
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)

        # Load the label (assuming the label is in the second column)
        label = self.data_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        # print(label)
        return image, torch.tensor(label, dtype=torch.long)


# Example of using the dataset
transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        # transforms.Resize((128, 128)),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=(-30, 30)),
        transforms.Normalize(mean=0.485, std=0.229),  # Normalize
    ]
)
print("loading data...")
# Create dataset instance
# train_dataset = CustomImageDataset(
#     csv_file="./English_Dataset/train_dataset.csv",
#     root_dir="./English_Dataset",
#     transform=transform,
# )
# test_dataset = CustomImageDataset(
#     csv_file="./English_Dataset/test_dataset.csv",
#     root_dir="./English_Dataset",
#     transform=transform,
# )
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=False
)


# train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

print("data loaded")
# hidden_size = 512
# num_classes = label_map.__len__()

# model = Cnn(hidden_size, num_classes)
# model = ResNet(ResBox)
# model.load_state_dict(torch.load("./model.pt"))
# print(model)


num_epochs = 10
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fct = nn.CrossEntropyLoss()
losses = []
model.train()
for epoch in range(num_epochs):
    for idx, (image, label) in enumerate(train_loader):
        # forward pass
        output = model(image)
        loss = loss_fct(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (idx % 5) == 0:
            losses.append(loss.item())
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{idx}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

torch.save(model.state_dict(), "MNIST_ResNet18_v2.pt")


fig, ax = plt.subplots()
ax.plot(losses)
ax.grid()
fig.savefig("MNIST_ResNet18_v2_loss.png")

all_labels = []
all_predictions = []

with torch.no_grad():
    for data, label in test_loader:
        output = model(data)
        _, prediction = torch.max(output, 1)
        all_labels.extend(label.cpu().numpy())
        all_predictions.extend(prediction.cpu().numpy())

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Compute the accuracy
accuracy = accuracy_score(all_labels, all_predictions)
print(f"Accuracy: {accuracy:.4f}")

# Compute precision, recall, and F1-score
precision = precision_score(all_labels, all_predictions, average="weighted")
recall = recall_score(all_labels, all_predictions, average="weighted")
f1 = f1_score(all_labels, all_predictions, average="weighted")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Generate a classification report
report = classification_report(all_labels, all_predictions)
print("Classification Report:\n", report)

# Compute the confusion matrix
cm = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:\n", cm)
