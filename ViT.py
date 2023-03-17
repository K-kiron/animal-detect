import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from vit_pytorch import ViT

# Download and preprocess the AWA2 dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data = datasets.ImageFolder('/Users/kironrothschild/Downloads/Animals_with_Attributes2/JPEGImages', transform=transform)
train_data, val_data = train_test_split(data, test_size=0.2)

# Load and prepare the ViT model
model = ViT(
    image_size = 224,
    patch_size = 32,
    num_classes = 50,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# Train the ViT model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    running_loss = 0.0
    for inputs, labels in train_data:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(0))
        loss = criterion(outputs, labels.unsqueeze(0))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1} loss: {running_loss/len(train_data)}')

# Evaluate the ViT model on the validation set
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_data:
        outputs = model(inputs.unsqueeze(0))
        _, predicted = torch.max(outputs.data, 1)
        total += 1
        correct += (predicted == labels.unsqueeze(0)).sum().item()

print(f'Accuracy on validation set: {100*correct/total}%')