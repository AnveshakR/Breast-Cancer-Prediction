import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_directory = "./train/"

#Constants
LEARNING_RATE = 0.001
BATCH_SIZE = 256
NUM_EPOCHS = 5

IMG_SIZE = 500
ENCODED_SIZE = 500

#Defining the Autoencoder class
class Autoencoder(nn.Module):

    def __init__(self, img_size, encoded_space_size):
        super(Autoencoder, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 7, stride=2, padding=0),
            nn.ReLU(True)
        )

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 7, 
            stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 3, stride=2, 
            padding=1, output_padding=1)
        )

    def forward(self, x):

        x = self.encoder_cnn(x)
        x = self.decoder_cnn(x)
        x = torch.sigmoid(x)
        return x
    
    def loss(self, x, y):
        return F.mse_loss(x, y)


#Defining the dataloader
transform = torchvision.transforms.Compose([

    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.ToTensor()
])

dataset = torchvision.datasets.ImageFolder(train_directory, transform=transform)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#Defining the model
model = Autoencoder(IMG_SIZE, ENCODED_SIZE).to(device)

#Defining the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training the model
for epoch in range(NUM_EPOCHS):

    for i, data in enumerate(dataloader):
        img, _ = data
        img = img.to(device)
        # ===================forward=====================
        output = model(img)
        loss = model.loss(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
        if i % 10 == 0:
            print('batch [{}/{}], loss:{:.4f}'
            .format(i + 1, len(dataloader), loss.item()))
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, NUM_EPOCHS, loss.item()))


val_dir = "./val/"

val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)

val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

#Testing the model
data = next(iter(val_dataloader))

load_model = torch.load("./autoencoder.pth")

accuracy = 0

data = next(iter(val_dataloader))

img, label = data

output = load_model(img)

loss = load_model.loss(output, img)

loss = 1 - torch.sigmoid(loss)


loss = model.loss(output, img)
loss

# Print the first 5 images in output
for i in range(5):
    plt.imshow(output[i].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()


# Print the first 5 images in img
for i in range(5):
    plt.imshow(img[i].permute(1, 2, 0).detach().cpu().numpy())
    plt.show()


#print the first 5 images in img and output side by side
for i in range(5):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img[i].permute(1, 2, 0).detach().cpu().numpy())
    ax1.set_xlabel("Original Image")
    ax2.imshow(output[i].permute(1, 2, 0).detach().cpu().numpy())
    ax2.set_xlabel("Reconstructed Image")
    plt.show()


model.to("cpu")
total = 128
correct = 0
for i, data in enumerate(val_dataloader):
    if i==128:
        break
    img, label = data
    output = model(img)
    loss = model.loss(output, img)
    prediction = 1 if loss.item() < 0.01 else 0
    if prediction == label.item():
        correct += 1

accuracy = correct / total


print("Model accuracy over 128 test images = ",accuracy)