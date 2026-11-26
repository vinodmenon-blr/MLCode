import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_sample_images
import torchvision
import torchvision.transforms.v2 as T
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from functools import partial

def check_which_gpu():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device

def plot_image(image):
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")


def check_feature_maps_conv():
    sample_imgs = np.stack(load_sample_images()["images"])
    sample_imgs = torch.tensor(sample_imgs, dtype=torch.float32)/255
    #change the dimensions from batch,height,width,channels to batch,channels,height,width
    sample_imgs_permuted = sample_imgs.permute(0,3,1,2)
    #plot images
    plt.figure(figsize=(8, 4))
    for index, image in enumerate(sample_imgs_permuted):
        plt.subplot(1, 2, index + 1)
        plot_image(image)
    plt.show()

    #crop the images
    cropped_images = T.CenterCrop((70, 120))(sample_imgs_permuted)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plot_image(cropped_images[0])
    plt.subplot(1, 2, 2)
    plot_image(cropped_images[1])
    plt.show()

    torch.manual_seed(42)
    #convolution with in chanels 3 (RGB) , out channels 32 (ie 32 filters used) which are 7x7 filters
    conv_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7)
    #There will be 32 feature maps since there are 32 filters used
    fmaps = conv_layer(cropped_images)
    #Let us check out what we have in these feature maps
    plt.figure(figsize=(8, 4))
    for image_idx in (0, 1):
        #We are plotting 2 feature maps of the two images
        #You can see initial convolution layers will be more acting as edge detectors which detects
        #different kinds of edges
        for fmap_idx in (0, 1):
            plt.subplot(2, 2, image_idx * 2 + fmap_idx + 1)
            plt.imshow(fmaps[image_idx, fmap_idx].detach(), cmap="gray")
            plt.axis("off")
    plt.show()

def evaluate_tm(model, data_loader, metric, device):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            metric.update(y_pred, y_batch)
    return metric.compute()

def train(model, optimizer, loss_fn, metric, train_loader, valid_loader,
          n_epochs, device):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.0
        metric.reset()
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric,device).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history

def train_conv_net():
    device = check_which_gpu()
    torch.manual_seed(42)  # extra code – ensure reproducibility
    DefaultConv2d = partial(nn.Conv2d, kernel_size=3, padding="same")
    model = nn.Sequential(
        DefaultConv2d(in_channels=1, out_channels=64, kernel_size=7), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        DefaultConv2d(in_channels=64, out_channels=128), nn.ReLU(),
        DefaultConv2d(in_channels=128, out_channels=128), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        DefaultConv2d(in_channels=128, out_channels=256), nn.ReLU(),
        DefaultConv2d(in_channels=256, out_channels=256), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(in_features=2304, out_features=128), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=128, out_features=64), nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=64, out_features=10),
    ).to(device)

    #Use FashionMNIST data set to train the Convnet for classification
    toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

    train_and_valid_data = torchvision.datasets.FashionMNIST(
        root="datasets", train=True, download=True, transform=toTensor)
    test_data = torchvision.datasets.FashionMNIST(
        root="datasets", train=False, download=True, transform=toTensor)

    torch.manual_seed(42)
    train_data, valid_data = torch.utils.data.random_split(
        train_and_valid_data, [55_000, 5_000])

    #construct batch loaders
    torch.manual_seed(42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    #Train using batch gradient descent using AdamW optimiser
    n_epochs = 5
    optimizer = torch.optim.AdamW(model.parameters())
    xentropy = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    history = train(model, optimizer, xentropy, accuracy,
                    train_loader, valid_loader, n_epochs, device)
    #predict for a test image
    image, true_label = test_data[0]
    #push to device
    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)  # Raw output: [1, 10]
        probabilities = torch.softmax(logits, dim=-1)  # Normalized probabilities
        pred_class_idx = probabilities.argmax(dim=-1).item()  # Predicted class index (0 to 9)

    class_names = test_data.classes  # FashionMNIST class names
    print(f"Predicted class: {class_names[pred_class_idx]} ({pred_class_idx})")
    print(f"True class:      {class_names[true_label]} ({true_label})")
    print(f"Confidence:      {probabilities[0, pred_class_idx].item() * 100:.2f}%")


if __name__ == '__main__':
    check_feature_maps_conv()
    train_conv_net()