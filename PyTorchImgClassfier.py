import torchmetrics
import torchvision
import torchvision.transforms.v2 as T
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

def check_which_gpu():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device

def get_mnist_data():
    #while data is loaded we want to transform the image to TorchVisionImage class and also change the data values to
    #32 bit float
    toTensor = T.Compose([T.ToImage(),T.ToDtype(torch.float32, scale=True)])
    #load data into tensors
    #Get training data, download in locally and convert to tensor using the tranforms define abobe
    train_and_valid_data = torchvision.datasets.FashionMNIST(
        root="datasets", train=True, download=True, transform=toTensor)

    test_data = torchvision.datasets.FashionMNIST(
        root="datasets", train=False, download=True, transform=toTensor)
    torch.manual_seed(42)
    #Do ramdom split to get a validation set of 5000 images
    train_data, valid_data = torch.utils.data.random_split(
        train_and_valid_data, [55_000, 5_000])

    #create batch data loaders for mini batch processing
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    return train_loader, valid_loader, test_loader

#define the image classifier model with 2 hidden layers and using ReLU activation
class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes)
        )

    def forward(self, X):
        return self.mlp(X)

#evaluate the model. For this we have to move to evaluation model
def evaluate_tm(model, dl, metric, d):
    #swi-tch from training to evaluation model
    model.eval()
    #reset the metric object which is RMSE in this case
    metric.reset()  # reset the metric at the beginning
    with torch.no_grad():
        for X_batch, y_batch in dl:
            X_batch, y_batch = X_batch.to(d), y_batch.to(d)
            y_pred = model(X_batch)
            #compute the metric in every iteration
            metric.update(y_pred, y_batch)
    return metric.compute()

#train the model using mini batch
def train_model(model, optimizer, criterion, metric,
                train_loader, valid_loader, n_epochs, device):
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        for X_batch, y_batch in train_loader:
            model.train()
            #move tensors to gpu
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            #forward prop
            y_pred = model(X_batch)
            #compute loss
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            #backward prop
            loss.backward()
            #gradient descent
            optimizer.step()
            optimizer.zero_grad()
            #update the metric with the prediction and labels for this itertation
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            #evaluate using validation set
            evaluate_tm(model, valid_loader, metric, device).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history

def get_class_prob(model, te_loader,device):
    #get one image from test data
    model.eval()
    X_new, y_new = next(iter(te_loader))
    #move the tensor to device
    X_new = X_new[:3].to(device)

    #prediction will be logit which is a classification score and not class probabilities
    with torch.no_grad():
        pred = model(X_new)

    #convert to class probabilities
    y_proba = F.softmax(pred, dim=1)
    print(y_proba)


if __name__ == '__main__':
    d = check_which_gpu()
    tr_loader, val_loader, te_loader = get_mnist_data()
    torch.manual_seed(42)
    #first hidden layer will have 300 units and second hidden layer will have 100 unites and output will have
    #10 classes. Create the model using the custom module defined
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                            n_classes=10).to(d)
    #loss function will be cross entropy because it is a classification problem
    xentropy = nn.CrossEntropyLoss()
    #using the stochastic gradient descent as the optimiser
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    #since it is classification the metric to be considered is accuracy. Accuracy streaming metric has to be moved
    # to GPU
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(d)
    #train the model and check the accuracy of training set and validation set in each epoch
    _ = train_model(model, optimizer, xentropy, accuracy, tr_loader, val_loader, 3, d)
    get_class_prob(model, te_loader, d)