import torch
import torch.nn as nn
import torchmetrics
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T

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
        print(f"Epoch {epoch + 1}/{n_epochs} ",
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history

#applying He initialisation to hidden layers for avoiding vanishing and exploding grads
#He Initialisation is based on variance sqrt(2/(num units in i/p layer))
def use_he_ini(module):
    if(isinstance(module,nn.Linear)):
        #multiply He based initialisation variance to all the weights of the layer
        nn.init.kaiming_uniform(module.weight)
        nn.init.constant_(module.bias,0)

def build_model(seed=42, device=torch.device("cpu")):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).to(device)
    model.apply(use_he_ini)
    return model

def test_opt(model, optimizer, train_loader, valid_loader, test_loader, n_epochs=10,  device=torch.device("cpu")):

    xentropy = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    history = train_model(model, optimizer, xentropy, accuracy.to(device),
                    train_loader, valid_loader, n_epochs,device)
    return history, evaluate_tm(model, test_loader, accuracy, device)

if __name__ == '__main__':
    tr_loader, val_loader, te_loader = get_mnist_data()
    device = check_which_gpu()


    #test SGD
    model = build_model(seed=42, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    history_sgd, acc_sgd = test_opt(model, optimizer,tr_loader,val_loader,te_loader, device=device)
    print("Completed SGD optimizer based model")

    tr_loader

    #test Momentum
    model = build_model(seed=42, device=device)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.05)
    history_momentum, acc_momentum = test_opt(model, optimizer,tr_loader,val_loader,te_loader, device=device)
    print("Completed Momemtum optimizer based model")

    #test RMS Prop
    model = build_model(seed=42, device=device)
    optimizer = torch.optim.RMSprop(model.parameters(), alpha=0.9, lr=0.05)
    history_rmsprop, acc_rmsprop = test_opt(model, optimizer,tr_loader,val_loader,te_loader, device=device)
    print("Completed RMS Prop optimizer based model")

    #test Adam
    model = build_model(seed=42, device=device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=0.05)
    history_adam, acc_adam = test_opt(model, optimizer,tr_loader,val_loader,te_loader, device=device)
    print("Completed Adam optimizer based model")

    for plot in ("train_losses", "valid_metrics"):
        plt.figure(figsize=(8, 4))
        opt_names = "SGD Momentum Nesterov AdaGrad RMSProp Adam Adamax Nadam AdamW"
        for history, opt_name in zip(
                (history_sgd, history_momentum,
                 history_rmsprop, history_adam,
                 ), opt_names.split()):
            plt.plot(history[plot], label=opt_name, linewidth=3)

        plt.grid()
        plt.xlabel("Epochs")
        plt.ylabel({"train_losses": "Training loss", "valid_metrics": "Validation accuracy"}[plot])
        plt.legend(loc="upper right")
        plt.show()



