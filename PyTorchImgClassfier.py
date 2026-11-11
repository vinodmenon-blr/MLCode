import optuna
import torchmetrics
import torchvision
import torchvision.transforms.v2 as T
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import optuna
import os
import copy

best_trained_model = None
global_best_accuracy = 0.0

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
        print(
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

def objective_func(trial, tr_loader, val_loader):
    #optima provides the learning rate and number of hidden units
    #log=True means learning rate is considered in log scale distribution to sample
    global best_trained_model, global_best_accuracy
    device = check_which_gpu()
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)
    best_validation_accuracy = 0.0
    model = ImageClassifier(n_inputs=1*28*28, n_hidden1=n_hidden, n_hidden2=n_hidden, n_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    xentropy = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
    n_epochs = 5
    for epoch in range(5):
        history = train_model(model, optimizer, xentropy, accuracy, tr_loader, val_loader,1, device)
        validation_accuracy = max(history["valid_metrics"])
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
        if validation_accuracy > global_best_accuracy:
            global_best_accuracy = validation_accuracy
            #This is the best model so far so doing a deep copy to keep
            #model for compile and save
            best_trained_model = copy.deepcopy(model)
        #pass on the accuracy of the epoch
        trial.report(validation_accuracy, epoch)
        #check with median pruner on whether the trial has to be pruned
        if trial.should_prune():
            print("Trial Pruned: Trial number " + str(trial.number))
            raise optuna.TrialPruned()

        print(f"Epoch {epoch + 1}/{n_epochs} ")

    return best_validation_accuracy


def compile_and_save_model(model, device, save_path="./trained_model.pt2"):
    model.eval()
    batch_dim = torch.export.Dim("batch", min=1, max=1024)
    example_input = (torch.randn(2, 1, 28, 28, device=device),)

    export_pgm = torch.export.export(
        model,
        example_input,
        dynamic_shapes=({0: batch_dim},)
    )
    torch.export.save(export_pgm, save_path)
    print(f"Model successfully exported to {save_path}")

def load_and_run_model(save_path="./trained_model.pt2", te_loader=None, device="cpu"):
    loaded_pgm = torch.export.load(save_path)
    loaded_model = loaded_pgm.module() #Is is in evaluation model which can be used for inference

    X_test, y_test = next(iter(te_loader))
    X_sample = X_test[:3].to(device)

    #inference from loaded model
    with torch.no_grad():
        logits = loaded_model(X_sample)
        probabilities = F.softmax(logits, dim=1)
        preds = torch.argmax(probabilities, dim=1)

    print("Loaded model output probabilities:\n", probabilities)
    print("Predicted classes:", preds.cpu().numpy())
    print("True classes:     ", y_test[:3].numpy())


if __name__ == '__main__':
    d = check_which_gpu()
    tr_loader, val_loader, te_loader = get_mnist_data()
    torch.manual_seed(42)
    #Sampler used for choosing the hyper parameters is Tree Structured Parzen Estimator.
    #It starts with random sampling and progressively focuses on most promising spaces in the
    #hyper parameter space leading to better sampling than random sampling
    sampler = optuna.samplers.TPESampler()
    #using lambda to pass the loaders to objective function
    objective_func_with_params = lambda trial: objective_func(trial, tr_loader=tr_loader, val_loader=val_loader)
    #Will use a pruner to interrupt training raising an exception to ignore the specific trial all together
    #because the metrics do not look good. In the case of MediaPruner it will prune the trial if the performance is
    #is lower than the median performance. This is checked in regular intervals.
    pruner = optuna.pruners.MedianPruner()
    #Will do 5 trials with different values of hyper params sampled by TPE
    #create study with sampler and pruner
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective_func_with_params, n_trials=5)
    #What is the best params out of the 5 trials done?
    print(study.best_params)

    compile_and_save_model(best_trained_model, d, "./trained_model.pt2")
    load_and_run_model("./trained_model.pt2", te_loader, d)
