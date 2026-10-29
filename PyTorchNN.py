import numpy as np
import torch
import timeit

import torchmetrics
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def check_which_gpu():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device

def check_cuda_available():
    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available? {cuda_available}")

    if cuda_available:
        # 2. Get the number of available GPUs
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        # 3. Get the name of the active GPU
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

        # 4. Perform a real test: Create a tensor directly on the GPU
        try:
            device = torch.device("cuda")
            x = torch.rand(3, 3, device=device)
            y = torch.rand(3, 3, device=device)
            z = torch.matmul(x, y)
            print("\nSuccess! Matrix multiplication performed on GPU:")
            print(z)
        except Exception as e:
            print(f"\nError initializing tensor on GPU: {e}")
    else:
        print("\nCUDA is not available. PyTorch is defaulting to the CPU.")


def compare_cpu_vs_gpu(d):
    M_cpu = torch.rand((1000, 1000))
    # Pass globals=globals() so timeit can find M_cpu
    cpu_time = timeit.timeit(lambda: M_cpu @ M_cpu.T, number=10)
    print("CPU time: " + str(cpu_time) + " seconds")

    # GPU timing
    #Push the tensor to GPU
    M_gpu = M_cpu.to(d)

    # Warm-up run for GPU initialization
    _ = M_gpu @ M_gpu.T
    if d == "cuda":
        """
         GPU synchronization: GPU operations in PyTorch run asynchronously on the CUDA stream. 
         If you do not call torch.cuda.synchronize(), timeit only measures the tiny CPU queue time rather than the 
         actual GPU calculation time.
        """
        torch.cuda.synchronize()

    def run_gpu():
        _ = M_gpu @ M_gpu.T
        if d == "cuda":
            torch.cuda.synchronize()

    gpu_time = timeit.timeit(run_gpu, number=10)

    print("GPU time: " + str(gpu_time) + " seconds")

def get_housing_data():
    housing = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,random_state=42)
    X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,random_state=42)

    #create tensors
    X_train = torch.FloatTensor(X_train)
    X_valid = torch.FloatTensor(X_valid)
    X_test = torch.FloatTensor(X_test)

    #normalise training data
    means = X_train.mean(dim=0, keepdims=True)
    stds = X_train.std(dim=0, keepdims=True)
    X_train = (X_train - means) / stds
    """
     The model learns weights based on the scale and shift determined by the training set. 
     If validation or test data were scaled using their own separate mean and std, the feature values would 
     no longer align with the weights learned by the model.
    """
    X_valid = (X_valid - means) / stds
    X_test = (X_test - means) / stds

    #change the shape with one row for target per sample
    y_train = torch.FloatTensor(y_train).view(-1, 1)
    y_valid = torch.FloatTensor(y_valid).view(-1, 1)
    y_test = torch.FloatTensor(y_test).view(-1, 1)

    return X_train,X_valid,X_test, y_train,y_valid, y_test

def linear_reg_using_torch_auto_diff(X_train,X_valid,X_test, y_train,y_valid, y_test):

    #generate ramdom weights to initialise the NN
    torch.manual_seed(42)
    n_features = X_train.shape[1]  # there are 8 input features
    w = torch.randn((n_features, 1), requires_grad=True)
    b = torch.tensor(0., requires_grad=True)

    #Run forward and backward prop
    learning_rate = 0.4
    n_epochs = 20
    for epoch in range(n_epochs):
        #forward prop
        y_pred = X_train @ w + b
        #loss function which is mean square error
        loss = ((y_pred - y_train) ** 2).mean()
        #gradient compute for lossfunction. since the loss function is defined based on w and b
        #using the computation graph of loss the grad of w and b will be computed
        loss.backward()
        with torch.no_grad():
            #We do not want the gradient descent compute to be part of the computation graph, hence
            #we call no_grad()
            #Gradient descent step
            b -= learning_rate * b.grad
            w -= learning_rate * w.grad
            #revert grad to zero that it will not accumulate across epochs
            b.grad.zero_()
            w.grad.zero_()
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

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

def linear_reg_using_pytorch_api(X_train, X_valid, X_test, y_train, y_valid, y_test):
    torch.manual_seed(42)  # to get reproducible results
    n_features = X_train.shape[1]  # there are 8 input features

    model = nn.Linear(in_features=n_features, out_features=1)
    #parameters - weights and bias is randomly initialised
    print("weights initialized ", model.weight)
    print("bias initialised ", model.bias)
    #We will use the stochastic gradient descent for optimisation of the loss function
    #Since we are using linear regression will be mean square error
    learning_rate = 0.4
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()

    n_epochs = 20
    for epoch in range(n_epochs):
        #forward prop
        y_pred = model(X_train)
        #compute loss
        loss = mse(y_pred, y_train)
        #backward prop
        loss.backward()
        #SGD
        optimizer.step()
        #zero out grads
        optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

def multi_layer_preceptron_using_gpu(X_train, X_valid, X_test, y_train, y_valid, y_test, device):
    torch.manual_seed(42)
    metric = torchmetrics.MeanSquaredError(squared=False).to(device)
    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    n_features = X_train.shape[1]  # there are 8 input features

    #Use the Sequential module to compose the multi layer perceptron with three hidden layer and one o/p layer
    model = nn.Sequential(
        #Hidden layer 1. Takes input(8 x m) where there are n number of samples. Number of output from the
        # first hidden layer depends on how many nodes does the first layer which is hyper paramenter which could be
        #tuned. Here we will use 50
        nn.Linear(n_features, 50),
        #We are using ReLU (Rectified Linear Unit) as the activation for the first hidden layer
        nn.ReLU(),
        #Second hidden layer will have 40 nodes, so output
        nn.Linear(50, 40),
        #Second  layer activation
        nn.ReLU(),
        #Third layer with ReLU activation
        nn.Linear(40, 30), nn.ReLU(),
        #o/p layer
        nn.Linear(30, 1)
    )

    #implement mini batch, where we will use Dataloader for creating batching from training data with shuffling
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    #data loader for the validation set to evaluate the model
    valid_dataset = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=32)

    #we will run the model in GPU
    model.to(d)
    learning_rate = 0.02
    #optimiser is Stochastic Gradient Descent without momentum
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    mse = nn.MSELoss()
    n_epochs = 20
    model.train()

    #You can see the MLP us performing much better than single perceptron model we saw earlier
    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        #creating mini batch and train using that in each epoch than using the whole
        #training data set
        for X_batch, y_batch in train_loader:
            #switch to training mode
            model.train()
            #move tensors to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            #compute the MSE loss
            loss = mse(y_pred, y_batch)
            #add losses for each mini batch
            total_loss += loss.item()
            #back prop
            loss.backward()
            #compute grads
            optimizer.step()
            optimizer.zero_grad()
            #update the values in every iteration for the metric compute
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(train_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        #evaluate the metric for mini batch descent using the validation set
        history["valid_metrics"].append(
            evaluate_tm(model, valid_loader, metric, d).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")

    #plot the training metric and see whether the model is overfitting
    plt.plot(np.arange(n_epochs) + 0.5, history["train_metrics"], ".--",
             label="Training")
    plt.plot(np.arange(n_epochs) + 1.0, history["valid_metrics"], ".-",
             label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.grid()
    plt.title("Learning curves")
    plt.axis([0.5, 20, 0.4, 1.0])
    plt.legend()
    plt.show()
    #You can see the validation and training metrics are more or less the same
    # and reducing with more epochs

    return history

#class derives from nn.Module
class nn_with_multi_inputs(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        #deep nn with three layers
        self.deep_stack = nn.Sequential(
            #n_features - 2 because remaining 2 features are fed directly into o/p layer
            nn.Linear(n_features - 2, 50), nn.ReLU(),
            nn.Linear(50, 40), nn.ReLU(),
            nn.Linear(40, 30), nn.ReLU(),
        )
        self.output_layer = nn.Linear(30 + 5, 1)

    def forward(self, X_wide, X_deep):
        deep_output = self.deep_stack(X_deep)
        #o/p of deep with three hidden layers and
        wide_and_deep = torch.concat([X_wide, deep_output], dim=1)
        return self.output_layer(wide_and_deep)

def evaluate_multi_in(model, data_loader, metric,d):
    #switch to eval mode
    model.eval()
    metric.reset()  # reset the metric at the beginning
    with torch.no_grad():
        for X_batch_wide, X_batch_deep, y_batch in data_loader:
            X_batch_wide = X_batch_wide.to(d)
            X_batch_deep = X_batch_deep.to(d)
            y_batch = y_batch.to(d)
            #forward prop
            y_pred = model(X_batch_wide, X_batch_deep)
            #add predictions for compute for each iteration of mini batch
            metric.update(y_pred, y_batch)  # update it at each iteration
    return metric.compute()  # compute the final result at the end

def get_data_loaders(X_train, X_valid, X_test, y_train, y_valid, y_test):
    #we have two tensors in training with different set of features since it is sliced
    train_data_wd = TensorDataset(X_train[:, :5], X_train[:, 2:], y_train)
    train_loader_wd = DataLoader(train_data_wd, batch_size=32, shuffle=True)
    valid_data_wd = TensorDataset(X_valid[:, :5], X_valid[:, 2:], y_valid)
    valid_loader_wd = DataLoader(valid_data_wd, batch_size=32)
    test_data_wd = TensorDataset(X_test[:, :5], X_test[:, 2:], y_test)
    test_loader_wd = DataLoader(test_data_wd, batch_size=32)
    return train_loader_wd, valid_loader_wd, test_loader_wd

def train_multi_in(model, optimizer,  metric,
                   t_loader, v_loader, n_epochs, d):

    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}

    for epoch in range(n_epochs):
        total_loss = 0.
        metric.reset()
        mse = nn.MSELoss()
        #X_batch_inputs have both deep and wide inputs. GD in mini batches
        for *X_batch_inputs, y_batch in t_loader:
            #switch to train mode
            model.train()
            #send tensors to gpu
            X_batch_inputs = [X.to(d) for X in X_batch_inputs]
            y_batch = y_batch.to(d)
            #Both inputs given and model will concat deep nn o/p with wide data set
            #forward prop
            y_pred = model(*X_batch_inputs)
            #loss compute using MSE
            loss = mse(y_pred, y_batch)
            total_loss += loss.item()
            #backard prop
            loss.backward()
            #Gradient descent of mini batch
            optimizer.step()
            optimizer.zero_grad()
            metric.update(y_pred, y_batch)
        mean_loss = total_loss / len(t_loader)
        history["train_losses"].append(mean_loss)
        history["train_metrics"].append(metric.compute().item())
        history["valid_metrics"].append(
            #evaluate using validation set
            evaluate_multi_in(model, v_loader, metric, d).item())
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.4f}, "
              f"valid metric: {history['valid_metrics'][-1]:.4f}")
    return history


if __name__ == '__main__':
    check_cuda_available()
    d = check_which_gpu()
    compare_cpu_vs_gpu(d)
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_housing_data()
    linear_reg_using_torch_auto_diff(X_train, X_valid, X_test, y_train, y_valid, y_test)
    linear_reg_using_pytorch_api(X_train, X_valid, X_test, y_train, y_valid, y_test)
    multi_layer_preceptron_using_gpu(X_train, X_valid, X_test, y_train, y_valid, y_test, d)

    #multi input nn
    torch.manual_seed(42)
    learning_rate = 0.01
    n_features = X_train.shape[1]
    n_epochs = 20
    model = nn_with_multi_inputs(n_features).to(d)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    rmse = torchmetrics.MeanSquaredError(squared=False).to(d)
    tr_loader, v_loader, te_loader = get_data_loaders(X_train, X_valid, X_test, y_train, y_valid, y_test)
    history = train_multi_in(model, optimizer, rmse, tr_loader, v_loader, n_epochs,d)