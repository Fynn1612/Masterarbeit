# class for the custom neural network models for MC Dropout and Deep Ensembles
# method for the mse loss and heteroscedastic loss
# method for the training of the model
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# create a class for Neural Network with a custom architecture
class Custom_NN_Model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, do_rate, loss_type = None):
        """
        Neural Network model with a custom architecture.
        
        Args:
            input_dim:   number of input features
            hidden_dims: list of integers representing the number of neurons in each hidden layer e.g. [64, 128, 64, 32]
            output_dim:  number of output features (usually 1 for regression tasks)
            do_rate:     dropout rate for regularization
            loss_type:   type of loss function to use, either 'mse' for Mean Squared Error or 'heteroscedastic' for heteroscedastic regression
        """
        super(Custom_NN_Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.do_rate = do_rate
        self.loss_type = loss_type
        
        # create the layers of the model
        layers = []
        last_dim = input_dim
        for dim in hidden_dims: # iterate over the hidden dimensions, each dimension consists of a linear layer, a ReLU activation and a dropout layer
            layers.append(torch.nn.Linear(last_dim, dim)) # create a linear layer
            layers.append(torch.nn.ReLU())                # create a ReLU activation layer
            layers.append(torch.nn.Dropout(do_rate))      # create a dropout layer
            last_dim = dim                                     # update the last dimension to the current hidden layer dimension to match number of in- and outputs for the layers
        self.hidden_layers = torch.nn.Sequential(*layers) # create a sequential layer block for the hidden layers
        self.mean_layer = torch.nn.Linear(last_dim, output_dim) # create the output layer for the mean prediction for eather mse or heteroscedastic loss
        
        if loss_type == "heteroscedastic":
            # if heteroscedastic loss is used, we need two output layers
            self.var_layer = torch.nn.Linear(last_dim, output_dim)

        
    def forward(self, x):
        x = self.hidden_layers(x)  # pass the input through the hidden layers
        mean = self.mean_layer(x)      # get the mean prediction from the output layer

        if self.loss_type == "heteroscedastic":
            # if heteroscedastic loss is used, we need to pass the output through the variance layer
            log_var = self.var_layer(x)  
                        
            return mean, log_var
        
        return mean
    
# create a custom loss function for heteroscedastic regression
def heteroscedastic_loss(model, x, y ):
    """
    Heteroscedastic loss function for regression tasks.

    Args:
        model: The neural network model.
        x:     The input features.
        y:     The target values.

    Returns:
        loss:  The computed heteroscedastic loss."""
    mean, log_var =  model(x)
    precision = torch.exp(-1* log_var)  # precision = 1/var = exp(-log_var)
    loss = torch.mean(0.5 * precision * (y - mean) ** 2 + 0.5 * log_var)
    
    return loss

def mse_loss(model, x, y):
    """
    Mean Squared Error loss function for regression tasks.

    Args:
        model: The neural network model.
        x:     The input features.
        y:     The target values.

    Returns:    
        mse:   The computed Mean Squared Error loss.
    """
    # Mean Squared Error loss function
    output = model(x)
    loss = torch.nn.MSELoss(reduction='mean')
    mse = loss(output, y)
    
    return mse

# training functions for the model, optimizer Adam, loss function MSELoss, data loader for batching the data, early stopping
def train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, batch_size=128, 
                optimizer=None, n_epochs=1000,  patience=50, loss_type= 'mse', device= None):
    """
    Train a neural network model.
    
    Args:
        model: The neural network model to be trained.
        X_train_tensor: The matrix of features for the training data.
        y_train_tensor: The vector of target values for the training data.
        X_val_tensor: The matrix of features for the validation data.
        y_val_tensor: The vector of target values for the validation data.
        batch_size: The size of the batches for training.
        n_epochs: The number of epochs for training.
        patience: The number of epochs with no improvement after which training will be stopped.
        loss_type: The type of loss function to be used, e.g., 'mse' for Mean Squared Error or 'heteroscedastic' for heteroscedastic regression.
        device: The device to run the model on, e.g., 'cpu' or 'cuda'.
        model_type: The type of model, e.g., 'MC_Dropout' or 'Deep_Ensemble'.

    Returns:
        model: The trained neural network model.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Adam optimizer with weight decay for regularization
    if optimizer is None:  # If no optimizer is provided, create a new one
        optimizer = torch.optim.AdamW(params = model.parameters(), lr = 0.001, weight_decay=0.0001)  
        print("Using default AdamW optimizer with lr=0.01 and weight_decay=0.0001")

    if torch.cuda.is_available():
        X_train_tensor = X_train_tensor.cuda()
        X_val_tensor = X_val_tensor.cuda()
        y_train_tensor = y_train_tensor.cuda()
        y_val_tensor = y_val_tensor.cuda()
        model = model.cuda()

    # DataLoader for batching the data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Early Stopping values
    best_val_loss = np.inf
    epochs_no_improve = 0
    loss_history = []
    val_loss_history = []   

    for epoch in range(n_epochs):
        model.train()  # Set model to training mode                        
        batch_losses = []
        for X_batch, y_batch in train_loader:   # loop over all batches in the DataLoader
            X_batch = X_batch.to(device)                  # Move data to the device (GPU or CPU)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()               # Reset gradients

            # Forward pass
            # Depending on the loss type, we either use heteroscedastic loss or MSE
            if loss_type == 'mse' :
                loss = mse_loss(model, X_batch, y_batch)
                
            elif loss_type == 'heteroscedastic':
                loss = heteroscedastic_loss(model, X_batch, y_batch)
                
            loss.backward()                     # Backpropagation
            optimizer.step()                    # Update weights
            batch_losses.append(loss.item())   
        loss_history.append(np.mean(batch_losses))    # Save loss value

        # calculate validation loss
        model.eval()                            # Set model to evaluation mode
        with torch.no_grad():
            
            # Depending on the loss type, we either use heteroscedastic loss or MSE
            if loss_type == 'mse' :
                val_loss = mse_loss(model, X_val_tensor, y_val_tensor)
                
            elif loss_type == 'heteroscedastic':
                val_loss = heteroscedastic_loss(model, X_val_tensor, y_val_tensor)
                
            val_loss_history.append(val_loss.item())
            
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {np.mean(batch_losses):.4f}, Val Loss: {val_loss.item():.4f}, Best Val Loss: {best_val_loss:.4f}")
            
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}, Best Val Loss: {best_val_loss:.4f}")
                model.load_state_dict(best_model_state) # Load the best model state
                break  
    model.load_state_dict(best_model_state)  # Load the best model state
    return model

# function to create an ensemble of n models with the same architecture and optimizer
def create_ensemble(n, input_dim, hidden_dims, do_rate, loss_type, lr, weight_decay, output_dim=1):
    """
    Create an ensemble of n models with the same architecture and optimizer.

    Args:
        n: number of models in the ensemble
        input_dim: number of input features
        hidden_dims: list of integers representing the number of neurons in each hidden layer e.g.
        output_dim: number of output features 
        do_rate: dropout rate for regularization
        loss_type: type of loss function to use, either 'mse' for Mean Squared Error or 'heteroscedastic' for heteroscedastic regression
        lr: learning rate for the optimizer
        weight_decay: weight decay for regularization in the optimizer

    Returns:
        nets_ops: list of tuples containing the model and optimizer for each model in the ensemble
    """

    nets_ops = []

    for i in range(n):
        net = Custom_NN_Model(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, 
                              do_rate=do_rate, loss_type=loss_type)  # Create model instance
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)  # Create optimizer
        nets_ops.append((net, optimizer))

    return nets_ops

