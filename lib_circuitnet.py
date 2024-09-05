import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math

# ClipSin activation function as described in the paper
class ClipSin(nn.Module):
    def __init__(self, A=3.388235):
        super(ClipSin, self).__init__()
        self.A = A

    def forward(self, x):
        return torch.where(x < -3, -self.A + 0.01 * (x + 3),
                           torch.where(x > 3, self.A + 0.01 * (x - 3), self.A * torch.sin(math.pi / 6 * x)))

# CircuitMotifUnit with simplified attention mechanism and weight initialization
class CircuitMotifUnit(nn.Module):
    def __init__(self, num_neurons):
        super(CircuitMotifUnit, self).__init__()
        self.num_neurons = num_neurons
        self.W_ij = nn.Parameter(torch.randn(num_neurons, num_neurons))
        self.Vi = nn.Parameter(torch.randn(num_neurons))  
        self.Ki = nn.Parameter(torch.randn(num_neurons))  
        self.Qi = nn.Parameter(torch.randn(num_neurons))  
        self.W_ijk = nn.Parameter(torch.randn(num_neurons, num_neurons, num_neurons))
        self.activation = ClipSin()

        # Initialize weights
        nn.init.xavier_uniform_(self.W_ij)
        nn.init.xavier_uniform_(self.W_ijk)

    def forward(self, X):
        batch_size = X.size(0)
        num_features = X.size(1)

        # Linear transformation
        W_ij_expanded = self.W_ij.unsqueeze(0).expand(batch_size, -1, -1)
        X_expanded = X.unsqueeze(2)
        X_linear = torch.bmm(W_ij_expanded, X_expanded).squeeze(2)

        # Attention mechanism (corrected)
        QX = torch.matmul(X, self.Qi)
        KX = torch.matmul(X, self.Ki)
        attention_weights = torch.softmax(QX.unsqueeze(1) @ KX.unsqueeze(0), dim=-1)
        X_attention = attention_weights @ X  # Shape (batch_size, num_features)
        X_attention = X_attention @ self.Vi.unsqueeze(1)  # Shape (batch_size, 1)
        X_attention = X_attention.expand(-1, num_features)  # Expand back to (batch_size, num_features)

        # Product term (quadratic interaction)
        X_product = 0.5 * torch.einsum('ijk,bj,bk->bi', self.W_ijk, X, X)
        
        return self.activation(X_linear + X_attention + X_product)

# CircuitNetLayer
class CircuitNetLayer(nn.Module):
    def __init__(self, num_cm_units, num_neurons, num_iterations=2, sparsity_level=0.5):
        super(CircuitNetLayer, self).__init__()
        self.num_cm_units = num_cm_units
        self.num_neurons = num_neurons
        self.num_iterations = num_iterations  
        self.sparsity_level = sparsity_level  
        self.cm_units = nn.ModuleList([CircuitMotifUnit(num_neurons) for _ in range(num_cm_units)])
        self.W_inter = nn.Parameter(torch.randn(num_cm_units, num_cm_units))

    def forward(self, X_split):
        cmu_outputs = [cmu(X_split[i]) for i, cmu in enumerate(self.cm_units)]
        for _ in range(self.num_iterations):
            cmu_outputs_concat = torch.cat(cmu_outputs, dim=1)
            cmu_outputs_concat = torch.stack(torch.split(cmu_outputs_concat, self.num_neurons, dim=1), dim=1)
            inter_cmu_signals = torch.matmul(self.W_inter, cmu_outputs_concat)
            cmu_outputs = torch.unbind(inter_cmu_signals, dim=1)
        
        batch_size = inter_cmu_signals.size(0)
        inter_cmu_signals = inter_cmu_signals.view(batch_size, -1)
        return inter_cmu_signals

# CircuitNet with pre-defined projection layers
class CircuitNet(nn.Module):
    def __init__(self, num_layers, num_cm_units, num_neurons, input_image_size=(1, 28, 28)):
        super(CircuitNet, self).__init__()
        
        # Convolution and pooling layers
        self.conv = nn.Conv2d(in_channels=input_image_size[0], out_channels=32, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.num_cm_units = num_cm_units
        # Calculate the output size after conv and pooling layers
        conv_output_size = self._get_conv_output_size(input_image_size)

        # Linear projection to map convolutional output to the required neuron count per CMU (num_neurons)
        self.feature_projections = nn.ModuleList([
            nn.Linear(conv_output_size // num_cm_units, num_neurons) 
            for _ in range(num_cm_units)
        ])
        
        # Define CircuitNet layers
        self.layers = nn.ModuleList([CircuitNetLayer(num_cm_units, num_neurons) for _ in range(num_layers)])
        
        # Feature aggregation and final classification
        self.fc_aggregate = nn.Linear(num_neurons * num_cm_units, num_neurons * num_cm_units // 2)
        self.fc = nn.Linear(num_neurons * num_cm_units // 2, 10)

    def _get_conv_output_size(self, input_image_size):
        # Helper function to compute the size of the output after conv and pooling layers
        dummy_input = torch.ones(1, *input_image_size)
        conv_output = self.pool(self.conv(dummy_input))
        return int(torch.prod(torch.tensor(conv_output.size()[1:])))  # Flattened size after conv+pool

    def forward(self, X):
        batch_size = X.shape[0]
        X = self.pool(F.relu(self.conv(X)))  # Apply convolution and pooling layers
        X = X.view(batch_size, -1)  # Flatten the conv output

        # Split the flattened conv output into chunks for each CMU
        X_split = torch.split(X, X.size(1) // self.num_cm_units, dim=1)

        # Apply the pre-defined projection layers to each chunk
        X_split = [proj(chunk) for proj, chunk in zip(self.feature_projections, X_split)]

        # Pass through CircuitNet layers
        for layer in self.layers:
            X = layer(X_split)

        X = F.relu(self.fc_aggregate(X))  # Feature aggregation
        X = self.fc(X)  # Final classification
        return X