import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from lib_circuitnet import CircuitNet

# Dataset setup
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,0.1307, 0.1307), (0.3081,0.3081,0.3081))
])

"""
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
"""
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparametres a optimiser avec l'algorithme genetique
hyperparameter_choices = {
    "num_layers": [1, 2, 3, 5],
    "num_cm_units": [4, 8, 16, 32],
    "num_neurons": [16, 32, 64, 128],
    "learning_rate": [0.001, 0.0001, 0.00001],
    "batch_size": [32, 64, 128, 256]
}

# Generer un ensemble aleatoire d'hyperparametres
def create_random_hyperparameters():
    return {
        "num_layers": random.choice(hyperparameter_choices["num_layers"]),
        "num_cm_units": random.choice(hyperparameter_choices["num_cm_units"]),
        "num_neurons": random.choice(hyperparameter_choices["num_neurons"]),
        "learning_rate": random.choice(hyperparameter_choices["learning_rate"]),
        "batch_size": random.choice(hyperparameter_choices["batch_size"])
    }

# Fonction d'evaluation d'un modele avec un ensemble de donnees
def evaluate_model(model, device, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = 100. * correct / total
    return accuracy

# Algorithme genetique pour optimiser les hyperparametres
def genetic_algorithm(population_size=10, generations=5):
    # Initialisation de la population
    population = [create_random_hyperparameters() for _ in range(population_size)]
    
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        # evaluation de chaque individu
        scores = []
        for individual in population:
            # Creation du modele avec les hyperparametres actuels
            model = CircuitNet(individual["num_layers"], individual["num_cm_units"], individual["num_neurons"], input_image_size=(3, 32, 32)).to(device)
            optimizer = optim.Adam(model.parameters(), lr=individual["learning_rate"])
            criterion = nn.CrossEntropyLoss()
            
            # Ajuster la taille de batch du DataLoader
            train_loader = DataLoader(dataset=train_dataset, batch_size=individual["batch_size"], shuffle=True)
            
            # Entrainement rapide sur 2 epoques
            model.train()
            for epoch in range(1, 3):  # 2 epochs pour une evaluation rapide
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # evaluer l'accuracy sur le test set
            accuracy = evaluate_model(model, device, test_loader, criterion)
            scores.append((individual, accuracy))
            print(f"Hyperparameters: {individual}, Accuracy: {accuracy:.2f}%")
        
        # Selection des meilleurs individus
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        selected_individuals = scores[:population_size // 2] # pression de selection a 0.5 
        
        # Croisement et mutation pour la generation suivante
        next_population = []
        while len(next_population) < population_size:
            # Croisement
            parent1 = random.choice(selected_individuals)[0]
            parent2 = random.choice(selected_individuals)[0]
            child = {
                "num_layers": random.choice([parent1["num_layers"], parent2["num_layers"]]),
                "num_cm_units": random.choice([parent1["num_cm_units"], parent2["num_cm_units"]]),
                "num_neurons": random.choice([parent1["num_neurons"], parent2["num_neurons"]]),
                "learning_rate": random.choice([parent1["learning_rate"], parent2["learning_rate"]]),
                "batch_size": random.choice([parent1["batch_size"], parent2["batch_size"]])
            }
            
            # Mutation aleatoire
            if random.random() < 0.1:  # 10% de chances de mutation
                hyperparam_to_mutate = random.choice(list(hyperparameter_choices.keys()))
                child[hyperparam_to_mutate] = random.choice(hyperparameter_choices[hyperparam_to_mutate])
            
            next_population.append(child)
        
        population = next_population

# Lancer l'algorithme genetique
genetic_algorithm(population_size=12, generations=8)
