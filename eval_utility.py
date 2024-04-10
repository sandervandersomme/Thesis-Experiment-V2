import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import visualise

class ClassifierRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 10, num_of_layers: int = 1):
        super(ClassifierRNN, self).__init__()

        self.rnn = nn.GRU(input_dim, hidden_dim, num_of_layers, batch_first=True)
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, data: torch.Tensor):
        output, _ = self.rnn(data)
        return self.output_layer(output[:, -1, :])
    
    def train_and_evaluate(self, train_data, test_data, learning_rate, epochs):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Initialize data loaders
        train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

        # Initialise arrays for loss and accuracy tracking
        losses = torch.zeros(epochs)
        accuracies = torch.zeros(epochs)

        # Train loop
        for epoch in range(epochs):
            self.train()
            for i, (sequences, labels) in enumerate(train_loader):
                outputs = self(sequences)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses[epoch] = loss.item()

            # Evaluate the model
            self.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for i, (sequences, labels) in enumerate(test_loader):
                    outputs = self(sequences)
                    predicted_labels = (outputs > 0.5).float()
                    correct += (predicted_labels == labels).sum().item()
                    total += len(sequences)
                    
            accuracy = 100 * correct / total
            accuracies[epoch] = accuracy
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}, Accuracy:: {accuracy:.2f}, correct: {correct}/{total}")

        visualise(epochs, losses, "Loss", "Epochs", "Loss eval-classifier", "Loss-classifier")
        visualise(epochs, accuracies, "Accuracy", "Epochs", "Accuracy eval-classifier", "Accuracy-classifier")

    

def classify_real_synthetic(train_data, test_data, learning_rate, epochs):
    print("Classfying real and synthetic samples")
    # Initialize model, loss function and optimizer
    number_of_features = train_data.dataset.tensors[0].shape[2]
    model = ClassifierRNN(number_of_features)
    model.train_and_evaluate(train_data, test_data, learning_rate, epochs)