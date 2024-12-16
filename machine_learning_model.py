# models.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from collections import OrderedDict
from datasets import download_iris, split_data_train_val_test
from sklearn.svm import SVC
import torch

# MLP Classifier
class ClassifierMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, layers: int = 5, hidden_dim: int = 25, act_fn=None):
        super(ClassifierMLP, self).__init__()
        act_fn = nn.ReLU() if act_fn is None else act_fn
        self.input_dim = input_dim
        self.num_classes = num_classes

        mlp_layers = OrderedDict()
        mlp_layers['input'] = nn.Linear(input_dim, hidden_dim)
        mlp_layers['input_act'] = act_fn
        for i in range(layers):
            mlp_layers[f'hidden_{i}'] = nn.Linear(hidden_dim, hidden_dim)
            mlp_layers[f'hidden_{i}_act'] = act_fn
        mlp_layers['output_lin'] = nn.Linear(hidden_dim, num_classes)
        mlp_layers['output'] = nn.Softmax(dim=-1)

        self.mlp = nn.Sequential(mlp_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def fit(self, X_train, Y_train, epochs=100, lr=0.001):
        X_train = torch.tensor(X_train, dtype=torch.float32) if not isinstance(X_train, torch.Tensor) else X_train
        Y_train = torch.tensor(Y_train, dtype=torch.long) if not isinstance(Y_train, torch.Tensor) else Y_train

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = criterion(outputs, Y_train)

            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, X_valid):
        X_valid = torch.tensor(X_valid, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X_valid)
            predictions = outputs.argmax(dim=1)
        return predictions


# Logistic Regression
class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear.out_features == 1:
            return torch.sigmoid(self.linear(x))
        return torch.softmax(self.linear(x), dim=-1)

    def fit(self, X_train: torch.Tensor, Y_train: torch.Tensor, lr=0.01, epochs=100, batch_size=32):
        X_train = torch.tensor(X_train, dtype=torch.float32) if not isinstance(X_train, torch.Tensor) else X_train
        Y_train = torch.tensor(Y_train, dtype=torch.long) if not isinstance(Y_train, torch.Tensor) else Y_train
        criterion = nn.BCELoss() if self.linear.out_features == 1 else nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        dataset = TensorDataset(X_train, Y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                batch_y = batch_y.float().view(-1, 1) if self.linear.out_features == 1 else batch_y
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(loader):.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            return (outputs >= 0.5).int() if self.linear.out_features == 1 else outputs.argmax(dim=1)


# SVM

class SVM:
    def __init__(self, input_dim, num_classes, C=1.0, kernel='linear'):
        """
        SVM Classifier using scikit-learn's SVC.
        
        Args:
            input_dim (int): Number of input features.
            num_classes (int): Number of output classes.
            C (float): Regularization parameter.
            kernel (str): Kernel type ('linear', 'rbf', 'poly', etc.).
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.C = C
        self.kernel = kernel
        self.model = SVC(C=self.C, kernel=self.kernel, probability=True)

    def fit(self, X_train, Y_train):
        """
        Train the SVM classifier on the given data.
        
        Args:
            X_train (ndarray): Training features.
            Y_train (ndarray): Training labels.
        """
        # Huấn luyện mô hình
        print("Training SVM...")
        self.model.fit(X_train, Y_train)
        print("SVM training completed.")

    def predict(self, X_valid):
        """
        Predict the class labels for the given input data.
        
        Args:
            X_valid (ndarray): Validation features.
        
        Returns:
            Tensor: Predicted class labels.
        """
        # Dự đoán nhãn
        predictions = self.model.predict(X_valid)
        return torch.tensor(predictions)

    def predict_proba(self, X_valid):
        """
        Predict the probabilities for each class.
        
        Args:
            X_valid (ndarray): Validation features.
        
        Returns:
            Tensor: Predicted probabilities for each class.
        """
        probabilities = self.model.predict_proba(X_valid)
        return torch.tensor(probabilities)


# if __name__ == "__main__":

#     X, Y =  download_iris()
#     X_train, Y_train, X_valid, Y_valid, X_test, Y_test = split_data_train_val_test(X, Y, train_size=0.6, valid_size=0.2)
#     input_dim = X_train.shape[1]
#     num_classes = len(set(Y_train))
#     # Logistic Regression Test
#     print("\nLogistic Regression:")
#     model_lr = LogisticRegression(input_dim=input_dim, num_classes=num_classes)
#     model_lr.fit(X_train, Y_train, epochs=1000)
#     predictions_lr = model_lr.predict(X_valid)
#     print("Predictions (Logistic Regression):", predictions_lr)

#     # MLP Test
#     print("\nMLP Classifier:")
#     model_mlp = ClassifierMLP(input_dim=input_dim, num_classes=num_classes)
#     model_mlp.fit(X_train, Y_train, epochs=100)
#     predictions_mlp = model_mlp.predict(X_valid)
#     print("Predictions (MLP):", predictions_mlp)

#     # SVM Test
#     print("\nSVM:")
#     model_svm = SVM(input_dim=input_dim, num_classes=num_classes)
#     model_svm.fit(X_train, Y_train)
#     predictions_svm = model_svm.predict(X_valid)
#     print("Predictions (SVM):", predictions_svm)
#     print('true prediction:', Y_valid)
