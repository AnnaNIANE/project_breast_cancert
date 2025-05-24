import torch
import tensorflow as tf
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import numpy as np
from models.cnn_torch import CNN_Torch
from models.cnn_tensorflow import create_cnn_tensorflow
from utils.preprocessing import get_data

class TrainerTorch:
    def __init__(self, model, train_dataloader, test_dataloader, lr, wd, epochs, device):
        self.epochs = epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = []
        self.train_loss = []

    def train(self, save_path="anna_model.torch"):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)
            for batch in progress_bar:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                _, preds = outputs.max(1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
                batch_accuracy = 100.0 * (preds == labels).sum().item() / labels.size(0)
                average_accuracy = 100.0 * total_correct / total_samples
                average_loss = total_loss / (total_samples / self.train_dataloader.batch_size)
                progress_bar.set_postfix({
                    'Batch Acc': f'{batch_accuracy:.2f}%',
                    'Avg Acc': f'{average_accuracy:.2f}%',
                    'Loss': f'{average_loss:.4f}'
                })
            self.train_acc.append(average_accuracy)
            self.train_loss.append(average_loss)
        torch.save(self.model.state_dict(), save_path)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0
        for inputs, labels in tqdm(self.test_dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            _, preds = outputs.max(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples
        print(f"\nPyTorch Test Accuracy: {accuracy:.2f}%  |  Test Loss: {avg_loss:.4f}")
        return accuracy, avg_loss

class TrainerTensorFlow:
    def __init__(self, model, train_dataloader, test_dataloader, lr, epochs):
        self.epochs = epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        self.train_acc = []
        self.train_loss = []

    def train(self, save_path="anna_model.tensorflow"):
        for epoch in range(self.epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)
            for inputs, labels in progress_bar:
                inputs, labels = inputs.numpy(), labels.numpy()
                history = self.model.train_on_batch(inputs, labels)
                total_loss += history[0] * inputs.shape[0]
                total_correct += np.sum(np.argmax(self.model.predict(inputs, verbose=0), axis=1) == labels)
                total_samples += inputs.shape[0]
                batch_accuracy = 100.0 * np.sum(np.argmax(self.model.predict(inputs, verbose=0), axis=1) == labels) / inputs.shape[0]
                average_accuracy = 100.0 * total_correct / total_samples
                average_loss = total_loss / total_samples
                progress_bar.set_postfix({
                    'Batch Acc': f'{batch_accuracy:.2f}%',
                    'Avg Acc': f'{average_accuracy:.2f}%',
                    'Loss': f'{average_loss:.4f}'
                })
            self.train_acc.append(average_accuracy)
            self.train_loss.append(average_loss)
        self.model.save(save_path)

    def evaluate(self):
        total_correct = 0
        total_samples = 0
        total_loss = 0
        for inputs, labels in tqdm(self.test_dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.numpy(), labels.numpy()
            loss, acc = self.model.evaluate(inputs, labels, verbose=0)
            total_loss += loss * inputs.shape[0]
            total_samples += inputs.shape[0]
            total_correct += acc * inputs.shape[0]
        avg_loss = total_loss / total_samples
        accuracy = 100.0 * total_correct / total_samples
        print(f"\nTensorFlow Test Accuracy: {accuracy:.2f}%  |  Test Loss: {avg_loss:.4f}")
        return accuracy, avg_loss

def main():
    parser = argparse.ArgumentParser(description="Train a CNN model for breast cancer classification")
    parser.add_argument('--framework', type=str, choices=['pytorch', 'tensorflow'], required=True,
                        help="Choose the framework: 'pytorch' or 'tensorflow'")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--wd', type=float, default=0.0001, help="Weight decay (PyTorch only)")
    parser.add_argument('--cuda', action='store_true', help="Use GPU if available")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    train_loader, test_loader, classes = get_data()

    if args.framework == 'pytorch':
        model = CNN_Torch(num_classes=4).to(device)
        trainer = TrainerTorch(model, train_loader, test_loader, args.lr, args.wd, args.epochs, device)
        trainer.train(save_path="anna_model.torch")
        trainer.evaluate()
    else:
        model = create_cnn_tensorflow()
        trainer = TrainerTensorFlow(model, train_loader, test_loader, args.lr, args.epochs)
        trainer.train(save_path="anna_model.tensorflow")
        trainer.evaluate()

if __name__ == "__main__":
    main()