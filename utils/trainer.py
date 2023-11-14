import torch
import torch.nn.functional as F


class Trainer:
    def __init__(
        self,
        device,
        model: str,
        loss_function,
        optimizer=None,
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.criterion = loss_function
        self.optimizer = optimizer
        self.mse = torch.nn.MSELoss(size_average=None, reduce=None, reduction="mean")

    def train(self, train_loader):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for _, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs = F.softmax(outputs, dim=1)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        train_loss = train_loss / total
        train_accuracy = correct / total

        return (train_loss, train_accuracy, self.model)

    def test(self, test_loader, id: bool = True):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        outputs_sum = (
            torch.Tensor(len(test_loader) * test_loader.batch_size, 3)
            .zero_()
            .to(self.device)
        )
        # In-Distribution
        if id:
            with torch.no_grad():
                for _, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    outputs_sum[total : (total + inputs.size(0)), :] += outputs
                    outputs = F.softmax(outputs, dim=1)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                    total += targets.size(0)

                test_loss = test_loss / total
                test_accuracy = correct / total

            return (test_loss, test_accuracy, outputs_sum)
        # Out-of-Distribution
        else:
            with torch.no_grad():
                for _, inputs in enumerate(test_loader):
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    outputs = F.softmax(outputs, dim=1)
                    outputs_sum[total : (total + inputs.size(0)), :] += outputs
                    total += inputs.size(0)

            return outputs_sum
