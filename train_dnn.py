import argparse
import os

import torch
import torch.nn as nn
from torchvision import transforms

import utils


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = utils.data.GenerateToyDataset(
        transform,
        num_samples=10000,
        num_classes=args.num_classes,
        center=(0.5, 0.5),
        train=True,
        noise=0.4,
        dim=args.dim,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_dataset = utils.data.GenerateToyDataset(
        transform,
        num_samples=1000,
        num_classes=args.num_classes,
        center=(0.5, 0.5),
        train=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=2
    )
    model = utils.model.FCNet(
        in_features=args.dim,
        hidden_size=args.hidden_size,
        out_features=args.num_classes,
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = utils.trainer.Trainer(device, model, loss_function, optimizer)
    # train
    for i in range(args.epoch):
        train_loss, train_acc, model = trainer.train(train_loader)
        print(f"Train@{i+1} loss {train_loss}," f"accuracy {train_acc:.2%}")
        test_loss, test_acc, _ = trainer.test(test_loader)
        print(f"Test@{i+1} loss {test_loss}," f"accuracy {test_acc:.2%}")

    output_model_path = f"./outputs/pretrain_dnn/{args.hidden_size}"
    os.makedirs(output_model_path, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(output_model_path, str(args.trial) + ".pt"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()
    main(args)
