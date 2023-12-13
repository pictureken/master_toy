import argparse
import os

import torch
import torch.nn as nn
from torchvision import transforms

import utils


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])

    # grid data loader
    grid_test_dataset = utils.data.GenerateGridDataset()
    grid_test_loader = torch.utils.data.DataLoader(
        grid_test_dataset, batch_size=100, shuffle=False, num_workers=2
    )

    model = utils.model.FCNet(
        in_features=args.dim,
        hidden_size=args.hidden_size,
        out_features=args.num_classes,
    )

    output_model_path = f"./outputs/pretrain_dnn/{args.hidden_size}/"
    pretrained_model_path = os.path.join(output_model_path, str(args.trial) + ".pt")
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    loss_function = nn.CrossEntropyLoss()
    trainer = utils.trainer.Trainer(device, model, loss_function)

    # grid test
    outputs_sum = trainer.test(grid_test_loader, args.num_classes, id=False)
    output_tensor_path = f"./outputs/tensor/grid/{args.hidden_size}/"
    os.makedirs(output_tensor_path, exist_ok=True)
    torch.save(
        outputs_sum,
        os.path.join(output_tensor_path, str(args.trial) + ".pt"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--dim", type=int, default=2)
    args = parser.parse_args()
    main(args)
