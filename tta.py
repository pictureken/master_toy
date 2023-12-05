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
    grid_tta_dataset = utils.data.TTAGenerateGridDataset(
        num_samples=100, k=10, sigma=args.sigma
    )
    grid_tta_loader = torch.utils.data.DataLoader(
        grid_tta_dataset, batch_size=100, shuffle=False, num_workers=2
    )

    model = utils.model.NeuralNet(
        in_features=2,
        hidden_size=args.hidden_size,
        out_features=args.num_classes,
        batch_normalize=args.bn,
    )

    if args.bn:
        output_model_path = f"./outputs/pretrain/{args.hidden_size}bn/"
    else:
        output_model_path = f"./outputs/pretrain/{args.hidden_size}/"
    pretrained_model_path = os.path.join(output_model_path, str(args.trial) + ".pt")
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    loss_function = nn.CrossEntropyLoss()
    trainer = utils.trainer.Trainer(device, model, loss_function)

    # grid test
    outputs_sum = trainer.tta(grid_tta_loader)
    if args.bn:
        output_tensor_path = (
            f"./outputs/tensor/grid_tta_{args.sigma}/{args.hidden_size}bn/"
        )
    else:
        output_tensor_path = (
            f"./outputs/tensor/grid_tta_{args.sigma}/{args.hidden_size}/"
        )
    os.makedirs(output_tensor_path, exist_ok=True)
    torch.save(
        outputs_sum,
        os.path.join(output_tensor_path, str(args.trial) + ".pt"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=1000)
    parser.add_argument("--bn", action="store_true")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
