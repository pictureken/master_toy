import argparse
import os

import torch
import torch.nn as nn
from torchvision import transforms

import utils


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = utils.data.GenerateToyDataset(
        transform,
        num_samples=1000,
        num_classes=args.num_classes,
        center=(0.5, 0.5),
        train=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=2
    )
    model = utils.model.NeuralNet(
        in_features=2, hidden_size=args.hidden_size, out_features=args.num_classes
    )
    output_model_path = "./outputs/pretrain/"
    pretrained_model_path = os.path.join(
        output_model_path, str(args.hidden_size) + ".pt"
    )
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    loss_function = nn.CrossEntropyLoss()
    trainer = utils.trainer.Trainer(device, model, loss_function)
    # test
    test_loss, test_acc, outputs_sum = trainer.test(test_loader)
    print(f"loss {test_loss}," f"accuracy {test_acc:.2%}")
    output_tensor_path = "./outputs/tensor/"
    os.makedirs(output_tensor_path, exist_ok=True)
    torch.save(
        outputs_sum,
        os.path.join(output_tensor_path, str(args.hidden_size) + ".pt"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--hidden_size", type=int, default=1000)
    args = parser.parse_args()
    main(args)
