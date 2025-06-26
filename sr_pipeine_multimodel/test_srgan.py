import argparse

import torch

from configs import Config
from metrics.calculator import MetricCalculator
from models.srgan_components import Generator


def test_single_image(checkpoint_path, lr_path, hr_path=None):
    # Load model
    config = Config()
    model = Generator(config).eval()
    model.load_state_dict(torch.load(checkpoint_path)["generator"])

    # Process inputs
    lr = torch.load(lr_path.replace(".png", ".pt"))
    if hr_path:
        hr = torch.load(hr_path.replace(".png", ".pt"))

    # Generate
    with torch.no_grad():
        fake = model(lr.unsqueeze(0), get_text_embedding(lr_path))

    # Calculate metrics
    if hr_path:
        metrics = MetricCalculator("cpu")(fake.squeeze(), hr)
        print(f"Metrics: {metrics}")

    # Save output
    torch.save(fake.squeeze(), "output.pt")
    return fake


def get_text_embedding(image_path):
    # Implement caption lookup from CSV
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lr", required=True)
    parser.add_argument("--hr", default=None)
    args = parser.parse_args()

    test_single_image(args.checkpoint, args.lr, args.hr)
