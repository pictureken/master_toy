import matplotlib.pyplot as plt
import torchvision
from PIL import Image


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def display_image(image):
    plt.imshow(image)
    plt.show()


def main():
    transform = torchvision.transforms.RandAugment()
    image = load_image("./otsuka.png")
    for _ in range(20):
        print(transform)
        transformed_image = transform(image)
        display_image(transformed_image)
    return


if __name__ == "__main__":
    main()
