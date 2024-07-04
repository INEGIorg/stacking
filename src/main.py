from align import align_images, load_images, save_image, stack_images
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    dir = Path(args.path)
    files = list(dir.glob("*.png"))

    if len(files) < 1:
        raise Exception("Not enough images")

    images = load_images(files)
    images = align_images(images)
    image = stack_images(images)
    save_image("result.png", image)


if __name__ == "__main__":
    main()
