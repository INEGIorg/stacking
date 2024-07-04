from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import cv2

from align import align_images, stack_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    image_folder = Path(args.path)
    image_files = list(image_folder.glob("*.png"))

    with ThreadPoolExecutor() as executor:
        images = list(executor.map(lambda f: cv2.imread(str(f)), image_files))

    images = align_images(images)
    image = stack_images(images)
    cv2.imwrite(f"{args.path}_stacked.png", image)


if __name__ == "__main__":
    main()
