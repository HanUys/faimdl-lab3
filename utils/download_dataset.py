import os
import zipfile
import shutil
import urllib.request


DATA_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")
EXTRACTED_DIR = os.path.join(DATA_DIR, "tiny-imagenet-200")


def download_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(EXTRACTED_DIR):
        print("Dataset already exists. Skipping download.")
        return

    if not os.path.exists(ZIP_PATH):
        print("Downloading Tiny ImageNet dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("Download completed.")
    else:
        print("Zip file already exists. Skipping download.")

    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    print("Extraction completed.")


def organize_validation_set():
    val_dir = os.path.join(EXTRACTED_DIR, "val")
    images_dir = os.path.join(val_dir, "images")
    annotations_path = os.path.join(val_dir, "val_annotations.txt")
    output_dir = os.path.join(EXTRACTED_DIR, "val_images_by_class")

    if os.path.exists(output_dir):
        print("Validation set already organized.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("Organizing validation set into ImageFolder format...")

    with open(annotations_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            image_name = parts[0]
            class_id = parts[1]

            class_dir = os.path.join(output_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)

            src_path = os.path.join(images_dir, image_name)
            dst_path = os.path.join(class_dir, image_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

    print("Validation set organized.")


def main():
    download_dataset()
    organize_validation_set()


if __name__ == "__main__":
    main()