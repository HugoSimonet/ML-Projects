"""
Script to download datasets for Multi-Modal VLM
Downloads COCO Captions, VQA v2.0, and Flickr30K datasets
"""

import os
import urllib.request
import zipfile
import json
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_coco(data_dir='data'):
    """Download COCO Captions dataset"""
    print("\n" + "="*60)
    print("DOWNLOADING COCO CAPTIONS DATASET")
    print("="*60)

    coco_dir = Path(data_dir) / 'coco'
    images_dir = coco_dir / 'images'
    annotations_dir = coco_dir / 'annotations'

    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # URLs for COCO 2017
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }

    # Download train images (18GB)
    print("\n[1/3] Downloading training images (~18GB)...")
    train_zip = coco_dir / 'train2017.zip'
    if not (images_dir / 'train2017').exists():
        download_url(urls['train_images'], str(train_zip))
        print("Extracting training images...")
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
        os.remove(train_zip)
        print("✓ Training images downloaded and extracted")
    else:
        print("✓ Training images already exist")

    # Download validation images (1GB)
    print("\n[2/3] Downloading validation images (~1GB)...")
    val_zip = coco_dir / 'val2017.zip'
    if not (images_dir / 'val2017').exists():
        download_url(urls['val_images'], str(val_zip))
        print("Extracting validation images...")
        with zipfile.ZipFile(val_zip, 'r') as zip_ref:
            zip_ref.extractall(images_dir)
        os.remove(val_zip)
        print("✓ Validation images downloaded and extracted")
    else:
        print("✓ Validation images already exist")

    # Download annotations
    print("\n[3/3] Downloading annotations...")
    ann_zip = coco_dir / 'annotations_trainval2017.zip'
    if not (annotations_dir / 'captions_train2017.json').exists():
        download_url(urls['annotations'], str(ann_zip))
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall(coco_dir)
        os.remove(ann_zip)
        print("✓ Annotations downloaded and extracted")
    else:
        print("✓ Annotations already exist")

    print("\n✓ COCO dataset ready!")
    print(f"Location: {coco_dir.absolute()}")


def download_vqa(data_dir='data'):
    """Download VQA v2.0 dataset"""
    print("\n" + "="*60)
    print("DOWNLOADING VQA v2.0 DATASET")
    print("="*60)

    vqa_dir = Path(data_dir) / 'vqa'
    vqa_dir.mkdir(parents=True, exist_ok=True)

    # VQA v2.0 URLs
    base_url = 'https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa'

    files = {
        'train_questions': f'{base_url}/v2_Questions_Train_mscoco.zip',
        'train_annotations': f'{base_url}/v2_Annotations_Train_mscoco.zip',
        'val_questions': f'{base_url}/v2_Questions_Val_mscoco.zip',
        'val_annotations': f'{base_url}/v2_Annotations_Val_mscoco.zip',
    }

    for i, (name, url) in enumerate(files.items(), 1):
        print(f"\n[{i}/{len(files)}] Downloading {name}...")
        zip_path = vqa_dir / f'{name}.zip'

        # Check if already extracted
        json_name = name.replace('_', '_').replace('train', 'Train').replace('val', 'Val')
        json_path = vqa_dir / f'v2_mscoco_{json_name}.json'

        if not any(vqa_dir.glob(f'*{name.split("_")[0]}*{name.split("_")[1]}*.json')):
            try:
                download_url(url, str(zip_path))
                print(f"Extracting {name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(vqa_dir)
                os.remove(zip_path)
                print(f"✓ {name} downloaded and extracted")
            except Exception as e:
                print(f"⚠ Error downloading {name}: {e}")
                print(f"Please download manually from: {url}")
        else:
            print(f"✓ {name} already exists")

    print("\n✓ VQA dataset ready!")
    print(f"Location: {vqa_dir.absolute()}")
    print("\nNote: VQA uses COCO images, which should already be downloaded.")


def create_flickr30k_readme(data_dir='data'):
    """Create instructions for Flickr30K (requires registration)"""
    print("\n" + "="*60)
    print("FLICKR30K DATASET")
    print("="*60)

    flickr_dir = Path(data_dir) / 'flickr30k'
    flickr_dir.mkdir(parents=True, exist_ok=True)

    readme = """
# Flickr30K Dataset Download Instructions

The Flickr30K dataset requires registration and cannot be automatically downloaded.

## Steps to Download:

1. Go to: http://shannon.cs.illinois.edu/DenotationGraph/
2. Fill out the form to request access
3. Download the dataset (Flickr30K Images)
4. Extract the images to: {images_dir}
5. Download Karpathy split annotations from:
   https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
6. Extract and place in: {ann_dir}

## Alternative:

For training and testing purposes, you can start with just COCO and VQA datasets.
Flickr30K is optional and mainly used for additional evaluation.
    """.format(
        images_dir=flickr_dir / 'images',
        ann_dir=flickr_dir / 'annotations'
    )

    readme_path = flickr_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme)

    print(readme)
    print(f"\n✓ Instructions saved to: {readme_path.absolute()}")


def download_sample_data(data_dir='data'):
    """Download a small sample dataset for quick testing"""
    print("\n" + "="*60)
    print("CREATING SAMPLE DATASET FOR TESTING")
    print("="*60)

    sample_dir = Path(data_dir) / 'sample'
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Check if COCO is downloaded
    coco_dir = Path(data_dir) / 'coco'
    if not coco_dir.exists() or not (coco_dir / 'annotations' / 'captions_val2017.json').exists():
        print("⚠ COCO dataset not found. Please download COCO first.")
        return

    # Load COCO annotations
    with open(coco_dir / 'annotations' / 'captions_val2017.json', 'r') as f:
        coco_data = json.load(f)

    # Create small subset (100 images)
    print("Creating sample dataset with 100 images...")
    sample_images = coco_data['images'][:100]
    sample_image_ids = set([img['id'] for img in sample_images])
    sample_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in sample_image_ids]

    sample_data = {
        'images': sample_images,
        'annotations': sample_annotations,
        'info': coco_data['info'],
        'licenses': coco_data['licenses']
    }

    # Save sample annotations
    sample_ann_path = sample_dir / 'captions_sample.json'
    with open(sample_ann_path, 'w') as f:
        json.dump(sample_data, f)

    print(f"✓ Sample dataset created with {len(sample_images)} images")
    print(f"Location: {sample_dir.absolute()}")
    print("\nNote: Sample uses COCO validation images. No need to copy images.")


def main():
    """Main download function"""
    print("""
    ============================================================
       Multi-Modal VLM Dataset Download Script
    ============================================================
       This will download:
       1. COCO Captions 2017 (~20GB)
       2. VQA v2.0 (~1GB)
       3. Create sample dataset for testing

       Total download size: ~21GB
       Total disk space needed: ~25GB
    ============================================================
    """)

    # Ask for confirmation
    response = input("\nDo you want to proceed with downloading? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return

    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    try:
        # Download datasets
        download_coco(data_dir)
        download_vqa(data_dir)
        create_flickr30k_readme(data_dir)
        download_sample_data(data_dir)

        print("\n" + "="*60)
        print("✓ ALL DOWNLOADS COMPLETE!")
        print("="*60)
        print(f"\nData location: {data_dir.absolute()}")
        print("\nNext steps:")
        print("1. Run: python scripts/prepare_data.py")
        print("2. Run: python examples/train_caption.py")

    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
    except Exception as e:
        print(f"\n\n⚠ Error during download: {e}")
        print("You may need to download datasets manually.")


if __name__ == "__main__":
    main()
