"""
Script to prepare and verify downloaded datasets
"""

import os
import json
from pathlib import Path
from collections import Counter


def check_coco(data_dir='data'):
    """Check COCO dataset"""
    print("\n" + "="*60)
    print("CHECKING COCO DATASET")
    print("="*60)

    coco_dir = Path(data_dir) / 'coco'

    # Check directories
    required_dirs = [
        coco_dir / 'images' / 'train2017',
        coco_dir / 'images' / 'val2017',
        coco_dir / 'annotations'
    ]

    for dir_path in required_dirs:
        if dir_path.exists():
            if 'images' in str(dir_path):
                num_images = len(list(dir_path.glob('*.jpg')))
                print(f"✓ {dir_path.name}: {num_images:,} images")
            else:
                num_files = len(list(dir_path.glob('*.json')))
                print(f"✓ {dir_path.name}: {num_files} annotation files")
        else:
            print(f"✗ {dir_path.name}: NOT FOUND")
            return False

    # Check annotations
    ann_files = [
        'captions_train2017.json',
        'captions_val2017.json'
    ]

    for ann_file in ann_files:
        ann_path = coco_dir / 'annotations' / ann_file
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                data = json.load(f)
            print(f"  - {ann_file}: {len(data['images'])} images, {len(data['annotations'])} captions")
        else:
            print(f"✗ {ann_file}: NOT FOUND")
            return False

    print("\n✓ COCO dataset is ready!")
    return True


def check_vqa(data_dir='data'):
    """Check VQA dataset"""
    print("\n" + "="*60)
    print("CHECKING VQA DATASET")
    print("="*60)

    vqa_dir = Path(data_dir) / 'vqa'

    # Look for VQA files (flexible naming)
    required_files = {
        'train_questions': ['*Questions*Train*.json', '*questions*train*.json'],
        'train_annotations': ['*Annotations*Train*.json', '*annotations*train*.json'],
        'val_questions': ['*Questions*Val*.json', '*questions*val*.json'],
        'val_annotations': ['*Annotations*Val*.json', '*annotations*val*.json']
    }

    found_files = {}
    for file_type, patterns in required_files.items():
        found = False
        for pattern in patterns:
            matches = list(vqa_dir.glob(pattern))
            if matches:
                found_files[file_type] = matches[0]
                found = True
                break

        if found:
            print(f"✓ {file_type}: {found_files[file_type].name}")

            # Load and check
            try:
                with open(found_files[file_type], 'r') as f:
                    data = json.load(f)
                if 'questions' in file_type:
                    print(f"  - {len(data['questions'])} questions")
                else:
                    print(f"  - {len(data['annotations'])} annotations")
            except Exception as e:
                print(f"  ⚠ Error reading file: {e}")
        else:
            print(f"✗ {file_type}: NOT FOUND")
            print(f"  Expected patterns: {patterns}")
            return False

    print("\n✓ VQA dataset is ready!")
    return True


def build_answer_vocab(data_dir='data', top_k=3129):
    """Build answer vocabulary for VQA"""
    print("\n" + "="*60)
    print("BUILDING VQA ANSWER VOCABULARY")
    print("="*60)

    vqa_dir = Path(data_dir) / 'vqa'

    # Find training annotations
    ann_files = list(vqa_dir.glob('*Annotations*Train*.json'))
    if not ann_files:
        ann_files = list(vqa_dir.glob('*annotations*train*.json'))

    if not ann_files:
        print("✗ Training annotations not found")
        return

    print(f"Loading annotations from: {ann_files[0].name}")

    with open(ann_files[0], 'r') as f:
        data = json.load(f)

    # Count answers
    answer_counter = Counter()
    for ann in data['annotations']:
        for answer in ann['answers']:
            answer_counter[answer['answer'].lower()] += 1

    print(f"Total unique answers: {len(answer_counter)}")
    print(f"Creating vocabulary with top {top_k} answers...")

    # Get top-k answers
    top_answers = [ans for ans, _ in answer_counter.most_common(top_k)]

    # Create vocabulary
    answer_to_idx = {ans: idx for idx, ans in enumerate(top_answers)}
    idx_to_answer = {idx: ans for ans, idx in answer_to_idx.items()}

    # Save vocabulary
    vocab_path = vqa_dir / 'answer_vocab.json'
    with open(vocab_path, 'w') as f:
        json.dump({
            'answer_to_idx': answer_to_idx,
            'idx_to_answer': idx_to_answer,
            'num_answers': len(answer_to_idx)
        }, f, indent=2)

    print(f"✓ Answer vocabulary saved to: {vocab_path}")
    print(f"  Coverage: {sum(answer_counter[ans] for ans in top_answers) / sum(answer_counter.values()) * 100:.2f}%")

    # Show some examples
    print("\nTop 10 answers:")
    for i, (ans, count) in enumerate(answer_counter.most_common(10), 1):
        print(f"  {i}. {ans}: {count:,}")


def build_caption_vocab(data_dir='data', vocab_size=30000):
    """Build vocabulary from COCO captions"""
    print("\n" + "="*60)
    print("BUILDING CAPTION VOCABULARY")
    print("="*60)

    from data.tokenizer import SimpleTokenizer

    coco_dir = Path(data_dir) / 'coco'
    ann_path = coco_dir / 'annotations' / 'captions_train2017.json'

    if not ann_path.exists():
        print("✗ COCO training annotations not found")
        return

    print("Loading COCO captions...")
    with open(ann_path, 'r') as f:
        data = json.load(f)

    # Extract all captions
    captions = [ann['caption'] for ann in data['annotations']]
    print(f"Processing {len(captions):,} captions...")

    # Build tokenizer
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, max_length=77)
    tokenizer.build_vocab(captions)

    # Save tokenizer
    tokenizer_path = Path(data_dir) / 'tokenizer.json'
    tokenizer.save(str(tokenizer_path))

    print(f"✓ Tokenizer saved to: {tokenizer_path}")
    print(f"  Vocabulary size: {len(tokenizer)}")


def create_dataset_stats(data_dir='data'):
    """Create dataset statistics"""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    stats = {}

    # COCO stats
    coco_dir = Path(data_dir) / 'coco'
    if (coco_dir / 'annotations' / 'captions_train2017.json').exists():
        with open(coco_dir / 'annotations' / 'captions_train2017.json', 'r') as f:
            train_data = json.load(f)
        with open(coco_dir / 'annotations' / 'captions_val2017.json', 'r') as f:
            val_data = json.load(f)

        stats['coco'] = {
            'train_images': len(train_data['images']),
            'train_captions': len(train_data['annotations']),
            'val_images': len(val_data['images']),
            'val_captions': len(val_data['annotations'])
        }

    # VQA stats
    vqa_dir = Path(data_dir) / 'vqa'
    train_q = list(vqa_dir.glob('*Questions*Train*.json'))
    if train_q:
        with open(train_q[0], 'r') as f:
            train_questions = json.load(f)
        val_q = list(vqa_dir.glob('*Questions*Val*.json'))
        with open(val_q[0], 'r') as f:
            val_questions = json.load(f)

        stats['vqa'] = {
            'train_questions': len(train_questions['questions']),
            'val_questions': len(val_questions['questions'])
        }

    # Print stats
    print("\nCOCO Captions:")
    if 'coco' in stats:
        for key, value in stats['coco'].items():
            print(f"  {key}: {value:,}")

    print("\nVQA v2.0:")
    if 'vqa' in stats:
        for key, value in stats['vqa'].items():
            print(f"  {key}: {value:,}")

    # Save stats
    stats_path = Path(data_dir) / 'dataset_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Statistics saved to: {stats_path}")


def main():
    """Main preparation function"""
    print("""
    ============================================================
       Multi-Modal VLM Data Preparation Script
    ============================================================
    """)

    data_dir = Path('data')

    if not data_dir.exists():
        print("✗ Data directory not found!")
        print("Please run: python scripts/download_data.py")
        return

    # Check datasets
    coco_ok = check_coco(data_dir)
    vqa_ok = check_vqa(data_dir)

    if coco_ok and vqa_ok:
        # Build vocabularies
        build_answer_vocab(data_dir)
        build_caption_vocab(data_dir)

        # Create stats
        create_dataset_stats(data_dir)

        print("\n" + "="*60)
        print("✓ DATA PREPARATION COMPLETE!")
        print("="*60)
        print("\nYou can now train models:")
        print("  python examples/train_caption.py")
        print("  python scripts/train_all.py")
    else:
        print("\n✗ Some datasets are missing. Please download them first:")
        print("  python scripts/download_data.py")


if __name__ == "__main__":
    main()
