"""
Example: Inference demonstration for all tasks
"""

import torch
from PIL import Image
from models import VLMModel
from inference import ImageCaptioner, VQASystem, RetrievalSystem
from data import get_tokenizer, get_val_transforms


def demo_captioning():
    """Demo image captioning"""
    print("\n" + "=" * 50)
    print("IMAGE CAPTIONING DEMO")
    print("=" * 50)

    # Load model
    model = VLMModel(vision_encoder_type='resnet', language_encoder_type='custom')
    # model.load_state_dict(torch.load('checkpoint.pt')['model_state_dict'])

    tokenizer = get_tokenizer('simple')
    transform = get_val_transforms()

    captioner = ImageCaptioner(
        model=model,
        tokenizer=tokenizer,
        transform=transform,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Generate captions for sample images
    image_paths = ['examples/sample1.jpg', 'examples/sample2.jpg']

    for img_path in image_paths:
        try:
            image = Image.open(img_path)
            caption = captioner.generate(image)
            print(f"\nImage: {img_path}")
            print(f"Caption: {caption}")
        except FileNotFoundError:
            print(f"Sample image not found: {img_path}")
            print("Please provide your own test images")


def demo_vqa():
    """Demo visual question answering"""
    print("\n" + "=" * 50)
    print("VISUAL QUESTION ANSWERING DEMO")
    print("=" * 50)

    # Load model
    model = VLMModel(vision_encoder_type='resnet', language_encoder_type='custom')

    tokenizer = get_tokenizer('simple')
    transform = get_val_transforms()

    # Create dummy answer vocabulary
    answer_vocab = {i: f"answer_{i}" for i in range(100)}

    vqa_system = VQASystem(
        model=model,
        tokenizer=tokenizer,
        transform=transform,
        answer_vocab=answer_vocab,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Answer questions about sample image
    image_path = 'examples/sample1.jpg'
    questions = [
        "What color is the car?",
        "How many people are in the image?",
        "What is the weather like?"
    ]

    try:
        image = Image.open(image_path)

        for question in questions:
            answers = vqa_system.answer(image, question, top_k=3)
            print(f"\nQuestion: {question}")
            for i, ans in enumerate(answers):
                print(f"  {i + 1}. {ans['answer']} ({ans['confidence']:.4f})")

    except FileNotFoundError:
        print(f"Sample image not found: {image_path}")
        print("Please provide your own test images")


def demo_retrieval():
    """Demo cross-modal retrieval"""
    print("\n" + "=" * 50)
    print("CROSS-MODAL RETRIEVAL DEMO")
    print("=" * 50)

    # Load model
    model = VLMModel(vision_encoder_type='resnet', language_encoder_type='custom')

    tokenizer = get_tokenizer('simple')
    transform = get_val_transforms()

    retrieval = RetrievalSystem(
        model=model,
        tokenizer=tokenizer,
        transform=transform,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Index sample images
    print("\nIndexing images...")
    image_paths = [f'examples/sample{i}.jpg' for i in range(1, 6)]
    image_ids = [f'img_{i}' for i in range(1, 6)]

    try:
        images = [Image.open(path) for path in image_paths]
        retrieval.index_images(images, image_ids)

        # Search by text
        query = "a cat sitting on a chair"
        print(f"\nText Query: '{query}'")
        results = retrieval.search_images_by_text(query, top_k=3)

        print("Top matching images:")
        for i, result in enumerate(results):
            print(f"  {i + 1}. {result['image_id']} (similarity: {result['similarity']:.4f})")

    except FileNotFoundError:
        print("Sample images not found. Please provide your own test images.")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("MULTI-MODAL VLM INFERENCE DEMO")
    print("=" * 60)

    # Run demos
    demo_captioning()
    demo_vqa()
    demo_retrieval()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
