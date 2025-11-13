"""
Quick Start Example for NLP-LLM-Applications

This script demonstrates the basic usage of the implemented features:
- Loading language models
- Prompt engineering
- Fine-tuning with LoRA
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import GPTModel, BERTModel, T5Model
from models import PromptTemplate, FewShotLearner, Example, ChainOfThought
from models import LoRAFineTuner, LoRAConfig, apply_lora_to_model

print("="*70)
print("NLP-LLM-Applications - Quick Start Demo")
print("="*70)

# Example 1: GPT Text Generation
print("\n1. GPT-2 Text Generation")
print("-"*70)
print("Loading GPT-2 model...")
gpt = GPTModel(model_name="gpt2", use_pretrained=True)

prompt = "In a world where AI and humans coexist,"
print(f"Prompt: {prompt}")
print("Generating text...")

generated = gpt.generate(
    prompt,
    max_length=100,
    temperature=0.8,
    top_p=0.95,
    do_sample=True
)

print(f"\nGenerated text:\n{generated[0]}")

# Example 2: Prompt Engineering
print("\n\n2. Prompt Engineering - Few-Shot Learning")
print("-"*70)

# Create examples for sentiment classification
examples = [
    Example("This product is amazing! I love it!", "positive"),
    Example("Terrible experience, would not recommend.", "negative"),
    Example("It's okay, nothing special.", "neutral"),
]

learner = FewShotLearner(examples, n_shots=3, include_reasoning=False)

# Test classification
test_input = "Best purchase I've ever made!"
prompt = learner.build_prompt(test_input, instruction="Classify sentiment:")

print(f"Test input: {test_input}")
print(f"\nFew-shot prompt:\n{prompt}")

# Example 3: Template-based Prompting
print("\n\n3. Template-Based Prompting")
print("-"*70)

template = PromptTemplate(
    template="Summarize the following in one sentence: {text}\n\nSummary:",
    variables=["text"]
)

text = "Artificial intelligence is transforming how we live and work. Machine learning algorithms can now perform tasks that were once thought to be exclusively human. From medical diagnosis to creative writing, AI is making significant impacts across all industries."

prompt = template.format(text=text)
print(f"Template prompt:\n{prompt}")

# Example 4: LoRA Fine-Tuning Setup
print("\n\n4. LoRA Fine-Tuning Setup")
print("-"*70)

print("Setting up LoRA for parameter-efficient fine-tuning...")

# Create LoRA configuration
lora_config = LoRAConfig(
    r=8,  # Rank
    lora_alpha=16,  # Scaling
    lora_dropout=0.1,
    target_modules=["c_attn"]  # For GPT-2
)

# Apply LoRA to GPT model
print(f"Applying LoRA with rank={lora_config.r}, alpha={lora_config.lora_alpha}")

lora_model = apply_lora_to_model(
    gpt.model,
    r=lora_config.r,
    lora_alpha=lora_config.lora_alpha,
    target_modules=lora_config.target_modules
)

# Get parameter counts
trainable, total = lora_model.get_trainable_parameters()
percentage = 100 * trainable / total

print(f"\nParameter Statistics:")
print(f"  Trainable parameters: {trainable:,}")
print(f"  Total parameters: {total:,}")
print(f"  Trainable percentage: {percentage:.2f}%")
print(f"\nMemory savings: {100 - percentage:.2f}% of parameters frozen!")

# Example 5: Chain-of-Thought Reasoning
print("\n\n5. Chain-of-Thought Reasoning")
print("-"*70)

cot = ChainOfThought(model=gpt, max_steps=5)

problem = "If 3 books cost $15, how much would 7 books cost at the same rate?"
goal = "calculate the cost of 7 books"

print(f"Problem: {problem}")
print(f"Goal: {goal}")

# Create reasoning prompt
reasoning_prompt = cot._default_template().format(goal=goal, problem=problem)
print(f"\nChain-of-thought prompt:\n{reasoning_prompt[:200]}...")

# Example 6: Dynamic Prompt Selection
print("\n\n6. Classification with Multiple Prompts")
print("-"*70)

# Create classification prompt
classes = ["positive", "negative", "neutral"]
text = "This is absolutely fantastic!"

classification_prompt = PromptTemplate.create_classification_prompt(
    text, classes, instruction="Classify the sentiment"
)

print(f"Text to classify: {text}")
print(f"\nClassification prompt:\n{classification_prompt}")

# Example 7: Question Answering Setup
print("\n\n7. Question Answering Setup")
print("-"*70)

context = "The Eiffel Tower is located in Paris, France. It was completed in 1889 and stands 324 meters tall."
question = "Where is the Eiffel Tower located?"

qa_prompt = PromptTemplate.create_qa_prompt(question, context)

print(f"Context: {context}")
print(f"Question: {question}")
print(f"\nQA Prompt:\n{qa_prompt}")

print("\n" + "="*70)
print("Quick Start Demo Complete!")
print("="*70)
print("\nKey Features Demonstrated:")
print("  [OK] GPT-2 text generation")
print("  [OK] Few-shot learning")
print("  [OK] Template-based prompting")
print("  [OK] LoRA parameter-efficient fine-tuning")
print("  [OK] Chain-of-thought reasoning")
print("  [OK] Classification prompts")
print("  [OK] Question answering prompts")
print("\nNext steps:")
print("  - Explore models/language_models.py for model details")
print("  - Check models/prompt_engineering.py for advanced prompting")
print("  - See models/fine_tuning.py for fine-tuning options")
