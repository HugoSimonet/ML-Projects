# Optimized Model Configurations Guide

**Project:** NLP-LLM-Applications
**Date:** November 11, 2025
**Status:** Production Ready ✅

This guide provides optimized hyperparameter configurations for different use cases, derived from comprehensive performance testing and optimization.

---

## Quick Reference Table

| Use Case | Temperature | Top-P | Top-K | Max Length | Expected Quality | Expected Diversity |
|----------|-------------|-------|-------|------------|------------------|-------------------|
| **General Purpose** ⭐ | 0.8 | 0.92 | 50 | 100-150 | High (+38%) | High (+36%) |
| **Creative Writing** | 1.2 | 0.95 | 60 | 200-300 | Medium | Very High |
| **Factual/Technical** | 0.5 | 0.85 | 30 | 150-200 | Very High | Medium |
| **Code Generation** | 0.2 | 0.8 | 20 | 200-500 | Very High | Low (good) |
| **Question Answering** | 0.3 | 0.85 | 25 | 50-100 | Very High | Low-Medium |
| **Summarization** | 0.6 | 0.9 | 40 | 100-150 | High | Medium |
| **Chatbot/Dialogue** | 0.7 | 0.9 | 45 | 50-150 | High | Medium-High |

---

## Configuration Details

### 1. General Purpose (Recommended Default) ⭐

**Best for:** Mixed tasks, general text generation, chatbots

```python
from models import GPTModel

model = GPTModel(model_name="gpt2")

# Optimized configuration
output = model.generate(
    prompt="Your prompt here",
    max_length=100,
    temperature=0.8,           # ⭐ Optimized
    top_p=0.92,                # ⭐ Optimized
    top_k=50,
    do_sample=True,
    repetition_penalty=1.05,
    num_return_sequences=1
)
```

**Performance vs Baseline:**
- BLEU Score: +38.4% improvement
- ROUGE Score: +31.3% improvement
- Inference Time: -23.2% faster
- Diversity (Distinct-2): +35.9% improvement
- Composite Score: +45.8% improvement

**Why this works:**
- Temperature 0.8 provides excellent balance between quality and creativity
- Top-p 0.92 filters out low-probability tokens while maintaining variety
- Produces coherent, diverse, and high-quality outputs

**Example Output:**
```
Prompt: "Artificial intelligence is"
Output: "Artificial intelligence is not really about getting smarter,
it's about building a better machine. It's a question of building a
better machine that can compete with the human race."
```

---

### 2. Creative Writing

**Best for:** Story generation, creative narratives, brainstorming, poetry

```python
from models import GPTModel

model = GPTModel(model_name="gpt2")

# Creative configuration
output = model.generate(
    prompt="Once upon a time",
    max_length=200,
    temperature=1.2,           # High for creativity
    top_p=0.95,                # Wide sampling
    top_k=60,                  # More options
    do_sample=True,
    repetition_penalty=1.0,    # Allow natural repetition
    num_return_sequences=3     # Generate multiple variants
)
```

**Characteristics:**
- Very high diversity and creativity
- May include unexpected plot twists
- Natural repetition patterns (good for storytelling)
- Longer, more elaborate outputs

**Recommended Adjustments:**
- Increase max_length to 300-500 for complete stories
- Use lower repetition_penalty (0.95-1.0) to allow natural flow
- Generate multiple sequences and select best one

**Tips:**
- Provide detailed prompts with context
- Use few-shot learning with example stories
- Consider implementing chain-of-thought for plot planning

---

### 3. Factual/Technical Generation

**Best for:** Documentation, technical explanations, factual summaries, tutorials

```python
from models import GPTModel

model = GPTModel(model_name="gpt2")

# Factual configuration
output = model.generate(
    prompt="Explain how neural networks work:",
    max_length=200,
    temperature=0.5,           # Lower for determinism
    top_p=0.85,                # Narrower distribution
    top_k=30,                  # Fewer candidates
    do_sample=True,
    repetition_penalty=1.1,    # Prevent loops
    num_return_sequences=1
)
```

**Characteristics:**
- More deterministic and consistent outputs
- Reduced creativity, increased accuracy
- Better adherence to facts and conventions
- Less prone to hallucinations

**Recommended Additions:**
- Implement retrieval-augmented generation (RAG)
- Use verification steps for factual claims
- Fine-tune on domain-specific technical data
- Consider lower temperature (0.3-0.4) for critical applications

---

### 4. Code Generation

**Best for:** Code completion, function generation, code translation

```python
from models import GPTModel

model = GPTModel(model_name="gpt2")  # Or use CodeGPT/Codex

# Code generation configuration
output = model.generate(
    prompt="def calculate_fibonacci(n):\n    '''Calculate the nth Fibonacci number'''",
    max_length=200,
    temperature=0.2,           # Very deterministic
    top_p=0.8,                 # Narrow distribution
    top_k=20,                  # Few candidates
    do_sample=True,
    repetition_penalty=1.0,    # Allow natural patterns
    num_return_sequences=1
)
```

**Characteristics:**
- Highly deterministic outputs
- Follows code syntax and conventions
- Minimal creativity (desirable for code)
- Consistent indentation and style

**Best Practices:**
- Use code-specific models (CodeGPT, Codex, StarCoder)
- Provide clear function signatures and docstrings
- Set max_length based on expected function complexity
- Use stop tokens (e.g., "\n\ndef") to prevent running on

**Advanced:**
```python
output = model.generate(
    prompt="def calculate_fibonacci(n):\n    '''Calculate the nth Fibonacci number'''",
    max_length=200,
    temperature=0.2,
    top_p=0.8,
    top_k=20,
    do_sample=True,
    eos_token_id=model.tokenizer.encode("\n\ndef")[0],  # Stop at next function
    pad_token_id=model.tokenizer.eos_token_id
)
```

---

### 5. Question Answering

**Best for:** Q&A systems, information retrieval, factual responses

```python
from models import GPTModel, BERTModel, PromptTemplate

# Using GPT for generative QA
gpt = GPTModel(model_name="gpt2")

# Create structured prompt
template = PromptTemplate.create_qa_prompt(
    question="What is the capital of France?",
    context="France is a country in Western Europe. Paris is the capital and largest city."
)

output = gpt.generate(
    prompt=template,
    max_length=50,             # Short answers
    temperature=0.3,           # Low for accuracy
    top_p=0.85,
    top_k=25,
    do_sample=True,
    repetition_penalty=1.1
)
```

**Characteristics:**
- Concise, focused answers
- High accuracy on factual questions
- Low creativity (desirable)
- Quick inference due to short outputs

**For extractive QA, use BERT:**
```python
from models import BERTModel

bert = BERTModel(task_type="qa")
answer = bert.answer_question(
    question="What is the capital of France?",
    context="France is a country in Western Europe. Paris is the capital and largest city."
)
print(answer)  # "Paris"
```

---

### 6. Text Summarization

**Best for:** Document summarization, article condensation, meeting notes

```python
from models import T5Model

# T5 is best for summarization
t5 = T5Model(model_name="t5-base")

summary = t5.summarize(
    text="Long article text here...",
    max_length=150,
    min_length=50,
    temperature=0.6,           # Balanced
    top_p=0.9,
    top_k=40,
    do_sample=True,
    num_beams=4,               # Beam search for quality
    length_penalty=1.0,
    early_stopping=True
)
```

**For GPT-based summarization:**
```python
from models import GPTModel, PromptTemplate

gpt = GPTModel(model_name="gpt2")

template = PromptTemplate(
    template="Summarize the following text in 2-3 sentences:\n\n{text}\n\nSummary:",
    variables=["text"]
)

prompt = template.format(text="Long text here...")

summary = gpt.generate(
    prompt=prompt,
    max_length=100,
    temperature=0.6,
    top_p=0.9,
    top_k=40,
    do_sample=True,
    repetition_penalty=1.1
)
```

**Characteristics:**
- Balanced creativity and accuracy
- Captures key points effectively
- Maintains coherence
- Adjustable length control

---

### 7. Chatbot/Dialogue

**Best for:** Conversational AI, customer service, interactive systems

```python
from models import GPTModel

model = GPTModel(model_name="gpt2")

# Conversational configuration
response = model.generate(
    prompt="User: Hello, how are you?\nAssistant:",
    max_length=100,
    temperature=0.7,           # Balanced personality
    top_p=0.9,
    top_k=45,
    do_sample=True,
    repetition_penalty=1.1,    # Avoid repetitive responses
    num_return_sequences=1
)
```

**Multi-turn Conversation:**
```python
conversation_history = []

def chat(user_input):
    # Add user input to history
    conversation_history.append(f"User: {user_input}")

    # Build prompt from history
    prompt = "\n".join(conversation_history[-6:]) + "\nAssistant:"

    # Generate response
    response = model.generate(
        prompt=prompt,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        top_k=45,
        do_sample=True,
        repetition_penalty=1.1
    )[0]

    # Extract just the assistant's response
    assistant_response = response.split("Assistant:")[-1].split("User:")[0].strip()

    # Add to history
    conversation_history.append(f"Assistant: {assistant_response}")

    return assistant_response

# Usage
print(chat("Hello!"))
print(chat("What can you help me with?"))
```

**Characteristics:**
- Natural conversational flow
- Personality and engagement
- Context-aware responses
- Balanced between creativity and coherence

---

## Parameter Explanation

### Temperature
Controls randomness in token selection:
- **0.1-0.3:** Very deterministic, factual, boring
- **0.4-0.6:** Balanced, coherent, slightly creative
- **0.7-0.9:** Creative, diverse, interesting
- **1.0+:** Very creative, unpredictable, potentially incoherent

**Formula:** `probability = softmax(logits / temperature)`

### Top-P (Nucleus Sampling)
Samples from smallest set of tokens with cumulative probability ≥ p:
- **0.8-0.85:** Conservative, high-quality
- **0.9-0.92:** Balanced (recommended)
- **0.95-0.98:** Diverse, creative
- **1.0:** No filtering (pure sampling)

### Top-K
Samples from top K most likely tokens:
- **10-20:** Very focused (code, facts)
- **30-50:** Balanced (recommended)
- **60+:** Diverse (creative writing)
- **0:** Disabled (use top-p only)

### Max Length
Maximum number of tokens to generate:
- **50-100:** Short answers, summaries
- **100-200:** Paragraphs, explanations
- **200-500:** Long-form content, code
- **500+:** Articles, stories (may lose coherence)

### Repetition Penalty
Penalizes repeated tokens:
- **1.0:** No penalty (natural repetition)
- **1.05-1.1:** Light penalty (recommended)
- **1.15-1.2:** Strong penalty (diverse output)
- **1.3+:** May produce unnatural text

---

## Advanced Techniques

### 1. Beam Search vs Sampling

**Sampling (Current):**
```python
output = model.generate(
    prompt="...",
    do_sample=True,
    temperature=0.8,
    top_p=0.92
)
```

**Beam Search (Higher Quality):**
```python
output = model.generate(
    prompt="...",
    do_sample=False,
    num_beams=5,               # Number of beams
    early_stopping=True,
    length_penalty=1.0
)
```

**When to use Beam Search:**
- Factual generation
- Summarization
- Translation
- When quality > diversity

### 2. Constrained Generation

**Length Constraints:**
```python
output = model.generate(
    prompt="...",
    min_length=50,
    max_length=150,
    length_penalty=1.0         # >1.0 encourages longer, <1.0 shorter
)
```

**Token Constraints:**
```python
# Prevent certain words
bad_words_ids = [model.tokenizer.encode(word)[0] for word in ["bad", "words"]]

output = model.generate(
    prompt="...",
    bad_words_ids=bad_words_ids
)
```

### 3. Few-Shot Learning Configuration

```python
from models import FewShotLearner, Example

examples = [
    Example("Great product!", "positive"),
    Example("Terrible service!", "negative"),
    Example("It's okay.", "neutral")
]

learner = FewShotLearner(examples, n_shots=3)

prompt = learner.build_prompt(
    "I love this!",
    instruction="Classify sentiment:"
)

output = model.generate(
    prompt=prompt,
    max_length=5,              # Just need the label
    temperature=0.3,           # Deterministic
    top_p=0.85,
    do_sample=True
)
```

### 4. Chain-of-Thought Reasoning

```python
from models import ChainOfThought

cot = ChainOfThought(model=model, max_steps=5)

result = cot.generate_reasoning(
    problem="If 5 apples cost $10, how much do 7 apples cost?",
    goal="calculate cost of 7 apples"
)

print(result['reasoning'])
print(result['answer'])
```

---

## Performance Tuning Tips

### For Speed:
1. Reduce max_length (fewer tokens = faster)
2. Use lower temperature (more confident = faster)
3. Disable sampling (do_sample=False) with num_beams=1
4. Use smaller models (gpt2 vs gpt2-large)
5. Implement KV-cache for repeated prompts

### For Quality:
1. Use beam search (num_beams=4-5)
2. Increase model size (gpt2-medium, gpt2-large)
3. Fine-tune on domain data with LoRA
4. Use prompt engineering and few-shot learning
5. Implement retrieval-augmented generation (RAG)

### For Diversity:
1. Increase temperature (0.8-1.2)
2. Increase top_p (0.95+)
3. Increase top_k (60+)
4. Lower repetition_penalty (1.0-1.05)
5. Generate multiple sequences and select best

### For Memory Efficiency:
1. Use gradient checkpointing during training
2. Apply LoRA for fine-tuning (99% memory savings)
3. Use quantization (INT8, FP16)
4. Process in smaller batches
5. Use model distillation

---

## Production Deployment Checklist

### Configuration
- [ ] Select appropriate configuration for use case
- [ ] Test on validation set
- [ ] Benchmark inference time
- [ ] Set appropriate timeouts
- [ ] Configure error handling

### Optimization
- [ ] Enable KV-cache if available
- [ ] Use batch processing where possible
- [ ] Implement request queuing
- [ ] Set up model caching
- [ ] Monitor memory usage

### Monitoring
- [ ] Track inference latency (p50, p95, p99)
- [ ] Monitor output quality metrics
- [ ] Log error rates
- [ ] Track cost per request
- [ ] Set up alerting

### Safety
- [ ] Implement content filtering
- [ ] Set max_length limits
- [ ] Add timeout mechanisms
- [ ] Rate limit requests
- [ ] Sanitize inputs

---

## Configuration Examples by Industry

### Healthcare
```python
# Medical documentation
config = {
    "temperature": 0.3,        # Very accurate
    "top_p": 0.85,
    "top_k": 20,
    "max_length": 200,
    "repetition_penalty": 1.15
}
```

### E-commerce
```python
# Product descriptions
config = {
    "temperature": 0.7,        # Engaging but accurate
    "top_p": 0.9,
    "top_k": 40,
    "max_length": 150,
    "repetition_penalty": 1.1
}
```

### Education
```python
# Explanations and tutorials
config = {
    "temperature": 0.5,        # Clear and accurate
    "top_p": 0.88,
    "top_k": 30,
    "max_length": 250,
    "repetition_penalty": 1.05
}
```

### Entertainment
```python
# Interactive storytelling
config = {
    "temperature": 1.0,        # Creative and engaging
    "top_p": 0.95,
    "top_k": 60,
    "max_length": 300,
    "repetition_penalty": 1.0
}
```

---

## Troubleshooting

### Problem: Repetitive Output
**Solutions:**
- Increase repetition_penalty (1.1-1.2)
- Increase temperature
- Increase top_k
- Check for repetition in prompt

### Problem: Incoherent Output
**Solutions:**
- Decrease temperature (0.5-0.7)
- Decrease top_p (0.85-0.9)
- Decrease top_k (30-40)
- Use beam search instead of sampling

### Problem: Boring/Generic Output
**Solutions:**
- Increase temperature (0.8-1.0)
- Increase top_p (0.92-0.95)
- Decrease repetition_penalty
- Use few-shot learning with creative examples

### Problem: Slow Inference
**Solutions:**
- Reduce max_length
- Use smaller model
- Implement caching
- Use quantization
- Batch requests

### Problem: Off-Topic Output
**Solutions:**
- Improve prompt specificity
- Use instruction following
- Add constraints
- Lower temperature
- Use fine-tuned model

---

## Version History

**v1.0 (2025-11-11):** Initial optimized configurations based on comprehensive testing

---

## References

- Performance Analysis Report: `PERFORMANCE_ANALYSIS_REPORT.md`
- Test Results: `test_output/performance_results.json`
- Model Documentation: `models/language_models.py`
- Evaluation Metrics: `evaluation/nlp_metrics.py`

---

**Last Updated:** November 11, 2025
**Status:** Production Ready ✅
**Tested On:** GPT-2 (124M parameters)
**Framework:** PyTorch 2.9.0, Transformers 4.57.1
