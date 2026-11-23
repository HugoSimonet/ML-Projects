"""
Prompt Engineering Framework for Large Language Models.

This module provides advanced prompt engineering techniques including:
- Template-based prompting
- Few-shot learning
- Chain-of-thought reasoning
- Dynamic prompt generation
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import json
import re


@dataclass
class Example:
    """A single example for few-shot learning."""
    input: str
    output: str
    reasoning: Optional[str] = None

    def format(self, include_reasoning: bool = False) -> str:
        """Format the example as a string."""
        if include_reasoning and self.reasoning:
            return f"Input: {self.input}\nReasoning: {self.reasoning}\nOutput: {self.output}"
        return f"Input: {self.input}\nOutput: {self.output}"


class PromptTemplate:
    """
    Template-based prompt generation.

    Features:
    - Variable substitution
    - Instruction formatting
    - Context injection
    - Output formatting
    """

    def __init__(
        self,
        template: str,
        variables: Optional[List[str]] = None,
        system_message: Optional[str] = None
    ):
        """
        Initialize prompt template.

        Args:
            template: Template string with {variable} placeholders
            variables: List of variable names
            system_message: Optional system message (for chat-based models)
        """
        self.template = template
        self.system_message = system_message
        self.variables = variables or self._extract_variables(template)

    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable names from template."""
        return re.findall(r'\{(\w+)\}', template)

    def format(self, **kwargs) -> str:
        """
        Format the prompt with given variables.

        Args:
            **kwargs: Variable values to substitute

        Returns:
            Formatted prompt string
        """
        # Validate all required variables are provided
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        prompt = self.template.format(**kwargs)

        if self.system_message:
            prompt = f"{self.system_message}\n\n{prompt}"

        return prompt

    def add_context(self, context: str, position: str = "before") -> 'PromptTemplate':
        """
        Add context to the template.

        Args:
            context: Context string to add
            position: Where to add context ('before' or 'after')

        Returns:
            New PromptTemplate with context added
        """
        if position == "before":
            new_template = f"{context}\n\n{self.template}"
        else:
            new_template = f"{self.template}\n\n{context}"

        return PromptTemplate(new_template, self.variables, self.system_message)

    @staticmethod
    def create_classification_prompt(
        text: str,
        classes: List[str],
        instruction: Optional[str] = None
    ) -> str:
        """Create a classification prompt."""
        instruction = instruction or "Classify the following text into one of these categories"
        classes_str = ", ".join(classes)
        return f"{instruction}: {classes_str}\n\nText: {text}\n\nCategory:"

    @staticmethod
    def create_qa_prompt(
        question: str,
        context: Optional[str] = None,
        instruction: Optional[str] = None
    ) -> str:
        """Create a question-answering prompt."""
        instruction = instruction or "Answer the following question"

        if context:
            return f"{instruction} based on the context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            return f"{instruction}:\n\nQuestion: {question}\n\nAnswer:"

    @staticmethod
    def create_summarization_prompt(
        text: str,
        max_length: Optional[int] = None,
        style: str = "concise"
    ) -> str:
        """Create a summarization prompt."""
        prompt = f"Summarize the following text in a {style} manner"

        if max_length:
            prompt += f" (max {max_length} words)"

        prompt += f":\n\n{text}\n\nSummary:"
        return prompt


class FewShotLearner:
    """
    Few-shot learning framework.

    Features:
    - Example selection
    - Example formatting
    - Dynamic shot number
    - Example diversity
    """

    def __init__(
        self,
        examples: List[Example],
        n_shots: int = 3,
        include_reasoning: bool = False,
        example_separator: str = "\n\n---\n\n"
    ):
        """
        Initialize few-shot learner.

        Args:
            examples: List of Example objects
            n_shots: Number of examples to use
            include_reasoning: Whether to include reasoning in examples
            example_separator: Separator between examples
        """
        self.examples = examples
        self.n_shots = n_shots
        self.include_reasoning = include_reasoning
        self.example_separator = example_separator

    def build_prompt(
        self,
        query: str,
        instruction: Optional[str] = None,
        select_diverse: bool = True
    ) -> str:
        """
        Build few-shot prompt.

        Args:
            query: The input query
            instruction: Optional instruction to prepend
            select_diverse: Whether to select diverse examples

        Returns:
            Formatted few-shot prompt
        """
        # Select examples
        if select_diverse and len(self.examples) > self.n_shots:
            selected = self._select_diverse_examples(query, self.n_shots)
        else:
            selected = self.examples[:self.n_shots]

        # Format examples
        examples_str = self.example_separator.join(
            [ex.format(self.include_reasoning) for ex in selected]
        )

        # Build prompt
        prompt_parts = []

        if instruction:
            prompt_parts.append(instruction)

        prompt_parts.append("Examples:")
        prompt_parts.append(examples_str)
        prompt_parts.append(f"Input: {query}\nOutput:")

        return "\n\n".join(prompt_parts)

    def _select_diverse_examples(self, query: str, n: int) -> List[Example]:
        """Select diverse examples (simple version using length)."""
        # Sort by diversity (here, simply by length difference)
        query_len = len(query)
        scored = [(abs(len(ex.input) - query_len), ex) for ex in self.examples]
        scored.sort(key=lambda x: x[0], reverse=True)

        return [ex for _, ex in scored[:n]]

    def add_example(self, input: str, output: str, reasoning: Optional[str] = None):
        """Add a new example."""
        self.examples.append(Example(input, output, reasoning))

    def remove_example(self, index: int):
        """Remove an example by index."""
        if 0 <= index < len(self.examples):
            self.examples.pop(index)

    @staticmethod
    def from_dict(data: List[Dict[str, str]], **kwargs) -> 'FewShotLearner':
        """Create FewShotLearner from list of dictionaries."""
        examples = [
            Example(
                input=item['input'],
                output=item['output'],
                reasoning=item.get('reasoning')
            )
            for item in data
        ]
        return FewShotLearner(examples, **kwargs)


class ChainOfThought:
    """
    Chain-of-thought reasoning framework.

    Features:
    - Step-by-step reasoning
    - Intermediate step generation
    - Reasoning verification
    """

    def __init__(
        self,
        model=None,
        reasoning_template: Optional[str] = None,
        max_steps: int = 5
    ):
        """
        Initialize chain-of-thought reasoner.

        Args:
            model: Language model for generation
            reasoning_template: Template for reasoning steps
            max_steps: Maximum number of reasoning steps
        """
        self.model = model
        self.max_steps = max_steps
        self.reasoning_template = reasoning_template or self._default_template()

    def _default_template(self) -> str:
        """Default reasoning template."""
        return """Let's solve this step by step:

1. First, identify what we need to find: {goal}
2. Break down the problem into smaller steps
3. Solve each step carefully
4. Verify the solution

Problem: {problem}

Let's think through this:"""

    def generate_reasoning(
        self,
        problem: str,
        goal: Optional[str] = None,
        return_steps: bool = True
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate chain-of-thought reasoning.

        Args:
            problem: Problem to solve
            goal: What we want to find/achieve
            return_steps: Whether to return intermediate steps

        Returns:
            Reasoning (with or without steps)
        """
        if not self.model:
            raise ValueError("Model required for reasoning generation")

        # Format prompt
        goal = goal or "solve the problem"
        prompt = self.reasoning_template.format(goal=goal, problem=problem)

        # Generate reasoning
        reasoning = self.model.generate(prompt, max_length=500)[0]

        if return_steps:
            steps = self._extract_steps(reasoning)
            return {
                'problem': problem,
                'reasoning': reasoning,
                'steps': steps,
                'conclusion': steps[-1] if steps else None
            }

        return reasoning

    def _extract_steps(self, reasoning: str) -> List[str]:
        """Extract reasoning steps from generated text."""
        # Look for numbered steps or bullet points
        lines = reasoning.strip().split('\n')
        steps = []

        for line in lines:
            line = line.strip()
            # Match numbered steps (1., 2., etc.) or bullet points (-, *, etc.)
            if re.match(r'^(\d+\.|[-*])\s+', line):
                step = re.sub(r'^(\d+\.|[-*])\s+', '', line)
                if step:
                    steps.append(step)

        return steps

    def verify_reasoning(
        self,
        problem: str,
        reasoning: str,
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify reasoning quality.

        Args:
            problem: Original problem
            reasoning: Generated reasoning
            expected_answer: Expected answer (if known)

        Returns:
            Verification result with score and feedback
        """
        steps = self._extract_steps(reasoning)

        # Basic verification metrics
        has_steps = len(steps) > 0
        has_conclusion = any(
            keyword in reasoning.lower()
            for keyword in ['therefore', 'thus', 'conclusion', 'answer']
        )
        length_appropriate = 50 < len(reasoning) < 1000

        score = sum([has_steps, has_conclusion, length_appropriate]) / 3

        return {
            'score': score,
            'has_steps': has_steps,
            'num_steps': len(steps),
            'has_conclusion': has_conclusion,
            'length_appropriate': length_appropriate,
            'feedback': self._generate_feedback(has_steps, has_conclusion, length_appropriate)
        }

    def _generate_feedback(
        self,
        has_steps: bool,
        has_conclusion: bool,
        length_appropriate: bool
    ) -> str:
        """Generate feedback based on verification."""
        feedback = []

        if not has_steps:
            feedback.append("Reasoning should include clear steps")
        if not has_conclusion:
            feedback.append("Reasoning should include a clear conclusion")
        if not length_appropriate:
            feedback.append("Reasoning length should be appropriate")

        return "; ".join(feedback) if feedback else "Reasoning looks good"


class InstructionFollower:
    """
    Framework for instruction-following prompts.

    Features:
    - Instruction parsing
    - Task decomposition
    - Multi-step instructions
    """

    def __init__(self, model=None):
        """Initialize instruction follower."""
        self.model = model

    def format_instruction(
        self,
        task: str,
        input: str,
        constraints: Optional[List[str]] = None,
        format_spec: Optional[str] = None
    ) -> str:
        """
        Format an instruction prompt.

        Args:
            task: Task description
            input: Input data
            constraints: List of constraints
            format_spec: Output format specification

        Returns:
            Formatted instruction prompt
        """
        prompt_parts = [f"Task: {task}"]

        if constraints:
            prompt_parts.append("Constraints:")
            for i, constraint in enumerate(constraints, 1):
                prompt_parts.append(f"{i}. {constraint}")

        if format_spec:
            prompt_parts.append(f"Output format: {format_spec}")

        prompt_parts.append(f"\nInput:\n{input}")
        prompt_parts.append("\nOutput:")

        return "\n".join(prompt_parts)

    def decompose_task(self, complex_task: str) -> List[str]:
        """
        Decompose a complex task into subtasks.

        Args:
            complex_task: Complex task description

        Returns:
            List of subtasks
        """
        # Simple decomposition based on conjunctions
        subtasks = []

        # Split on common task separators
        parts = re.split(r'\sand\s|\sthen\s|,\s*and\s', complex_task, flags=re.IGNORECASE)

        for part in parts:
            cleaned = part.strip()
            if cleaned:
                subtasks.append(cleaned)

        return subtasks


class DynamicPromptSelector:
    """
    Dynamic prompt selection based on task and context.

    Features:
    - Task-based prompt selection
    - Performance tracking
    - Adaptive selection
    """

    def __init__(self):
        """Initialize dynamic prompt selector."""
        self.prompts: Dict[str, List[str]] = {}
        self.performance: Dict[str, Dict[str, float]] = {}

    def add_prompt(self, task: str, prompt: str, name: Optional[str] = None):
        """Add a prompt for a task."""
        if task not in self.prompts:
            self.prompts[task] = []
            self.performance[task] = {}

        self.prompts[task].append(prompt)

        if name:
            self.performance[task][name] = 0.0

    def select_prompt(
        self,
        task: str,
        method: str = "best",
        **context
    ) -> Optional[str]:
        """
        Select a prompt for a task.

        Args:
            task: Task name
            method: Selection method ('random', 'best', 'round_robin')
            **context: Additional context for selection

        Returns:
            Selected prompt
        """
        if task not in self.prompts or not self.prompts[task]:
            return None

        prompts = self.prompts[task]

        if method == "random":
            import random
            return random.choice(prompts)

        elif method == "best":
            # Select prompt with best performance
            if task in self.performance and self.performance[task]:
                best_name = max(self.performance[task], key=self.performance[task].get)
                best_idx = list(self.performance[task].keys()).index(best_name)
                return prompts[best_idx] if best_idx < len(prompts) else prompts[0]
            return prompts[0]

        elif method == "round_robin":
            # Simple round-robin selection (requires state tracking)
            return prompts[0]  # Simplified

        return prompts[0]

    def update_performance(self, task: str, prompt_name: str, score: float):
        """Update performance score for a prompt."""
        if task in self.performance and prompt_name in self.performance[task]:
            # Exponential moving average
            alpha = 0.3
            old_score = self.performance[task][prompt_name]
            self.performance[task][prompt_name] = alpha * score + (1 - alpha) * old_score


# Utility functions

def create_chat_prompt(
    messages: List[Dict[str, str]],
    system_message: Optional[str] = None
) -> str:
    """
    Create a chat-style prompt.

    Args:
        messages: List of message dicts with 'role' and 'content'
        system_message: Optional system message

    Returns:
        Formatted chat prompt
    """
    prompt_parts = []

    if system_message:
        prompt_parts.append(f"System: {system_message}\n")

    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        prompt_parts.append(f"{role.capitalize()}: {content}")

    return "\n".join(prompt_parts)


def extract_answer(
    generated_text: str,
    answer_pattern: str = r"Answer:\s*(.+?)(?:\n|$)"
) -> Optional[str]:
    """
    Extract answer from generated text.

    Args:
        generated_text: Generated text containing answer
        answer_pattern: Regex pattern to extract answer

    Returns:
        Extracted answer or None
    """
    match = re.search(answer_pattern, generated_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
