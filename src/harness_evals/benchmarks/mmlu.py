"""MMLU benchmark: Massive Multitask Language Understanding."""

from __future__ import annotations

from typing import Any

from harness_evals.benchmarks._answer_utils import extract_choice
from harness_evals.benchmarks.base import BaseBenchmark
from harness_evals.benchmarks.dataset_cache import fetch_hf_dataset

MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


class MMLU(BaseBenchmark):
    """MMLU: Massive Multitask Language Understanding.

    57-subject knowledge benchmark covering STEM, humanities, social sciences, and more.
    Default is 5-shot evaluation matching the original paper.
    """

    def __init__(self, *, subjects: list[str] | None = None, shots: int = 5) -> None:
        """Initialize MMLU benchmark.

        Args:
            subjects: List of subjects to evaluate. None = all 57 subjects.
            shots: Default number of few-shot examples (default 5, per original paper).
        """
        super().__init__(name="mmlu", default_shots=shots)
        if subjects:
            invalid = set(subjects) - set(MMLU_SUBJECTS)
            if invalid:
                raise ValueError(f"Unknown MMLU subjects: {invalid}")
        self.subjects = subjects or MMLU_SUBJECTS
        self._few_shot_examples: dict[str, list[dict]] = {}

    async def load_dataset(self, *, offline: bool = False) -> list[dict]:
        items: list[dict] = []
        for subject in self.subjects:
            data = await fetch_hf_dataset("cais/mmlu", split="test", config=subject, offline=offline)
            for item in data:
                item["subject"] = subject
            items.extend(data)

            dev_data = await fetch_hf_dataset("cais/mmlu", split="dev", config=subject, offline=offline)
            self._few_shot_examples[subject] = dev_data[:5]

        return items

    def format_prompt(self, item: dict, *, shots: int = 5) -> str:
        subject = item.get("subject", "general")
        question = item["question"]
        choices = item["choices"]

        subject_display = subject.replace("_", " ").title()
        header = f"The following are multiple choice questions about {subject_display}.\n\n"

        few_shot_prefix = ""
        if shots > 0:
            examples = self._few_shot_examples.get(subject, [])[:shots]
            for ex in examples:
                ex_choices = ex["choices"]
                choices_str = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(ex_choices))
                answer_idx = ex["answer"]
                answer_letter = chr(65 + answer_idx) if isinstance(answer_idx, int) else answer_idx
                few_shot_prefix += f"Question: {ex['question']}\n{choices_str}\nAnswer: {answer_letter}\n\n"

        choices_str = "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))

        return f"{header}{few_shot_prefix}Question: {question}\n{choices_str}\nAnswer:"

    def score_response(self, item: dict, response: str) -> tuple[float, str | None]:
        answer = item["answer"]
        expected = chr(65 + answer) if isinstance(answer, int) else answer.upper()

        num_choices = len(item["choices"])
        valid = [chr(65 + i) for i in range(num_choices)]
        predicted = extract_choice(response, choices=valid)

        if predicted is None:
            return 0.0, "Could not parse choice answer"

        if predicted == expected:
            return 1.0, None
        return 0.0, f"Expected {expected}, got {predicted}"

    def _get_expected(self, item: dict) -> str:
        answer = item["answer"]
        return chr(65 + answer) if isinstance(answer, int) else answer.upper()

    def _get_item_metadata(self, item: dict) -> dict[str, Any]:
        return {"subject": item.get("subject", "")}

    def _get_result_metadata(self) -> dict[str, Any]:
        return {"subjects": self.subjects, "num_subjects": len(self.subjects)}
