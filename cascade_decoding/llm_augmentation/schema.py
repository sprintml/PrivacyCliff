"""Dataset schema management for LLM-guided generation.

This module handles loading, saving, and inferring schemas that describe
the semantic structure of tabular datasets. Schemas include:
- Feature names and descriptions
- Data types (integer, float, categorical)
- Valid ranges and categories
- Cross-feature constraints
- Class-conditional distributions
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

SCHEMA_DIR = Path(__file__).parent / "schemas"
SCHEMA_DIR.mkdir(exist_ok=True)


@dataclass
class FeatureSchema:
    """Schema for a single feature."""

    name: str
    description: str
    dtype: Literal["float", "integer", "categorical", "binary"]
    min_value: float | None = None
    max_value: float | None = None
    categories: list[str | int] | None = None
    unit: str | None = None
    # For class-conditional generation
    class_hints: dict[str, str] | None = (
        None  # e.g., {"malignant": "larger values", "benign": "smaller values"}
    )


@dataclass
class DatasetSchema:
    """Complete schema for a dataset.

    Attributes:
        name: Dataset identifier (e.g., "adult", "breast_cancer")
        description: Human-readable description of the dataset
        task_type: Either "classification" or "regression"
        n_features: Number of features in the dataset
        features: List of FeatureSchema objects describing each feature
        classes: Class names for classification tasks (e.g., ["<=50K", ">50K"])
        class_descriptions: Optional descriptions for each class
        constraints: Cross-feature constraints (e.g., "age >= 17")
        conditional_reasoning_questions: Questions for enhanced conditional generation
        enhanced_diversity_strategies: Diversity instructions for enhanced mode
        diversity_hints: General diversity hints for conditional generation

    Note:
        For conditional generation (Issue #43), schemas must define:
        - diversity_hints: Required for basic conditional generation
        - enhanced_diversity_strategies: Required for enhanced mode
        - conditional_reasoning_questions: Required for enhanced mode
    """

    name: str
    description: str
    task_type: Literal["classification", "regression"]
    n_features: int
    features: list[FeatureSchema]
    classes: list[str] | None = None  # For classification
    class_descriptions: dict[str, str] | None = None  # Description of each class
    constraints: list[str] = field(default_factory=list)  # Cross-feature constraints

    # Issue #43: Dataset-specific prompt configuration fields
    # These fields make LLM prompts dataset-agnostic by defining dataset-specific
    # instructions that are injected into prompts at generation time.
    #
    # conditional_reasoning_questions: Questions for chain-of-thought reasoning in
    #     enhanced conditional generation. Should be 2-4 questions specific to the
    #     dataset that help the LLM reason about feature relationships.
    #     Example: ["What education levels are common?", "What age ranges are typical?"]
    #
    # enhanced_diversity_strategies: Instructions for generating diverse samples in
    #     enhanced mode. Should cover different demographic/value segments.
    #     Example: ["Include samples from younger demographics (age 17-35).",
    #               "Include samples from older demographics (age 55+)."]
    #
    # diversity_hints: General hints for basic conditional generation to encourage
    #     diverse output. Rotated across batches using batch_id % len(hints).
    #     Example: ["Generate samples with varied ages and backgrounds.",
    #               "Create diverse samples covering different demographics."]
    #
    # Required for conditional generation: diversity_hints
    # Required for enhanced mode: all three fields
    conditional_reasoning_questions: list[str] = field(default_factory=list)
    enhanced_diversity_strategies: list[str] = field(default_factory=list)
    diversity_hints: list[str] = field(default_factory=list)

    def validate_for_conditional_generation(self, enhanced_mode: bool = False) -> None:
        """Validate schema has required fields for conditional generation.

        Args:
            enhanced_mode: Whether enhanced prompting will be used

        Raises:
            ValueError: If required fields are missing
        """
        if not self.diversity_hints:
            raise ValueError(
                f"Schema '{self.name}' must define 'diversity_hints' field. "
                f"This is required for conditional generation prompts (Issue #43)."
            )
        if enhanced_mode:
            if not self.enhanced_diversity_strategies:
                raise ValueError(
                    f"Schema '{self.name}' must define 'enhanced_diversity_strategies' field. "
                    f"This is required for enhanced conditional generation (Issue #43)."
                )
            if not self.conditional_reasoning_questions:
                raise ValueError(
                    f"Schema '{self.name}' must define 'conditional_reasoning_questions' field. "
                    f"This is required for enhanced conditional generation (Issue #43)."
                )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DatasetSchema:
        """Create from dictionary."""
        features = [FeatureSchema(**f) for f in d.pop("features")]
        return cls(features=features, **d)

    def get_fewshot_generation_prompt(
        self,
        fewshot_examples: list[dict],
        target_class: str | None = None,
        n_samples: int = 10,
        batch_id: int = 0,
        enhanced_mode: bool = False,
    ) -> str:
        """Generate a prompt with in-context few-shot examples.

        Args:
            fewshot_examples: List of example dicts with feature names as keys
            target_class: For classification, which class to generate
            n_samples: Number of samples to generate
            batch_id: Batch identifier for diversity
            enhanced_mode: If True, use chain-of-thought and edge case prompting

        Returns:
            Prompt string for LLM with in-context examples
        """
        if enhanced_mode:
            return self._get_enhanced_fewshot_prompt(
                fewshot_examples, target_class, n_samples, batch_id
            )

        # Build feature descriptions (brief version)
        feature_desc = []
        for f in self.features:
            desc = f"- {f.name}: {f.description}"
            if f.dtype == "categorical" and f.categories:
                desc += f" (categories: {f.categories})"
            elif f.dtype == "binary":
                desc += " (0 or 1)"
            elif f.dtype == "integer":
                if f.min_value is not None and f.max_value is not None:
                    desc += f" (integer, range: {int(f.min_value)}-{int(f.max_value)})"
            elif f.min_value is not None and f.max_value is not None:
                desc += f" (range: {f.min_value:.3f}-{f.max_value:.3f})"
            feature_desc.append(desc)

        feature_list = "\n".join(feature_desc)

        # Format few-shot examples as JSON
        examples_json = json.dumps(fewshot_examples, indent=2)

        # Class-specific instructions
        class_text = ""
        if target_class and self.class_descriptions:
            class_text = f"\n\nTarget class: '{target_class}' - {self.class_descriptions.get(target_class, '')}"

        # Diversity hint
        diversity_hints = [
            "Generate samples similar to the examples but with natural variation.",
            "Create new samples that follow the same patterns as the examples.",
            "Generate diverse samples consistent with the example distribution.",
            "Create realistic samples matching the example characteristics.",
        ]
        diversity_hint = diversity_hints[batch_id % len(diversity_hints)]

        return f"""You are generating synthetic tabular data based on real examples.

DATASET: {self.name} - {self.description}

FEATURES:
{feature_list}

REAL EXAMPLES FROM THE DATASET:
The following are {len(fewshot_examples)} real samples. Study their patterns, value distributions, and relationships between features:

```json
{examples_json}
```
{class_text}

TASK: {diversity_hint}
Generate {n_samples} NEW synthetic samples that:
1. Follow the same patterns and distributions as the examples above
2. Have realistic correlations between features
3. Stay within valid ranges but create NEW unique combinations
4. Are NOT exact copies of the examples - create variations

Return ONLY a valid JSON array with {n_samples} new samples:
```json
[
  {{"feature1": value1, "feature2": value2, ...}},
  ...
]
```

Generate {n_samples} new samples now:"""

    def _get_enhanced_fewshot_prompt(
        self,
        fewshot_examples: list[dict],
        target_class: str | None,
        n_samples: int,
        batch_id: int,
    ) -> str:
        """Generate enhanced prompt with chain-of-thought reasoning.

        This prompt asks the LLM to:
        1. First analyze the feature relationships in the examples
        2. Then generate diverse samples including edge cases
        3. Maintain realistic correlations between features
        """
        # Build feature descriptions with full ranges
        feature_desc = []
        for f in self.features:
            desc = f"- {f.name}: {f.description}"
            if f.dtype == "categorical" and f.categories:
                desc += f" (categories: {f.categories})"
            elif f.dtype == "binary":
                desc += " (0 or 1)"
            elif f.dtype == "integer":
                if f.min_value is not None and f.max_value is not None:
                    desc += f" (range: {int(f.min_value)}-{int(f.max_value)})"
            elif f.min_value is not None and f.max_value is not None:
                desc += f" (range: {f.min_value:.3f}-{f.max_value:.3f})"
            feature_desc.append(desc)

        feature_list = "\n".join(feature_desc)

        # Format few-shot examples as JSON
        examples_json = json.dumps(fewshot_examples, indent=2)

        # Class-specific context
        class_text = ""
        if target_class and self.class_descriptions:
            class_text = f"\nTarget class: '{target_class}' - {self.class_descriptions.get(target_class, '')}"

        # Diversity strategy based on batch_id
        diversity_strategies = [
            "Focus on edge cases with low/extreme values.",
            "Focus on edge cases with high/extreme values.",
            "Generate a diverse mix covering the full range.",
            "Include rare but valid feature combinations.",
        ]
        diversity_hint = diversity_strategies[batch_id % len(diversity_strategies)]

        return f"""You are generating synthetic tabular data. Before generating, THINK about the relationships between features.

DATASET: {self.name} - {self.description}{class_text}

FEATURES:
{feature_list}

EXAMPLES (study the relationships between features):
```json
{examples_json}
```

THINK STEP BY STEP:
1. What correlations exist between features? (e.g., if one feature increases, which others tend to increase/decrease?)
2. What constraints exist? (e.g., certain feature combinations that must go together or are impossible)
3. What is the full valid range for each feature? Don't just use "safe" middle values.

GENERATE {n_samples} samples that:
- Maintain realistic correlations you identified
- Use the FULL valid range (include edge cases, not just typical values)
- {diversity_hint}
- Are unique and not copies of examples

Return ONLY a valid JSON array:
```json
[{{"feature1": value1, ...}}, ...]
```"""

    def get_conditional_generation_prompt(
        self,
        fewshot_examples: list[dict],
        target_class: str | None = None,
        n_samples: int = 10,
        batch_id: int = 0,
        condition_feature: str = "",
        condition_value: str = "",
        enhanced_mode: bool = False,
    ) -> str:
        """Generate a prompt with conditional generation based on a specific feature value.

        This prompt instructs the LLM to generate samples where a specific feature
        has a fixed value, ensuring diverse stratified generation.

        Args:
            fewshot_examples: List of example dicts with feature names as keys
            target_class: For classification, which class to generate
            n_samples: Number of samples to generate
            batch_id: Batch identifier for diversity
            condition_feature: Feature name to condition on (REQUIRED)
            condition_value: Value that the condition_feature must have (REQUIRED)
            enhanced_mode: If True, use chain-of-thought prompting

        Returns:
            Prompt string for LLM with conditional generation instructions

        Raises:
            ValueError: If condition_feature/condition_value not provided
            ValueError: If schema doesn't define required fields
        """
        # Validate required parameters (Issue #43: removed Adult-specific defaults)
        if not condition_feature or not condition_value:
            raise ValueError(
                "condition_feature and condition_value are required parameters. "
                "These no longer have default values as of Issue #43."
            )

        # Validate schema has required fields
        self.validate_for_conditional_generation(enhanced_mode=enhanced_mode)

        if enhanced_mode:
            return self._get_enhanced_conditional_prompt(
                fewshot_examples, target_class, n_samples, batch_id,
                condition_feature, condition_value
            )

        # Build feature descriptions (brief version)
        feature_desc = []
        for f in self.features:
            desc = f"- {f.name}: {f.description}"
            if f.dtype == "categorical" and f.categories:
                desc += f" (categories: {f.categories})"
            elif f.dtype == "binary":
                desc += " (0 or 1)"
            elif f.dtype == "integer":
                if f.min_value is not None and f.max_value is not None:
                    desc += f" (integer, range: {int(f.min_value)}-{int(f.max_value)})"
            elif f.min_value is not None and f.max_value is not None:
                desc += f" (range: {f.min_value:.3f}-{f.max_value:.3f})"
            feature_desc.append(desc)

        feature_list = "\n".join(feature_desc)

        # Format few-shot examples as JSON
        examples_json = json.dumps(fewshot_examples, indent=2)

        # Class-specific instructions
        class_text = ""
        if target_class and self.class_descriptions:
            class_text = f"\n\nTarget class: '{target_class}' - {self.class_descriptions.get(target_class, '')}"

        # Use schema-defined diversity hints (Issue #43: dataset-agnostic)
        diversity_hint = self.diversity_hints[batch_id % len(self.diversity_hints)]

        return f"""You are generating synthetic tabular data based on real examples, with a SPECIFIC CONSTRAINT.

DATASET: {self.name} - {self.description}

FEATURES:
{feature_list}

REAL EXAMPLES FROM THE DATASET:
The following are {len(fewshot_examples)} real samples. Study their patterns, value distributions, and relationships between features:

```json
{examples_json}
```
{class_text}

CRITICAL CONSTRAINT:
ALL generated samples MUST have {condition_feature} = "{condition_value}"
This is a hard constraint - every sample must satisfy this requirement.

TASK: {diversity_hint}
Generate {n_samples} NEW synthetic samples that:
1. Have {condition_feature} = "{condition_value}" (REQUIRED for all samples)
2. Follow the same patterns and distributions as the examples above
3. Have realistic correlations between features given the {condition_feature}
4. Stay within valid ranges but create NEW unique combinations
5. Are NOT exact copies of the examples - create variations

Return ONLY a valid JSON array with {n_samples} new samples:
```json
[
  {{"{condition_feature}": "{condition_value}", "feature2": value2, ...}},
  ...
]
```

Generate {n_samples} new samples now (remember: ALL must have {condition_feature} = "{condition_value}"):"""

    def _get_enhanced_conditional_prompt(
        self,
        fewshot_examples: list[dict],
        target_class: str | None,
        n_samples: int,
        batch_id: int,
        condition_feature: str,
        condition_value: str,
    ) -> str:
        """Generate enhanced conditional prompt with chain-of-thought reasoning.

        This prompt asks the LLM to:
        1. First analyze what characteristics are typical for the condition_value
        2. Then generate diverse samples with that constraint
        3. Maintain realistic correlations
        """
        # Build feature descriptions with full ranges
        feature_desc = []
        for f in self.features:
            desc = f"- {f.name}: {f.description}"
            if f.dtype == "categorical" and f.categories:
                desc += f" (categories: {f.categories})"
            elif f.dtype == "binary":
                desc += " (0 or 1)"
            elif f.dtype == "integer":
                if f.min_value is not None and f.max_value is not None:
                    desc += f" (range: {int(f.min_value)}-{int(f.max_value)})"
            elif f.min_value is not None and f.max_value is not None:
                desc += f" (range: {f.min_value:.3f}-{f.max_value:.3f})"
            feature_desc.append(desc)

        feature_list = "\n".join(feature_desc)

        # Format few-shot examples as JSON
        examples_json = json.dumps(fewshot_examples, indent=2)

        # Class-specific context
        class_text = ""
        if target_class and self.class_descriptions:
            class_text = f"\nTarget class: '{target_class}' - {self.class_descriptions.get(target_class, '')}"

        # Use schema-defined diversity strategies (Issue #43: dataset-agnostic)
        diversity_hint = self.enhanced_diversity_strategies[
            batch_id % len(self.enhanced_diversity_strategies)
        ]

        # Build reasoning questions from schema (Issue #43: dataset-agnostic)
        reasoning_questions = "\n".join(
            f"   - {q}" for q in self.conditional_reasoning_questions
        )

        return f"""You are generating synthetic tabular data with a SPECIFIC CONSTRAINT. Before generating, THINK about what characteristics are typical for this constraint.

DATASET: {self.name} - {self.description}{class_text}

FEATURES:
{feature_list}

CONSTRAINT: All samples must have {condition_feature} = "{condition_value}"

EXAMPLES (study the relationships between features for samples with {condition_feature} = "{condition_value}"):
```json
{examples_json}
```

THINK STEP BY STEP:
1. What is typical for someone with {condition_feature} = "{condition_value}"?
{reasoning_questions}
2. What variations exist within this group? (There's diversity even within the same {condition_feature})
3. What realistic combinations of features would we expect?

GENERATE {n_samples} samples that:
- ALL have {condition_feature} = "{condition_value}" (REQUIRED)
- Maintain realistic correlations for samples with this {condition_feature}
- Show diversity within the group (not all samples should be identical profiles)
- {diversity_hint}
- Are unique and not copies of examples

Return ONLY a valid JSON array:
```json
[{{"{condition_feature}": "{condition_value}", ...}}, ...]
```"""

    def _generate_synthetic_examples(self, n_examples: int = 3, rng=None) -> list[dict]:
        """Generate synthetic diverse examples to show in the prompt.

        These examples demonstrate to the LLM what diverse data looks like,
        helping prevent the "template patient" problem in zero-shot generation.

        Args:
            n_examples: Number of synthetic examples to generate
            rng: Random number generator for reproducibility

        Returns:
            List of synthetic sample dictionaries with diverse values
        """
        import random as stdlib_random
        if rng is None:
            rng = stdlib_random.Random(42)

        examples = []
        for i in range(n_examples):
            sample = {}
            for f in self.features:
                if f.dtype == "categorical" and f.categories:
                    # Cycle through categories
                    sample[f.name] = f.categories[i % len(f.categories)]
                elif f.dtype == "binary":
                    # Alternate 0 and 1
                    sample[f.name] = i % 2
                elif f.dtype == "integer" and f.min_value is not None:
                    # Generate at low, mid, high points
                    range_val = f.max_value - f.min_value
                    if i == 0:
                        val = f.min_value + range_val * 0.2  # Low
                    elif i == 1:
                        val = f.min_value + range_val * 0.5  # Mid
                    else:
                        val = f.min_value + range_val * 0.8  # High
                    # Add some random variation
                    val += rng.uniform(-range_val * 0.1, range_val * 0.1)
                    sample[f.name] = int(max(f.min_value, min(f.max_value, val)))
                elif f.dtype == "float" and f.min_value is not None:
                    range_val = f.max_value - f.min_value
                    if i == 0:
                        val = f.min_value + range_val * 0.15
                    elif i == 1:
                        val = f.min_value + range_val * 0.55
                    else:
                        val = f.min_value + range_val * 0.85
                    val += rng.uniform(-range_val * 0.05, range_val * 0.05)
                    sample[f.name] = round(max(f.min_value, min(f.max_value, val)), 1)
                else:
                    # Fallback for features without ranges
                    sample[f.name] = 0

            examples.append(sample)

        return examples

    def get_generation_prompt(
        self,
        target_class: str | None = None,
        n_samples: int = 10,
        batch_id: int = 0,
        feature_stats: list | None = None,
        strict_constraints: bool = False,
    ) -> str:
        """Generate a prompt for LLM to create samples.

        Args:
            target_class: For classification, which class to generate
            n_samples: Number of samples to generate
            batch_id: Batch identifier for diversity (different batches get different hints)
            feature_stats: Optional list of FeatureStatistics for hard constraints
            strict_constraints: If True, inject STRICT constraint language from feature_stats

        Returns:
            Prompt string for LLM
        """
        import random

        # Use batch_id to seed randomness for this prompt
        rng = random.Random(batch_id * 12345 + 42)

        # Build feature descriptions with randomized range hints
        feature_desc = []
        for f in self.features:
            desc = f"- {f.name}: {f.description}"
            if f.dtype == "categorical" and f.categories:
                desc += f" (categories: {f.categories})"
            elif f.dtype == "binary":
                desc += " (0 or 1)"
            elif f.dtype == "integer":
                if f.min_value is not None and f.max_value is not None:
                    desc += f" (integer, range: {int(f.min_value)}-{int(f.max_value)})"
            elif f.min_value is not None and f.max_value is not None:
                desc += f" (range: {f.min_value:.3f}-{f.max_value:.3f})"
            if f.unit:
                desc += f" [{f.unit}]"
            if target_class and f.class_hints and target_class in f.class_hints:
                desc += f" | For {target_class}: {f.class_hints[target_class]}"
            feature_desc.append(desc)

        feature_list = "\n".join(feature_desc)

        # Build constraints section
        constraints_text = ""
        if self.constraints:
            constraints_text = "\n\nIMPORTANT CONSTRAINTS:\n" + "\n".join(
                f"- {c}" for c in self.constraints
            )

        # Class-specific instructions
        class_text = ""
        if target_class and self.class_descriptions:
            class_text = f"\n\nYou are generating samples for class '{target_class}': {self.class_descriptions.get(target_class, '')}"

        # Diversity instructions - vary per batch
        diversity_hints = [
            "Focus on generating samples from the LOWER end of valid ranges.",
            "Focus on generating samples from the UPPER end of valid ranges.",
            "Focus on generating samples near the MIDDLE of valid ranges.",
            "Generate a MIX of low, medium, and high values across features.",
            "Generate samples with HIGH variation between features.",
            "Generate samples representing TYPICAL cases.",
            "Generate samples representing EDGE cases near boundaries.",
            "Generate samples with CORRELATED feature patterns.",
        ]
        diversity_hint = diversity_hints[batch_id % len(diversity_hints)]

        # Random variation factors for this batch
        variation_seed = rng.randint(10000, 99999)

        # Build hard constraints section from statistics if provided
        strict_section = ""
        if feature_stats and strict_constraints:
            strict_section = "\n\nCRITICAL CONSTRAINTS - COMPUTED FROM REAL DATA:\n"
            strict_section += "="*70 + "\n"
            strict_section += f"These bounds are HARD CONSTRAINTS derived from real training samples.\n"
            strict_section += "ALL generated values MUST fall within these ranges.\n"
            strict_section += "Values outside these bounds are INVALID and will be rejected.\n"
            if target_class:
                strict_section += f"Statistics below are specific to class: {target_class}\n"
            strict_section += "="*70 + "\n\n"
            for stat in feature_stats:
                strict_section += stat.to_prompt_text(strict=True) + "\n"
            strict_section += "\n" + "="*70 + "\n"

        # Issue #44: Add CoT reasoning for zero-shot to improve diversity
        # Build explicit value distribution requirements for categorical/integer features
        value_dist_lines = []
        for f in self.features:
            if f.dtype == "categorical" and f.categories:
                cats = ", ".join(str(c) for c in f.categories)
                value_dist_lines.append(f"- {f.name}: MUST use ALL categories [{cats}] across your {n_samples} samples")
            elif f.dtype == "binary":
                value_dist_lines.append(f"- {f.name}: MUST use BOTH 0 AND 1 across your {n_samples} samples")
            elif f.dtype == "integer" and f.min_value is not None:
                value_dist_lines.append(f"- {f.name}: MUST vary from {int(f.min_value)} to {int(f.max_value)}, use LOW, MEDIUM, and HIGH values")
            elif f.dtype == "float" and f.min_value is not None:
                value_dist_lines.append(f"- {f.name}: MUST vary from {f.min_value:.1f} to {f.max_value:.1f}, use the FULL range")

        value_dist_section = "\n".join(value_dist_lines)

        # Generate synthetic diverse examples to show the model what diversity looks like
        synthetic_examples = self._generate_synthetic_examples(n_examples=3, rng=rng)
        synthetic_json = json.dumps(synthetic_examples, indent=2)

        cot_section = f"""
===== CRITICAL: DIVERSITY REQUIREMENT =====
Your output MUST look like these DIVERSE examples (all features vary):

{synthetic_json}

Notice how EVERY feature has DIFFERENT values in each sample!
DO NOT generate samples with constant values - they will be REJECTED.
===========================================

===== MANDATORY VALUE DISTRIBUTION =====
Each feature below MUST have different values across your {n_samples} samples:

{value_dist_section}
========================================
"""

        return f"""Generate {n_samples} DIVERSE synthetic samples for the {self.name} dataset.

DATASET: {self.description}
{cot_section}
DIVERSITY INSTRUCTION FOR THIS BATCH: {diversity_hint}
Variation seed: {variation_seed} (use this to inspire random choices)
{strict_section}
FEATURES (generate VARIED values for EACH feature):
{feature_list}{constraints_text}{class_text}

REQUIREMENTS:
1. Generate exactly {n_samples} samples with DIFFERENT values for ALL features
2. Each sample must have ALL {len(self.features)} features
3. Respect data types (integers must be whole numbers)
4. Use the FULL RANGE of valid values - low, medium, AND high
5. Each row should be UNIQUE with different values for EVERY feature
6. Categorical features: distribute across ALL valid categories
7. Return ONLY a valid JSON array, no explanations

Generate {n_samples} DIVERSE samples now (remember: vary ALL features, not just age):"""



def load_schema(dataset_name: str) -> DatasetSchema | None:
    """Load schema from cache.

    Args:
        dataset_name: Name of the dataset

    Returns:
        DatasetSchema if found, None otherwise
    """
    schema_path = SCHEMA_DIR / f"{dataset_name}.json"
    if not schema_path.exists():
        return None

    with open(schema_path) as f:
        data = json.load(f)

    return DatasetSchema.from_dict(data)


def save_schema(schema: DatasetSchema) -> Path:
    """Save schema to cache.

    Args:
        schema: Schema to save

    Returns:
        Path where schema was saved
    """
    schema_path = SCHEMA_DIR / f"{schema.name}.json"

    with open(schema_path, "w") as f:
        json.dump(schema.to_dict(), f, indent=2)

    return schema_path


# =============================================================================
# Pre-defined schemas for known datasets
# =============================================================================

BREAST_CANCER_SCHEMA = DatasetSchema(
    name="breast_cancer",
    description="Breast cancer Wisconsin dataset. Features computed from digitized images of fine needle aspirate (FNA) of breast masses, describing characteristics of cell nuclei.",
    task_type="classification",
    n_features=30,
    classes=["malignant", "benign"],
    class_descriptions={
        "malignant": "Cancerous tumors with irregular cell morphology, larger nuclei, and higher variation in measurements",
        "benign": "Non-cancerous tumors with more regular cell morphology, smaller nuclei, and lower variation",
    },
    constraints=[
        "For each characteristic X: X_worst >= X_mean (worst is mean of 3 largest values)",
        "For each characteristic X: X_se should be ~5-15% of X_mean (standard error)",
        "Geometric relationship: perimeter ≈ 2π × radius (approximately)",
        "Geometric relationship: area ≈ π × radius² (approximately)",
        "Malignant tumors typically have: larger radius (>14), higher concavity (>0.1), more concave points (>0.05)",
        "Benign tumors typically have: smaller radius (<14), lower concavity (<0.1), fewer concave points (<0.05)",
    ],
    features=[
        # Mean features (0-9)
        FeatureSchema(
            name="mean radius",
            description="Mean of distances from center to points on the perimeter",
            dtype="float",
            min_value=6.0,
            max_value=30.0,
            unit="arbitrary",
            class_hints={"malignant": "typically 15-28", "benign": "typically 6-15"},
        ),
        FeatureSchema(
            name="mean texture",
            description="Standard deviation of gray-scale values",
            dtype="float",
            min_value=9.0,
            max_value=40.0,
            class_hints={"malignant": "typically 17-35", "benign": "typically 10-22"},
        ),
        FeatureSchema(
            name="mean perimeter",
            description="Mean size of the core tumor perimeter",
            dtype="float",
            min_value=40.0,
            max_value=200.0,
            unit="arbitrary",
            class_hints={"malignant": "typically 90-190", "benign": "typically 40-100"},
        ),
        FeatureSchema(
            name="mean area",
            description="Mean area of the core tumor",
            dtype="float",
            min_value=140.0,
            max_value=2600.0,
            unit="arbitrary",
            class_hints={
                "malignant": "typically 500-2500",
                "benign": "typically 140-600",
            },
        ),
        FeatureSchema(
            name="mean smoothness",
            description="Mean of local variation in radius lengths",
            dtype="float",
            min_value=0.05,
            max_value=0.17,
            class_hints={
                "malignant": "typically 0.09-0.16",
                "benign": "typically 0.05-0.11",
            },
        ),
        FeatureSchema(
            name="mean compactness",
            description="Mean of perimeter²/area - 1.0",
            dtype="float",
            min_value=0.02,
            max_value=0.35,
            class_hints={
                "malignant": "typically 0.08-0.35",
                "benign": "typically 0.02-0.12",
            },
        ),
        FeatureSchema(
            name="mean concavity",
            description="Mean of severity of concave portions of the contour",
            dtype="float",
            min_value=0.0,
            max_value=0.45,
            class_hints={
                "malignant": "typically 0.08-0.43",
                "benign": "typically 0.0-0.1",
            },
        ),
        FeatureSchema(
            name="mean concave points",
            description="Mean number of concave portions of the contour",
            dtype="float",
            min_value=0.0,
            max_value=0.21,
            class_hints={
                "malignant": "typically 0.05-0.20",
                "benign": "typically 0.0-0.05",
            },
        ),
        FeatureSchema(
            name="mean symmetry",
            description="Mean symmetry of the cell nucleus",
            dtype="float",
            min_value=0.10,
            max_value=0.31,
            class_hints={
                "malignant": "typically 0.17-0.30",
                "benign": "typically 0.11-0.20",
            },
        ),
        FeatureSchema(
            name="mean fractal dimension",
            description="Mean of coastline approximation - 1",
            dtype="float",
            min_value=0.05,
            max_value=0.10,
            class_hints={
                "malignant": "typically 0.055-0.095",
                "benign": "typically 0.05-0.07",
            },
        ),
        # Standard Error features (10-19)
        FeatureSchema(
            name="radius error",
            description="Standard error of distances from center to perimeter",
            dtype="float",
            min_value=0.1,
            max_value=3.0,
            class_hints={
                "malignant": "typically 0.3-2.9",
                "benign": "typically 0.1-0.6",
            },
        ),
        FeatureSchema(
            name="texture error",
            description="Standard error of gray-scale values",
            dtype="float",
            min_value=0.3,
            max_value=5.0,
            class_hints={
                "malignant": "typically 0.5-4.9",
                "benign": "typically 0.3-1.8",
            },
        ),
        FeatureSchema(
            name="perimeter error",
            description="Standard error of tumor perimeter",
            dtype="float",
            min_value=0.7,
            max_value=22.0,
            class_hints={
                "malignant": "typically 2.0-21.0",
                "benign": "typically 0.8-4.0",
            },
        ),
        FeatureSchema(
            name="area error",
            description="Standard error of tumor area",
            dtype="float",
            min_value=6.0,
            max_value=550.0,
            class_hints={"malignant": "typically 25-540", "benign": "typically 6-50"},
        ),
        FeatureSchema(
            name="smoothness error",
            description="Standard error of local variation in radius lengths",
            dtype="float",
            min_value=0.002,
            max_value=0.032,
            class_hints={
                "malignant": "typically 0.005-0.03",
                "benign": "typically 0.003-0.01",
            },
        ),
        FeatureSchema(
            name="compactness error",
            description="Standard error of perimeter²/area - 1.0",
            dtype="float",
            min_value=0.002,
            max_value=0.14,
            class_hints={
                "malignant": "typically 0.01-0.14",
                "benign": "typically 0.005-0.04",
            },
        ),
        FeatureSchema(
            name="concavity error",
            description="Standard error of severity of concave portions",
            dtype="float",
            min_value=0.0,
            max_value=0.40,
            class_hints={
                "malignant": "typically 0.02-0.40",
                "benign": "typically 0.0-0.05",
            },
        ),
        FeatureSchema(
            name="concave points error",
            description="Standard error of number of concave portions",
            dtype="float",
            min_value=0.0,
            max_value=0.053,
            class_hints={
                "malignant": "typically 0.01-0.05",
                "benign": "typically 0.0-0.02",
            },
        ),
        FeatureSchema(
            name="symmetry error",
            description="Standard error of symmetry",
            dtype="float",
            min_value=0.008,
            max_value=0.08,
            class_hints={
                "malignant": "typically 0.015-0.07",
                "benign": "typically 0.01-0.03",
            },
        ),
        FeatureSchema(
            name="fractal dimension error",
            description="Standard error of coastline approximation - 1",
            dtype="float",
            min_value=0.001,
            max_value=0.03,
            class_hints={
                "malignant": "typically 0.002-0.03",
                "benign": "typically 0.001-0.006",
            },
        ),
        # Worst features (20-29)
        FeatureSchema(
            name="worst radius",
            description="Worst (largest) value of distances from center to perimeter",
            dtype="float",
            min_value=7.0,
            max_value=37.0,
            unit="arbitrary",
            class_hints={"malignant": "typically 18-36", "benign": "typically 7-18"},
        ),
        FeatureSchema(
            name="worst texture",
            description="Worst (largest) value of gray-scale standard deviation",
            dtype="float",
            min_value=12.0,
            max_value=50.0,
            class_hints={"malignant": "typically 22-49", "benign": "typically 12-28"},
        ),
        FeatureSchema(
            name="worst perimeter",
            description="Worst (largest) value of tumor perimeter",
            dtype="float",
            min_value=50.0,
            max_value=260.0,
            unit="arbitrary",
            class_hints={
                "malignant": "typically 115-255",
                "benign": "typically 50-115",
            },
        ),
        FeatureSchema(
            name="worst area",
            description="Worst (largest) value of tumor area",
            dtype="float",
            min_value=180.0,
            max_value=4300.0,
            unit="arbitrary",
            class_hints={
                "malignant": "typically 800-4250",
                "benign": "typically 180-800",
            },
        ),
        FeatureSchema(
            name="worst smoothness",
            description="Worst (largest) value of local variation in radius lengths",
            dtype="float",
            min_value=0.07,
            max_value=0.23,
            class_hints={
                "malignant": "typically 0.12-0.22",
                "benign": "typically 0.07-0.15",
            },
        ),
        FeatureSchema(
            name="worst compactness",
            description="Worst (largest) value of perimeter²/area - 1.0",
            dtype="float",
            min_value=0.03,
            max_value=1.1,
            class_hints={
                "malignant": "typically 0.15-1.05",
                "benign": "typically 0.03-0.25",
            },
        ),
        FeatureSchema(
            name="worst concavity",
            description="Worst (largest) value of severity of concave portions",
            dtype="float",
            min_value=0.0,
            max_value=1.3,
            class_hints={
                "malignant": "typically 0.20-1.25",
                "benign": "typically 0.0-0.25",
            },
        ),
        FeatureSchema(
            name="worst concave points",
            description="Worst (largest) value of number of concave portions",
            dtype="float",
            min_value=0.0,
            max_value=0.30,
            class_hints={
                "malignant": "typically 0.10-0.29",
                "benign": "typically 0.0-0.12",
            },
        ),
        FeatureSchema(
            name="worst symmetry",
            description="Worst (largest) value of symmetry",
            dtype="float",
            min_value=0.16,
            max_value=0.67,
            class_hints={
                "malignant": "typically 0.25-0.66",
                "benign": "typically 0.16-0.35",
            },
        ),
        FeatureSchema(
            name="worst fractal dimension",
            description="Worst (largest) value of coastline approximation - 1",
            dtype="float",
            min_value=0.055,
            max_value=0.21,
            class_hints={
                "malignant": "typically 0.07-0.21",
                "benign": "typically 0.055-0.10",
            },
        ),
    ],
)


# Iris dataset schema
IRIS_SCHEMA = DatasetSchema(
    name="iris",
    description="Iris flower measurements dataset for species classification",
    task_type="classification",
    n_features=4,
    classes=["setosa", "versicolor", "virginica"],
    class_descriptions={
        "setosa": "Iris setosa species, typically smallest measurements",
        "versicolor": "Iris versicolor species, medium-sized measurements",
        "virginica": "Iris virginica species, typically largest measurements",
    },
    features=[
        FeatureSchema(
            name="sepal length",
            description="Length of sepal in centimeters",
            dtype="float",
            min_value=4.3,
            max_value=7.9,
            unit="cm",
            class_hints={
                "setosa": "typically 4.3-5.8",
                "versicolor": "typically 4.9-7.0",
                "virginica": "typically 4.9-7.9",
            },
        ),
        FeatureSchema(
            name="sepal width",
            description="Width of sepal in centimeters",
            dtype="float",
            min_value=2.0,
            max_value=4.4,
            unit="cm",
            class_hints={
                "setosa": "typically 2.3-4.4",
                "versicolor": "typically 2.0-3.4",
                "virginica": "typically 2.2-3.8",
            },
        ),
        FeatureSchema(
            name="petal length",
            description="Length of petal in centimeters",
            dtype="float",
            min_value=1.0,
            max_value=6.9,
            unit="cm",
            class_hints={
                "setosa": "typically 1.0-1.9",
                "versicolor": "typically 3.0-5.1",
                "virginica": "typically 4.5-6.9",
            },
        ),
        FeatureSchema(
            name="petal width",
            description="Width of petal in centimeters",
            dtype="float",
            min_value=0.1,
            max_value=2.5,
            unit="cm",
            class_hints={
                "setosa": "typically 0.1-0.6",
                "versicolor": "typically 1.0-1.8",
                "virginica": "typically 1.4-2.5",
            },
        ),
    ],
)

# Diabetes regression dataset schema
DIABETES_SCHEMA = DatasetSchema(
    name="diabetes",
    description="Diabetes progression prediction from baseline measurements (regression)",
    task_type="regression",
    n_features=10,
    features=[
        FeatureSchema(
            name="age",
            description="Age normalized and mean-centered",
            dtype="float",
            min_value=-0.107,
            max_value=0.111,
        ),
        FeatureSchema(
            name="sex",
            description="Biological sex normalized and mean-centered",
            dtype="float",
            min_value=-0.045,
            max_value=0.051,
        ),
        FeatureSchema(
            name="bmi",
            description="Body mass index normalized and mean-centered",
            dtype="float",
            min_value=-0.090,
            max_value=0.171,
        ),
        FeatureSchema(
            name="bp",
            description="Average blood pressure normalized and mean-centered",
            dtype="float",
            min_value=-0.112,
            max_value=0.132,
        ),
        FeatureSchema(
            name="s1",
            description="Total serum cholesterol normalized and mean-centered",
            dtype="float",
            min_value=-0.127,
            max_value=0.154,
        ),
        FeatureSchema(
            name="s2",
            description="Low-density lipoproteins normalized and mean-centered",
            dtype="float",
            min_value=-0.116,
            max_value=0.199,
        ),
        FeatureSchema(
            name="s3",
            description="High-density lipoproteins normalized and mean-centered",
            dtype="float",
            min_value=-0.102,
            max_value=0.181,
        ),
        FeatureSchema(
            name="s4",
            description="Thyroid stimulating hormone normalized and mean-centered",
            dtype="float",
            min_value=-0.076,
            max_value=0.185,
        ),
        FeatureSchema(
            name="s5",
            description="Lamotrigine normalized and mean-centered",
            dtype="float",
            min_value=-0.126,
            max_value=0.134,
        ),
        FeatureSchema(
            name="s6",
            description="Blood sugar level normalized and mean-centered",
            dtype="float",
            min_value=-0.138,
            max_value=0.136,
        ),
    ],
)

# Heart Statlog dataset schema - Issue #44: Improved for zero-shot generation
HEART_STATLOG_SCHEMA = DatasetSchema(
    name="heart_statlog",
    description="Heart disease diagnosis from clinical and exercise test measurements. Patients undergo stress testing to evaluate cardiac function.",
    task_type="classification",
    n_features=13,
    classes=["no_disease", "disease"],
    class_descriptions={
        "no_disease": "No heart disease detected - patient has normal cardiac function with good exercise tolerance",
        "disease": "Heart disease present - patient shows signs of coronary artery disease or cardiac dysfunction",
    },
    # Issue #43: Dataset-specific prompt configuration
    conditional_reasoning_questions=[
        "What age ranges are typical for this condition?",
        "What blood pressure and cholesterol levels are common?",
        "What other clinical measurements correlate with this feature?",
        "How do exercise test results (max heart rate, ST depression, angina) relate to this?",
    ],
    enhanced_diversity_strategies=[
        "Include samples from younger patients (age 29-45) who may have early-onset risk factors.",
        "Include samples from middle-aged patients (age 45-60) with varying risk profiles.",
        "Include samples from older patients (age 60-77) with age-related cardiac changes.",
        "Generate a diverse mix covering athletes, sedentary individuals, and those with comorbidities.",
    ],
    diversity_hints=[
        "Generate samples with varied blood pressure (94-200 mmHg) and cholesterol (126-564 mg/dl) levels.",
        "Create diverse samples covering different max heart rates (71-202 bpm) during exercise.",
        "Generate samples with different ST depression values (0-6.2) and vessel involvement (0-3).",
        "Create realistic samples with varied chest pain presentations (types 1-4).",
        "Include samples with different ECG findings (normal, ST-T abnormality, LVH).",
        "Vary thalassemia results: some normal (3), some fixed defect (6), some reversible (7).",
    ],
    # Issue #44: Add medical correlations to improve LLM generation
    constraints=[
        "IMPORTANT: Each feature should vary independently - do NOT generate constant values.",
        "Age correlates with risk: older patients (>55) often have higher BP and cholesterol.",
        "Max heart rate decreases with age: typical max HR ≈ 220 - age (with variation ±20).",
        "Exercise-induced angina (1) often correlates with ST depression > 1.0.",
        "Chest pain type 4 (asymptomatic) is paradoxically more common in disease cases.",
        "More vessels colored (2-3) indicates more severe coronary artery disease.",
        "Fasting blood sugar > 120 (value=1) is present in ~15% of cardiac patients.",
        "ST slope: upsloping (1) is typically normal, flat (2) or downsloping (3) suggests ischemia.",
    ],
    features=[
        FeatureSchema(
            name="age",
            description="Patient age in years. Cardiac risk increases with age, especially after 45 for men and 55 for women",
            dtype="integer",
            min_value=29,
            max_value=77,
            unit="years",
        ),
        FeatureSchema(
            name="sex",
            description="Biological sex: 0=female, 1=male. Males have higher baseline cardiac risk",
            dtype="binary",
            categories=[0, 1],
        ),
        FeatureSchema(
            name="chest pain type",
            description="Type of chest pain experienced: 1=typical angina (classic exertional chest pain relieved by rest), 2=atypical angina (some but not all features of typical), 3=non-anginal pain (not cardiac in nature), 4=asymptomatic (no chest pain, often found incidentally)",
            dtype="integer",
            min_value=1,
            max_value=4,
        ),
        FeatureSchema(
            name="resting blood pressure",
            description="Resting systolic blood pressure at hospital admission. Normal <120, elevated 120-129, hypertension stage 1: 130-139, stage 2: ≥140",
            dtype="integer",
            min_value=94,
            max_value=200,
            unit="mm Hg",
        ),
        FeatureSchema(
            name="serum cholesterol",
            description="Total serum cholesterol level. Desirable <200, borderline 200-239, high ≥240 mg/dl. Higher levels increase cardiac risk",
            dtype="integer",
            min_value=126,
            max_value=564,
            unit="mg/dl",
        ),
        FeatureSchema(
            name="fasting blood sugar",
            description="Fasting blood sugar > 120 mg/dl indicates possible diabetes: 0=normal (≤120), 1=elevated (>120). About 15% of cardiac patients have this",
            dtype="binary",
            categories=[0, 1],
        ),
        FeatureSchema(
            name="resting ecg",
            description="Resting electrocardiogram results: 0=normal, 1=ST-T wave abnormality (T wave inversions or ST elevation/depression), 2=probable or definite left ventricular hypertrophy (enlarged heart)",
            dtype="integer",
            min_value=0,
            max_value=2,
        ),
        FeatureSchema(
            name="max heart rate",
            description="Maximum heart rate achieved during exercise stress test. Typical max HR = 220 - age. Lower than expected suggests poor cardiac reserve",
            dtype="integer",
            min_value=71,
            max_value=202,
            unit="bpm",
        ),
        FeatureSchema(
            name="exercise induced angina",
            description="Whether exercise caused chest pain (angina): 0=no angina during exercise, 1=angina induced by exercise. Presence suggests coronary artery disease",
            dtype="binary",
            categories=[0, 1],
        ),
        FeatureSchema(
            name="st depression",
            description="ST segment depression induced by exercise relative to rest, measured in mm. Values >1.0 suggest myocardial ischemia. Higher values indicate more severe disease",
            dtype="float",
            min_value=0.0,
            max_value=6.2,
        ),
        FeatureSchema(
            name="st slope",
            description="Slope of the peak exercise ST segment: 1=upsloping (typically normal response), 2=flat (borderline, may indicate ischemia), 3=downsloping (abnormal, strongly suggests ischemia)",
            dtype="integer",
            min_value=1,
            max_value=3,
        ),
        FeatureSchema(
            name="num vessels",
            description="Number of major coronary vessels (0-3) with >50% diameter narrowing seen on fluoroscopy. 0=no significant disease, 1-3=increasing severity of coronary artery disease",
            dtype="integer",
            min_value=0,
            max_value=3,
        ),
        FeatureSchema(
            name="thal",
            description="Thallium stress test result showing blood flow to heart muscle: 3=normal perfusion, 6=fixed defect (scar from prior heart attack, no blood flow at rest or stress), 7=reversible defect (reduced flow during stress but normal at rest, indicates ischemia)",
            dtype="categorical",
            categories=[3, 6, 7],
        ),
    ],
)

# Diabetes Pima dataset schema
# Issue #44: Improved diabetes schema with constraints and better descriptions
DIABETES_PIMA_SCHEMA = DatasetSchema(
    name="diabetes_pima",
    description="Pima Indians diabetes diagnosis from metabolic and demographic measurements. All patients are females of Pima Indian heritage, at least 21 years old.",
    task_type="classification",
    n_features=8,
    classes=["no_diabetes", "diabetes"],
    class_descriptions={
        "no_diabetes": "No diabetes detected - normal glucose metabolism and insulin response",
        "diabetes": "Diabetes present - impaired glucose tolerance or overt diabetes mellitus",
    },
    # Issue #43: Dataset-specific prompt configuration
    conditional_reasoning_questions=[
        "What glucose and insulin levels are typical for this metabolic state?",
        "What age ranges and pregnancy histories show these patterns?",
        "What BMI and blood pressure values correlate with diabetes risk?",
        "How do genetic factors (diabetes pedigree) relate to other measurements?",
    ],
    enhanced_diversity_strategies=[
        "Include samples from younger women (age 21-35) with varied pregnancy histories.",
        "Include samples from middle-aged women (age 35-55) with metabolic variations.",
        "Include samples from older women (age 55-81) with different risk profiles.",
        "Generate a diverse mix covering lean, overweight, and obese BMI ranges.",
    ],
    diversity_hints=[
        "Generate samples with varied glucose (44-199) and insulin (14-846) levels.",
        "Create diverse samples covering BMI from lean (18-25) to obese (35+).",
        "Generate samples with different blood pressure values (24-122 mmHg).",
        "Create realistic samples with varied pregnancy histories (0-17) and ages (21-81).",
        "Include different diabetes pedigree scores (0.08-2.42) showing genetic diversity.",
        "Vary skin thickness (7-99mm) which correlates with body fat.",
    ],
    # Issue #44: Add medical correlations to improve zero-shot generation
    constraints=[
        "IMPORTANT: Each feature should vary independently - do NOT generate constant values.",
        "Higher BMI (>30) often correlates with higher glucose and insulin resistance.",
        "Older age (>40) increases diabetes risk and often shows higher glucose levels.",
        "More pregnancies correlate with higher age (women can't have 10+ pregnancies at age 21).",
        "High glucose (>140) often correlates with high insulin (>200) due to insulin resistance.",
        "Diabetes pedigree reflects genetic risk - varies independently of other factors.",
        "Skin thickness correlates with BMI - higher BMI means thicker skin folds.",
        "Blood pressure tends to increase with age and BMI.",
    ],
    features=[
        FeatureSchema(
            name="pregnancies",
            description="Number of times pregnant. Correlates with age - younger women typically have fewer pregnancies. Range varies from nulliparous (0) to high parity (17)",
            dtype="integer",
            min_value=0,
            max_value=17,
        ),
        FeatureSchema(
            name="glucose",
            description="Plasma glucose concentration 2 hours after oral glucose tolerance test (OGTT). Normal <140, prediabetes 140-199, diabetes ≥200 mg/dl. Key diagnostic marker",
            dtype="integer",
            min_value=44,  # Issue #44: Zero values represent missing data in original dataset
            max_value=199,
            unit="mg/dl",
        ),
        FeatureSchema(
            name="blood pressure",
            description="Diastolic blood pressure (lower number). Normal <80, elevated 80-89, hypertension ≥90 mmHg. Correlates with age and BMI",
            dtype="integer",
            min_value=24,  # Issue #44: Zero values represent missing data in original dataset
            max_value=122,
            unit="mm Hg",
        ),
        FeatureSchema(
            name="skin thickness",
            description="Triceps skin fold thickness - measure of subcutaneous fat. Higher values indicate more body fat. Correlates with BMI",
            dtype="integer",
            min_value=7,  # Issue #44: Zero values represent missing data in original dataset
            max_value=99,
            unit="mm",
        ),
        FeatureSchema(
            name="insulin",
            description="2-Hour serum insulin level. Normal <100, elevated 100-200, high insulin resistance >200 mu U/ml. Higher values indicate insulin resistance",
            dtype="integer",
            min_value=14,  # Issue #44: Zero values represent missing data in original dataset
            max_value=846,
            unit="mu U/ml",
        ),
        FeatureSchema(
            name="bmi",
            description="Body mass index = weight(kg)/height(m)^2. Underweight <18.5, normal 18.5-25, overweight 25-30, obese >30. Major diabetes risk factor",
            dtype="float",
            min_value=18.2,  # Issue #44: Zero values represent missing data in original dataset
            max_value=67.1,
            unit="kg/m^2",
        ),
        FeatureSchema(
            name="diabetes pedigree",
            description="Diabetes pedigree function - scores genetic influence based on family history. Higher values (>0.5) indicate stronger genetic predisposition to diabetes",
            dtype="float",
            min_value=0.078,
            max_value=2.420,
        ),
        FeatureSchema(
            name="age",
            description="Age in years. All patients are at least 21. Diabetes risk increases significantly after age 40-45",
            dtype="integer",
            min_value=21,
            max_value=81,
            unit="years",
        ),
    ],
)


# Parkinson's Disease dataset schema (voice biomarkers)
PARKINSONS_SCHEMA = DatasetSchema(
    name="parkinsons",
    description="Biomedical voice measurements to distinguish healthy individuals from Parkinson's disease patients",
    task_type="classification",
    n_features=22,
    classes=["healthy", "parkinsons"],
    class_descriptions={
        "healthy": "Healthy individuals without Parkinson's disease",
        "parkinsons": "Patients diagnosed with Parkinson's disease",
    },
    features=[
        FeatureSchema(
            name="MDVP:Fo(Hz)",
            description="Average vocal fundamental frequency",
            dtype="float",
            min_value=88.0,
            max_value=260.0,
            unit="Hz",
        ),
        FeatureSchema(
            name="MDVP:Fhi(Hz)",
            description="Maximum vocal fundamental frequency",
            dtype="float",
            min_value=102.0,
            max_value=592.0,
            unit="Hz",
        ),
        FeatureSchema(
            name="MDVP:Flo(Hz)",
            description="Minimum vocal fundamental frequency",
            dtype="float",
            min_value=65.0,
            max_value=239.0,
            unit="Hz",
        ),
        FeatureSchema(
            name="MDVP:Jitter(%)",
            description="Percentage jitter (variation in fundamental frequency)",
            dtype="float",
            min_value=0.001,
            max_value=0.034,
        ),
        FeatureSchema(
            name="MDVP:Jitter(Abs)",
            description="Absolute jitter in microseconds",
            dtype="float",
            min_value=0.000007,
            max_value=0.00026,
            unit="μs",
        ),
        FeatureSchema(
            name="MDVP:RAP",
            description="Relative amplitude perturbation",
            dtype="float",
            min_value=0.0006,
            max_value=0.022,
        ),
        FeatureSchema(
            name="MDVP:PPQ",
            description="Five-point period perturbation quotient",
            dtype="float",
            min_value=0.0008,
            max_value=0.020,
        ),
        FeatureSchema(
            name="Jitter:DDP",
            description="Average absolute difference of differences between consecutive periods",
            dtype="float",
            min_value=0.002,
            max_value=0.066,
        ),
        FeatureSchema(
            name="MDVP:Shimmer",
            description="Measure of variation in amplitude",
            dtype="float",
            min_value=0.009,
            max_value=0.12,
        ),
        FeatureSchema(
            name="MDVP:Shimmer(dB)",
            description="Shimmer in decibels",
            dtype="float",
            min_value=0.085,
            max_value=1.30,
            unit="dB",
        ),
        FeatureSchema(
            name="Shimmer:APQ3",
            description="Three-point amplitude perturbation quotient",
            dtype="float",
            min_value=0.004,
            max_value=0.056,
        ),
        FeatureSchema(
            name="Shimmer:APQ5",
            description="Five-point amplitude perturbation quotient",
            dtype="float",
            min_value=0.006,
            max_value=0.079,
        ),
        FeatureSchema(
            name="MDVP:APQ",
            description="Amplitude perturbation quotient",
            dtype="float",
            min_value=0.007,
            max_value=0.137,
        ),
        FeatureSchema(
            name="Shimmer:DDA",
            description="Average absolute difference between consecutive amplitude differences",
            dtype="float",
            min_value=0.013,
            max_value=0.169,
        ),
        FeatureSchema(
            name="NHR",
            description="Noise-to-harmonics ratio",
            dtype="float",
            min_value=0.0006,
            max_value=0.31,
        ),
        FeatureSchema(
            name="HNR",
            description="Harmonics-to-noise ratio",
            dtype="float",
            min_value=8.4,
            max_value=33.0,
        ),
        FeatureSchema(
            name="RPDE",
            description="Recurrence period density entropy (nonlinear dynamical complexity)",
            dtype="float",
            min_value=0.26,
            max_value=0.69,
        ),
        FeatureSchema(
            name="DFA",
            description="Signal fractal scaling exponent (detrended fluctuation analysis)",
            dtype="float",
            min_value=0.57,
            max_value=0.83,
        ),
        FeatureSchema(
            name="spread1",
            description="Nonlinear measure of fundamental frequency variation",
            dtype="float",
            min_value=-7.96,
            max_value=-2.43,
        ),
        FeatureSchema(
            name="spread2",
            description="Nonlinear measure of fundamental frequency variation",
            dtype="float",
            min_value=0.006,
            max_value=0.45,
        ),
        FeatureSchema(
            name="D2",
            description="Correlation dimension (nonlinear dynamical complexity)",
            dtype="float",
            min_value=1.42,
            max_value=3.67,
        ),
        FeatureSchema(
            name="PPE",
            description="Pitch period entropy",
            dtype="float",
            min_value=0.044,
            max_value=0.53,
        ),
    ],
)

# Cholesterol (Cleveland Heart Disease) dataset schema - same features as heart_statlog
CHOLESTEROL_SCHEMA = DatasetSchema(
    name="cholesterol",
    description="Cleveland Heart Disease dataset for predicting heart disease severity",
    task_type="regression",
    n_features=13,
    features=[
        FeatureSchema(
            name="age",
            description="Age in years",
            dtype="integer",
            min_value=29,
            max_value=77,
            unit="years",
        ),
        FeatureSchema(
            name="sex",
            description="Biological sex",
            dtype="binary",
            min_value=0,
            max_value=1,
        ),
        FeatureSchema(
            name="chest_pain",
            description="Chest pain type (1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic)",
            dtype="integer",
            min_value=1,
            max_value=4,
        ),
        FeatureSchema(
            name="resting_bp",
            description="Resting blood pressure",
            dtype="integer",
            min_value=94,
            max_value=200,
            unit="mm Hg",
        ),
        FeatureSchema(
            name="cholesterol",
            description="Serum cholesterol level",
            dtype="integer",
            min_value=126,
            max_value=564,
            unit="mg/dl",
        ),
        FeatureSchema(
            name="fasting_bs",
            description="Fasting blood sugar > 120 mg/dl",
            dtype="binary",
            min_value=0,
            max_value=1,
        ),
        FeatureSchema(
            name="resting_ecg",
            description="Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)",
            dtype="integer",
            min_value=0,
            max_value=2,
        ),
        FeatureSchema(
            name="max_hr",
            description="Maximum heart rate achieved",
            dtype="integer",
            min_value=71,
            max_value=202,
            unit="bpm",
        ),
        FeatureSchema(
            name="exercise_angina",
            description="Exercise-induced angina",
            dtype="binary",
            min_value=0,
            max_value=1,
        ),
        FeatureSchema(
            name="oldpeak",
            description="ST depression induced by exercise",
            dtype="float",
            min_value=0.0,
            max_value=6.2,
        ),
        FeatureSchema(
            name="slope",
            description="Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)",
            dtype="integer",
            min_value=1,
            max_value=3,
        ),
        FeatureSchema(
            name="n_major_vessels",
            description="Number of major vessels colored by fluoroscopy",
            dtype="integer",
            min_value=0,
            max_value=3,
        ),
        FeatureSchema(
            name="thal",
            description="Thalassemia test result (3=normal, 6=fixed defect, 7=reversible defect)",
            dtype="categorical",
            categories=[3, 6, 7],
        ),
    ],
)

# Boston Housing dataset schema
BOSTON_HOUSING_SCHEMA = DatasetSchema(
    name="boston_housing",
    description="Housing data from Boston suburbs for predicting median home values",
    task_type="regression",
    n_features=13,
    features=[
        FeatureSchema(
            name="CRIM",
            description="Per capita crime rate by town",
            dtype="float",
            min_value=0.006,
            max_value=89.0,
        ),
        FeatureSchema(
            name="ZN",
            description="Proportion of residential land zoned for large lots",
            dtype="float",
            min_value=0.0,
            max_value=100.0,
        ),
        FeatureSchema(
            name="INDUS",
            description="Proportion of non-retail business acres per town",
            dtype="float",
            min_value=0.46,
            max_value=27.7,
        ),
        FeatureSchema(
            name="CHAS",
            description="Charles River dummy variable (1 if tract bounds river)",
            dtype="binary",
            min_value=0,
            max_value=1,
        ),
        FeatureSchema(
            name="NOX",
            description="Nitric oxides concentration (parts per 10 million)",
            dtype="float",
            min_value=0.39,
            max_value=0.87,
            unit="ppm×10",
        ),
        FeatureSchema(
            name="RM",
            description="Average number of rooms per dwelling",
            dtype="float",
            min_value=3.6,
            max_value=8.8,
        ),
        FeatureSchema(
            name="AGE",
            description="Proportion of owner-occupied units built before 1940",
            dtype="float",
            min_value=2.9,
            max_value=100.0,
        ),
        FeatureSchema(
            name="DIS",
            description="Weighted distances to five Boston employment centers",
            dtype="float",
            min_value=1.1,
            max_value=12.1,
        ),
        FeatureSchema(
            name="RAD",
            description="Index of accessibility to radial highways",
            dtype="integer",
            min_value=1,
            max_value=24,
        ),
        FeatureSchema(
            name="TAX",
            description="Full-value property-tax rate per $10,000",
            dtype="integer",
            min_value=187,
            max_value=711,
        ),
        FeatureSchema(
            name="PTRATIO",
            description="Pupil-teacher ratio by town",
            dtype="float",
            min_value=12.6,
            max_value=22.0,
        ),
        FeatureSchema(
            name="B",
            description="1000(Bk - 0.63)^2 where Bk is proportion of Black residents",
            dtype="float",
            min_value=0.32,
            max_value=396.9,
        ),
        FeatureSchema(
            name="LSTAT",
            description="Percentage of lower status population",
            dtype="float",
            min_value=1.7,
            max_value=38.0,
        ),
    ],
)

# Bike Sharing dataset schema
BIKE_SHARING_SCHEMA = DatasetSchema(
    name="bike_sharing",
    description="Hourly bike rental data from Capital Bikeshare system with weather and seasonal information",
    task_type="regression",
    n_features=12,
    features=[
        FeatureSchema(
            name="season",
            description="Season (1=spring, 2=summer, 3=fall, 4=winter)",
            dtype="integer",
            min_value=1,
            max_value=4,
        ),
        FeatureSchema(
            name="yr",
            description="Year (0=2011, 1=2012)",
            dtype="binary",
            min_value=0,
            max_value=1,
        ),
        FeatureSchema(
            name="mnth",
            description="Month (1-12)",
            dtype="integer",
            min_value=1,
            max_value=12,
        ),
        FeatureSchema(
            name="hr",
            description="Hour of day (0-23)",
            dtype="integer",
            min_value=0,
            max_value=23,
        ),
        FeatureSchema(
            name="holiday",
            description="Holiday flag",
            dtype="binary",
            min_value=0,
            max_value=1,
        ),
        FeatureSchema(
            name="weekday",
            description="Day of week (0=Sunday to 6=Saturday)",
            dtype="integer",
            min_value=0,
            max_value=6,
        ),
        FeatureSchema(
            name="workingday",
            description="Working day flag (1 if not weekend/holiday)",
            dtype="binary",
            min_value=0,
            max_value=1,
        ),
        FeatureSchema(
            name="weathersit",
            description="Weather situation (1=clear, 2=mist, 3=light rain, 4=heavy rain)",
            dtype="integer",
            min_value=1,
            max_value=4,
        ),
        FeatureSchema(
            name="temp",
            description="Normalized temperature",
            dtype="float",
            min_value=0.0,
            max_value=1.0,
        ),
        FeatureSchema(
            name="atemp",
            description="Normalized feeling temperature",
            dtype="float",
            min_value=0.0,
            max_value=1.0,
        ),
        FeatureSchema(
            name="hum",
            description="Normalized humidity",
            dtype="float",
            min_value=0.0,
            max_value=1.0,
        ),
        FeatureSchema(
            name="windspeed",
            description="Normalized wind speed",
            dtype="float",
            min_value=0.0,
            max_value=1.0,
        ),
    ],
)

# Grid Stability dataset schema
GRID_STABILITY_SCHEMA = DatasetSchema(
    name="grid_stability",
    description="Simulated data for a 4-node star electrical grid system to predict stability margin",
    task_type="regression",
    n_features=12,
    features=[
        FeatureSchema(
            name="tau1",
            description="Reaction time of participant 1",
            dtype="float",
            min_value=0.5,
            max_value=10.0,
        ),
        FeatureSchema(
            name="tau2",
            description="Reaction time of participant 2",
            dtype="float",
            min_value=0.5,
            max_value=10.0,
        ),
        FeatureSchema(
            name="tau3",
            description="Reaction time of participant 3",
            dtype="float",
            min_value=0.5,
            max_value=10.0,
        ),
        FeatureSchema(
            name="tau4",
            description="Reaction time of participant 4",
            dtype="float",
            min_value=0.5,
            max_value=10.0,
        ),
        FeatureSchema(
            name="p1",
            description="Nominal power of participant 1 (+ produced, - consumed)",
            dtype="float",
            min_value=-2.0,
            max_value=2.0,
        ),
        FeatureSchema(
            name="p2",
            description="Nominal power of participant 2",
            dtype="float",
            min_value=-2.0,
            max_value=2.0,
        ),
        FeatureSchema(
            name="p3",
            description="Nominal power of participant 3",
            dtype="float",
            min_value=-2.0,
            max_value=2.0,
        ),
        FeatureSchema(
            name="p4",
            description="Nominal power of participant 4",
            dtype="float",
            min_value=-2.0,
            max_value=2.0,
        ),
        FeatureSchema(
            name="g1",
            description="Price elasticity coefficient for participant 1",
            dtype="float",
            min_value=0.05,
            max_value=1.0,
        ),
        FeatureSchema(
            name="g2",
            description="Price elasticity coefficient for participant 2",
            dtype="float",
            min_value=0.05,
            max_value=1.0,
        ),
        FeatureSchema(
            name="g3",
            description="Price elasticity coefficient for participant 3",
            dtype="float",
            min_value=0.05,
            max_value=1.0,
        ),
        FeatureSchema(
            name="g4",
            description="Price elasticity coefficient for participant 4",
            dtype="float",
            min_value=0.05,
            max_value=1.0,
        ),
    ],
)

# Concrete Strength dataset schema
CONCRETE_STRENGTH_SCHEMA = DatasetSchema(
    name="concrete_strength",
    description="Data on concrete mixtures to predict compressive strength from ingredient quantities",
    task_type="regression",
    n_features=8,
    features=[
        FeatureSchema(
            name="cement",
            description="Cement content",
            dtype="float",
            min_value=102.0,
            max_value=540.0,
            unit="kg/m³",
        ),
        FeatureSchema(
            name="blast_furnace_slag",
            description="Blast furnace slag content",
            dtype="float",
            min_value=0.0,
            max_value=359.0,
            unit="kg/m³",
        ),
        FeatureSchema(
            name="fly_ash",
            description="Fly ash content",
            dtype="float",
            min_value=0.0,
            max_value=200.0,
            unit="kg/m³",
        ),
        FeatureSchema(
            name="water",
            description="Water content",
            dtype="float",
            min_value=122.0,
            max_value=247.0,
            unit="kg/m³",
        ),
        FeatureSchema(
            name="superplasticizer",
            description="Superplasticizer content",
            dtype="float",
            min_value=0.0,
            max_value=32.0,
            unit="kg/m³",
        ),
        FeatureSchema(
            name="coarse_aggregate",
            description="Coarse aggregate content",
            dtype="float",
            min_value=801.0,
            max_value=1145.0,
            unit="kg/m³",
        ),
        FeatureSchema(
            name="fine_aggregate",
            description="Fine aggregate content",
            dtype="float",
            min_value=594.0,
            max_value=992.0,
            unit="kg/m³",
        ),
        FeatureSchema(
            name="age",
            description="Age of concrete",
            dtype="integer",
            min_value=1,
            max_value=365,
            unit="days",
        ),
    ],
)

# Liver Disorders (BUPA) dataset schema
LIVER_DISORDERS_SCHEMA = DatasetSchema(
    name="liver_disorders",
    description="Blood test results related to liver function for studying alcohol consumption effects",
    task_type="regression",
    n_features=5,
    features=[
        FeatureSchema(
            name="mcv",
            description="Mean corpuscular volume",
            dtype="float",
            min_value=65.0,
            max_value=103.0,
            unit="fL",
        ),
        FeatureSchema(
            name="alkphos",
            description="Alkaline phosphatase",
            dtype="float",
            min_value=23.0,
            max_value=138.0,
            unit="IU/L",
        ),
        FeatureSchema(
            name="sgpt",
            description="Alanine aminotransferase (ALT)",
            dtype="float",
            min_value=4.0,
            max_value=155.0,
            unit="U/L",
        ),
        FeatureSchema(
            name="sgot",
            description="Aspartate aminotransferase (AST)",
            dtype="float",
            min_value=5.0,
            max_value=82.0,
            unit="U/L",
        ),
        FeatureSchema(
            name="gammagt",
            description="Gamma-glutamyl transpeptidase (GGT)",
            dtype="float",
            min_value=5.0,
            max_value=297.0,
            unit="U/L",
        ),
    ],
)

# Adult Census Income dataset schema
# Uses string categories for semantic LLM understanding (converted to indices in validator)
ADULT_SCHEMA = DatasetSchema(
    name="adult",
    description="Adult Census Income dataset for predicting whether income exceeds $50K/year based on census data. Features include demographics, education, occupation, and financial information.",
    task_type="classification",
    n_features=13,
    classes=["<=50K", ">50K"],
    class_descriptions={
        "<=50K": "Individual earns $50,000 or less per year",
        ">50K": "Individual earns more than $50,000 per year",
    },
    constraints=[
        "education-num should correspond to education level (e.g., Bachelors=13, HS-grad=9, Masters=14, Doctorate=16)",
        "If relationship is 'Husband', sex should be 'Male'",
        "If relationship is 'Wife', sex should be 'Female'",
        "capital-gain and capital-loss are typically mutually exclusive (one is usually 0)",
        "Higher education levels (Bachelors, Masters, Doctorate) correlate with professional occupations (Exec-managerial, Prof-specialty)",
        "Married individuals (Married-civ-spouse) are more likely to earn >50K",
    ],
    # Issue #43: Dataset-specific prompt configuration
    conditional_reasoning_questions=[
        "What education levels are common?",
        "What age ranges are typical?",
        "What other characteristics correlate with this feature?",
    ],
    enhanced_diversity_strategies=[
        "Include samples from younger demographics (age 17-35).",
        "Include samples from middle-aged demographics (age 35-55).",
        "Include samples from older demographics (age 55+).",
        "Generate a diverse mix covering the full age range.",
    ],
    diversity_hints=[
        "Generate samples with varied ages and backgrounds.",
        "Create diverse samples covering different demographics.",
        "Generate samples with different education levels and work hours.",
        "Create realistic samples with varied financial characteristics.",
    ],
    features=[
        FeatureSchema(
            name="age",
            description="Age of the individual in years",
            dtype="integer",
            min_value=17,
            max_value=90,
            unit="years",
            class_hints={
                "<=50K": "broader age range, includes many young workers (17-35)",
                ">50K": "typically prime working age (35-55), more experienced",
            },
        ),
        FeatureSchema(
            name="workclass",
            description="Type of employer or employment status",
            dtype="categorical",
            categories=[
                "Federal-gov",
                "Local-gov",
                "Never-worked",
                "Private",
                "Self-emp-inc",
                "Self-emp-not-inc",
                "State-gov",
                "Without-pay",
            ],
            class_hints={
                "<=50K": "more likely Private, Never-worked, Without-pay",
                ">50K": "more likely Self-emp-inc, Federal-gov, Private (managerial)",
            },
        ),
        FeatureSchema(
            name="education",
            description="Highest level of education achieved",
            dtype="categorical",
            categories=[
                "10th",
                "11th",
                "12th",
                "1st-4th",
                "5th-6th",
                "7th-8th",
                "9th",
                "Assoc-acdm",
                "Assoc-voc",
                "Bachelors",
                "Doctorate",
                "HS-grad",
                "Masters",
                "Preschool",
                "Prof-school",
                "Some-college",
            ],
            class_hints={
                "<=50K": "more likely HS-grad, Some-college, or lower education levels",
                ">50K": "more likely Bachelors, Masters, Doctorate, Prof-school",
            },
        ),
        FeatureSchema(
            name="education-num",
            description="Numeric encoding of education level (higher = more education). Maps to: Preschool=1, 1st-4th=2, 5th-6th=3, 7th-8th=4, 9th=5, 10th=6, 11th=7, 12th=8, HS-grad=9, Some-college=10, Assoc-voc=11, Assoc-acdm=12, Bachelors=13, Masters=14, Prof-school=15, Doctorate=16",
            dtype="integer",
            min_value=1,
            max_value=16,
            class_hints={
                "<=50K": "typically 1-12 (up to associate degree)",
                ">50K": "typically 10-16 (some college and above)",
            },
        ),
        FeatureSchema(
            name="marital-status",
            description="Marital status of the individual",
            dtype="categorical",
            categories=[
                "Divorced",
                "Married-AF-spouse",
                "Married-civ-spouse",
                "Married-spouse-absent",
                "Never-married",
                "Separated",
                "Widowed",
            ],
            class_hints={
                "<=50K": "more likely Never-married, Divorced, Separated, Widowed",
                ">50K": "more likely Married-civ-spouse, Married-AF-spouse",
            },
        ),
        FeatureSchema(
            name="occupation",
            description="Type of occupation or job role",
            dtype="categorical",
            categories=[
                "Adm-clerical",
                "Armed-Forces",
                "Craft-repair",
                "Exec-managerial",
                "Farming-fishing",
                "Handlers-cleaners",
                "Machine-op-inspct",
                "Other-service",
                "Priv-house-serv",
                "Prof-specialty",
                "Protective-serv",
                "Sales",
                "Tech-support",
                "Transport-moving",
            ],
            class_hints={
                "<=50K": "more likely Other-service, Handlers-cleaners, Farming-fishing, Machine-op-inspct, Adm-clerical",
                ">50K": "more likely Exec-managerial, Prof-specialty, Tech-support, Protective-serv, Sales",
            },
        ),
        FeatureSchema(
            name="relationship",
            description="Relationship status within household",
            dtype="categorical",
            categories=[
                "Husband",
                "Not-in-family",
                "Other-relative",
                "Own-child",
                "Unmarried",
                "Wife",
            ],
            class_hints={
                "<=50K": "more likely Own-child, Other-relative, Unmarried, Not-in-family",
                ">50K": "more likely Husband, Wife",
            },
        ),
        FeatureSchema(
            name="race",
            description="Race of the individual",
            dtype="categorical",
            categories=[
                "Amer-Indian-Eskimo",
                "Asian-Pac-Islander",
                "Black",
                "Other",
                "White",
            ],
        ),
        FeatureSchema(
            name="sex",
            description="Biological sex of the individual",
            dtype="categorical",
            categories=["Female", "Male"],
            class_hints={
                "<=50K": "relatively balanced between Male and Female",
                ">50K": "historically skewed toward Male in this dataset",
            },
        ),
        FeatureSchema(
            name="capital-gain",
            description="Capital gains from investments in the past year",
            dtype="integer",
            min_value=0,
            max_value=99999,
            unit="USD",
            class_hints={
                "<=50K": "typically 0 (no capital gains)",
                ">50K": "can be 0 or significant amounts (5000-99999)",
            },
        ),
        FeatureSchema(
            name="capital-loss",
            description="Capital losses from investments in the past year",
            dtype="integer",
            min_value=0,
            max_value=4356,
            unit="USD",
            class_hints={
                "<=50K": "typically 0",
                ">50K": "can be 0 or moderate amounts (1000-4000)",
            },
        ),
        FeatureSchema(
            name="hours-per-week",
            description="Number of hours worked per week",
            dtype="integer",
            min_value=1,
            max_value=99,
            unit="hours",
            class_hints={
                "<=50K": "typically 20-40 hours (part-time to standard)",
                ">50K": "typically 40-60 hours (full-time to overtime)",
            },
        ),
        FeatureSchema(
            name="native-country",
            description="Country of origin",
            dtype="categorical",
            categories=[
                "Cambodia",
                "Canada",
                "China",
                "Columbia",
                "Cuba",
                "Dominican-Republic",
                "Ecuador",
                "El-Salvador",
                "England",
                "France",
                "Germany",
                "Greece",
                "Guatemala",
                "Haiti",
                "Holand-Netherlands",
                "Honduras",
                "Hong",
                "Hungary",
                "India",
                "Iran",
                "Ireland",
                "Italy",
                "Jamaica",
                "Japan",
                "Laos",
                "Mexico",
                "Nicaragua",
                "Outlying-US(Guam-USVI-etc)",
                "Peru",
                "Philippines",
                "Poland",
                "Portugal",
                "Puerto-Rico",
                "Scotland",
                "South",
                "Taiwan",
                "Thailand",
                "Trinadad&Tobago",
                "United-States",
                "Vietnam",
                "Yugoslavia",
            ],
            class_hints={
                "<=50K": "United-States is most common, followed by Mexico and other countries",
                ">50K": "United-States is most common, with representation from various countries",
            },
        ),
    ],
)


# German Credit dataset schema (Issue #33)
# Financial attributes for credit risk assessment - ideal for LLM understanding
GERMAN_CREDIT_SCHEMA = DatasetSchema(
    name="german_credit",
    description="German Credit dataset for credit risk prediction based on financial and personal attributes. Contains loan applicant information used to assess creditworthiness.",
    task_type="classification",
    n_features=20,
    classes=["Good", "Bad"],
    class_descriptions={
        "Good": "Low credit risk - likely to repay loan on time",
        "Bad": "High credit risk - may default on loan",
    },
    constraints=[
        "checking_account status correlates with savings_account status",
        "Employment duration affects credit history assessment",
        "credit_amount should be realistic for the loan purpose",
        "Job type correlates with housing status and employment duration",
        "Younger applicants typically have shorter credit history",
    ],
    # Issue #43: Dataset-specific prompt configuration
    conditional_reasoning_questions=[
        "What credit amounts are typical for this loan purpose?",
        "What employment duration patterns are common?",
        "What account balances correlate with this feature?",
    ],
    enhanced_diversity_strategies=[
        "Include samples with shorter loan durations (4-18 months).",
        "Include samples with medium loan durations (18-36 months).",
        "Include samples with longer loan durations (36-72 months).",
        "Generate a diverse mix covering the full duration range.",
    ],
    diversity_hints=[
        "Generate samples with varied credit amounts and durations.",
        "Create diverse samples covering different employment situations.",
        "Generate samples with different savings and checking account statuses.",
        "Create realistic samples with varied housing and credit history.",
    ],
    features=[
        FeatureSchema(
            name="checking_account",
            description="Status of existing checking account",
            dtype="categorical",
            categories=["< 0 DM", "0-200 DM", ">= 200 DM", "no checking account"],
            class_hints={
                "Good": "more likely to have checking account with positive balance (>= 200 DM)",
                "Bad": "more likely to have negative balance or no checking account",
            },
        ),
        FeatureSchema(
            name="duration",
            description="Duration of the loan in months",
            dtype="integer",
            min_value=4,
            max_value=72,
            unit="months",
            class_hints={
                "Good": "typically shorter durations (6-24 months)",
                "Bad": "may have longer durations (24-72 months)",
            },
        ),
        FeatureSchema(
            name="credit_history",
            description="Credit history and payment behavior",
            dtype="categorical",
            categories=[
                "no credits/all paid",
                "all credits paid at this bank",
                "existing credits paid till now",
                "delay in past",
                "critical account",
            ],
            class_hints={
                "Good": "more likely 'existing credits paid till now' or 'all credits paid'",
                "Bad": "more likely 'delay in past' or 'critical account'",
            },
        ),
        FeatureSchema(
            name="purpose",
            description="Purpose of the loan",
            dtype="categorical",
            categories=[
                "car (new)",
                "car (used)",
                "furniture/equipment",
                "radio/television",
                "domestic appliances",
                "repairs",
                "education",
                "vacation",
                "retraining",
                "business",
                "others",
            ],
        ),
        FeatureSchema(
            name="credit_amount",
            description="Credit amount requested in Deutsche Mark (DM)",
            dtype="integer",
            min_value=250,
            max_value=18424,
            unit="DM",
            class_hints={
                "Good": "typically moderate amounts (1000-5000 DM)",
                "Bad": "may request larger amounts relative to income",
            },
        ),
        FeatureSchema(
            name="savings_account",
            description="Savings account/bonds balance",
            dtype="categorical",
            categories=[
                "< 100 DM",
                "100-500 DM",
                "500-1000 DM",
                ">= 1000 DM",
                "unknown/no savings",
            ],
            class_hints={
                "Good": "more likely to have savings (>= 500 DM)",
                "Bad": "more likely to have little or no savings",
            },
        ),
        FeatureSchema(
            name="employment_duration",
            description="Present employment duration",
            dtype="categorical",
            categories=[
                "unemployed",
                "< 1 year",
                "1-4 years",
                "4-7 years",
                ">= 7 years",
            ],
            class_hints={
                "Good": "more likely to have stable employment (>= 4 years)",
                "Bad": "more likely to be unemployed or recently employed",
            },
        ),
        FeatureSchema(
            name="installment_rate",
            description="Installment rate as percentage of disposable income",
            dtype="integer",
            min_value=1,
            max_value=4,
        ),
        FeatureSchema(
            name="personal_status_sex",
            description="Personal status and sex combined",
            dtype="categorical",
            categories=[
                "male: divorced/separated",
                "female: divorced/separated/married",
                "male: single",
                "male: married/widowed",
                "female: single",
            ],
        ),
        FeatureSchema(
            name="other_debtors",
            description="Other debtors or guarantors",
            dtype="categorical",
            categories=["none", "co-applicant", "guarantor"],
            class_hints={
                "Good": "may have guarantor support",
                "Bad": "more likely no guarantor",
            },
        ),
        FeatureSchema(
            name="present_residence",
            description="Present residence duration in years",
            dtype="integer",
            min_value=1,
            max_value=4,
        ),
        FeatureSchema(
            name="property",
            description="Type of property owned",
            dtype="categorical",
            categories=[
                "real estate",
                "building society savings/life insurance",
                "car or other",
                "unknown/no property",
            ],
            class_hints={
                "Good": "more likely to own real estate or have savings",
                "Bad": "more likely to have no property",
            },
        ),
        FeatureSchema(
            name="age",
            description="Age of the applicant in years",
            dtype="integer",
            min_value=19,
            max_value=75,
            unit="years",
            class_hints={
                "Good": "typically middle-aged (30-55) with established credit",
                "Bad": "may be younger (under 25) or older with financial difficulties",
            },
        ),
        FeatureSchema(
            name="other_installments",
            description="Other existing installment plans",
            dtype="categorical",
            categories=["bank", "stores", "none"],
        ),
        FeatureSchema(
            name="housing",
            description="Housing status",
            dtype="categorical",
            categories=["rent", "own", "for free"],
            class_hints={
                "Good": "more likely to own housing",
                "Bad": "more likely to rent",
            },
        ),
        FeatureSchema(
            name="num_credits",
            description="Number of existing credits at this bank",
            dtype="integer",
            min_value=1,
            max_value=4,
        ),
        FeatureSchema(
            name="job",
            description="Job type and skill level",
            dtype="categorical",
            categories=[
                "unemployed/unskilled non-resident",
                "unskilled resident",
                "skilled employee/official",
                "management/self-employed/highly qualified",
            ],
            class_hints={
                "Good": "more likely skilled or management positions",
                "Bad": "more likely unskilled or unemployed",
            },
        ),
        FeatureSchema(
            name="num_dependents",
            description="Number of people being liable to provide maintenance for",
            dtype="integer",
            min_value=1,
            max_value=2,
        ),
        FeatureSchema(
            name="telephone",
            description="Whether telephone is registered under applicant's name",
            dtype="categorical",
            categories=["none", "yes, registered"],
        ),
        FeatureSchema(
            name="foreign_worker",
            description="Whether the applicant is a foreign worker",
            dtype="categorical",
            categories=["yes", "no"],
        ),
    ],
)


# Registry of pre-defined schemas
SCHEMAS = {
    "adult": ADULT_SCHEMA,
    "german_credit": GERMAN_CREDIT_SCHEMA,
    "breast_cancer": BREAST_CANCER_SCHEMA,
    "iris": IRIS_SCHEMA,
    "diabetes": DIABETES_SCHEMA,
    "heart_statlog": HEART_STATLOG_SCHEMA,
    "diabetes_pima": DIABETES_PIMA_SCHEMA,
    "parkinsons": PARKINSONS_SCHEMA,
    "cholesterol": CHOLESTEROL_SCHEMA,
    "boston_housing": BOSTON_HOUSING_SCHEMA,
    "bike_sharing": BIKE_SHARING_SCHEMA,
    "grid_stability": GRID_STABILITY_SCHEMA,
    "concrete_strength": CONCRETE_STRENGTH_SCHEMA,
    "liver_disorders": LIVER_DISORDERS_SCHEMA,
}


def get_schema(dataset_name: str) -> DatasetSchema | None:
    """Get schema for a dataset (from cache or pre-defined).

    Args:
        dataset_name: Name of the dataset

    Returns:
        DatasetSchema if available, None otherwise
    """
    # First check cache
    cached = load_schema(dataset_name)
    if cached:
        return cached

    # Then check pre-defined schemas
    if dataset_name in SCHEMAS:
        return SCHEMAS[dataset_name]

    return None
