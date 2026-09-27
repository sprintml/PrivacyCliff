"""Feature metadata for conditional attribute inference attacks.

This module defines feature-level metadata for datasets used in privacy attacks,
including sensitivity levels, data types, and categorical information.

Sensitivity Levels (for Level 2/3 conditional attacks):
- sensitive: Private attributes the adversary wants to infer (target features)
- known: Attributes the adversary has access to (e.g., from public records, observation)
- irrelevant: Attributes the adversary doesn't know AND doesn't care about

In a conditional attack:
- Sensitive features → Adversary optimizes/reconstructs these
- Known features → Adversary fixes these at true values during optimization
- Irrelevant features → Adversary doesn't know these, but also doesn't target them
                        (can be ignored or optimized alongside sensitive features)

Example usage:
    from cascade_decoding.common.feature_metadata import (
        get_feature_indices_by_sensitivity,
        get_adult_feature_metadata,
        ADULT_FEATURES,
    )

    # Get indices by category
    sensitive_idx = get_sensitive_indices("adult")    # Features to infer
    known_idx = get_known_indices("adult")            # Features adversary has
    irrelevant_idx = get_irrelevant_indices("adult")  # Features to ignore

    # Get all metadata for Adult dataset
    metadata = get_adult_feature_metadata()
"""

from __future__ import annotations

from typing import Literal

SensitivityLevel = Literal["sensitive", "known", "irrelevant"]
FeatureDtype = Literal["integer", "categorical", "continuous", "binary"]


# =============================================================================
# Adult Census Income Dataset Feature Metadata
# =============================================================================
# Features after preprocessing (fnlwgt dropped, categoricals label-encoded)
# Order matches the output of prepare_adult_dataset.py

ADULT_FEATURES: dict[int, dict] = {
    # =========================================================================
    # KNOWN features - Adversary has access to these (public records, observable)
    # =========================================================================
    0: {
        "name": "age",
        "dtype": "integer",
        "sensitivity": "known",
        "description": "Age of the individual (observable, public records)",
        "min_value": 17,
        "max_value": 90,
    },
    2: {
        "name": "education",
        "dtype": "categorical",
        "sensitivity": "known",
        "description": "Highest level of education (resumes, public records)",
        "n_categories": 16,
        "categories": [
            "10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th",
            "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-grad",
            "Masters", "Preschool", "Prof-school", "Some-college"
        ],
    },
    7: {
        "name": "race",
        "dtype": "categorical",
        "sensitivity": "known",
        "description": "Race of the individual (observable)",
        "n_categories": 5,
        "categories": [
            "Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"
        ],
    },
    8: {
        "name": "sex",
        "dtype": "binary",
        "sensitivity": "known",
        "description": "Biological sex (observable)",
        "n_categories": 2,
        "categories": ["Female", "Male"],
    },
    12: {
        "name": "native-country",
        "dtype": "categorical",
        "sensitivity": "known",
        "description": "Country of origin (public records)",
        "n_categories": 42,
        "categories": None,  # Too many to list
    },
    # =========================================================================
    # SENSITIVE features - Adversary wants to infer these (target of attack)
    # =========================================================================
    4: {
        "name": "marital-status",
        "dtype": "categorical",
        "sensitivity": "sensitive",
        "description": "Marital status (private, target for inference)",
        "n_categories": 7,
        "categories": [
            "Divorced", "Married-AF-spouse", "Married-civ-spouse",
            "Married-spouse-absent", "Never-married", "Separated", "Widowed"
        ],
    },
    5: {
        "name": "occupation",
        "dtype": "categorical",
        "sensitivity": "sensitive",
        "description": "Type of occupation (private, target for inference)",
        "n_categories": 15,
        "categories": [
            "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
            "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct",
            "Other-service", "Priv-house-serv", "Prof-specialty", "Protective-serv",
            "Sales", "Tech-support", "Transport-moving", "?"
        ],
    },
    6: {
        "name": "relationship",
        "dtype": "categorical",
        "sensitivity": "sensitive",
        "description": "Relationship status in household (private, target for inference)",
        "n_categories": 6,
        "categories": [
            "Husband", "Not-in-family", "Other-relative", "Own-child",
            "Unmarried", "Wife"
        ],
    },
    # =========================================================================
    # IRRELEVANT features - Adversary doesn't know AND doesn't care about
    # =========================================================================
    1: {
        "name": "workclass",
        "dtype": "categorical",
        "sensitivity": "irrelevant",
        "description": "Type of employer (not easily known, not targeted)",
        "n_categories": 9,
        "categories": [
            "Federal-gov", "Local-gov", "Never-worked", "Private",
            "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay", "?"
        ],
    },
    3: {
        "name": "education-num",
        "dtype": "integer",
        "sensitivity": "irrelevant",
        "description": "Numeric representation of education level (derived, not targeted)",
        "min_value": 1,
        "max_value": 16,
    },
    9: {
        "name": "capital-gain",
        "dtype": "integer",
        "sensitivity": "irrelevant",
        "description": "Capital gains (financial, not known, not targeted)",
        "min_value": 0,
        "max_value": 99999,
    },
    10: {
        "name": "capital-loss",
        "dtype": "integer",
        "sensitivity": "irrelevant",
        "description": "Capital losses (financial, not known, not targeted)",
        "min_value": 0,
        "max_value": 4356,
    },
    11: {
        "name": "hours-per-week",
        "dtype": "integer",
        "sensitivity": "irrelevant",
        "description": "Hours worked per week (not easily known, not targeted)",
        "min_value": 1,
        "max_value": 99,
    },
}

# Target variable metadata
ADULT_TARGET = {
    "name": "income",
    "dtype": "binary",
    "sensitivity": "sensitive",
    "description": "Income >50K (1) or <=50K (0)",
    "categories": ["<=50K", ">50K"],
}


# =============================================================================
# Attack Configurations - Predefined sensitive/known feature sets
# =============================================================================
# These configs allow testing different threat models without modifying
# the base feature metadata. Use --config flag in run scripts to select.

ADULT_ATTACK_CONFIGS: dict[str, dict] = {
    "default": {
        # Default: 3 categorical sensitive features, age is known
        "description": "Standard setting - infer marital, occupation, relationship",
        "sensitive": [4, 5, 6],  # marital-status, occupation, relationship
        "known": [0, 2, 7, 8, 12],  # age, education, race, sex, native-country
        # irrelevant: [1, 3, 9, 10, 11] - workclass, education-num, capital-gain/loss, hours
    },
    "hard": {
        # Harder: age moved to sensitive (4 features including continuous)
        "description": "Hard setting - also infer age (tests feature entanglement)",
        "sensitive": [0, 4, 5, 6],  # age, marital-status, occupation, relationship
        "known": [2, 7, 8, 12],  # education, race, sex, native-country
    },
    "minimal": {
        # Minimal known features - harder attack
        "description": "Minimal known - only sex and race observable",
        "sensitive": [0, 4, 5, 6],  # age, marital-status, occupation, relationship
        "known": [7, 8],  # race, sex
    },
    "harder": {
        # Even harder: education also moved to sensitive (5 features to infer)
        "description": "Harder setting - infer age, education, marital, occupation, relationship",
        "sensitive": [0, 2, 4, 5, 6],  # age, education, marital-status, occupation, relationship
        "known": [7, 8, 12],  # race, sex, native-country
    },
}

# =============================================================================
# German Credit Dataset Feature Metadata
# =============================================================================
# Features from german.data after preprocessing
# Order matches prepare_german_credit_dataset.py output

GERMAN_CREDIT_FEATURES: dict[int, dict] = {
    # =========================================================================
    # KNOWN features - Adversary has access to these (observable, public records)
    # =========================================================================
    12: {
        "name": "age",
        "dtype": "integer",
        "sensitivity": "known",
        "description": "Age in years (observable)",
        "min_value": 19,
        "max_value": 75,
    },
    8: {
        "name": "personal_status_sex",
        "dtype": "categorical",
        "sensitivity": "known",
        "description": "Personal status and sex (observable)",
        "n_categories": 5,
    },
    14: {
        "name": "housing",
        "dtype": "categorical",
        "sensitivity": "known",
        "description": "Housing status (rent/own/free) - observable",
        "n_categories": 3,
    },
    16: {
        "name": "job",
        "dtype": "categorical",
        "sensitivity": "known",
        "description": "Job type (observable)",
        "n_categories": 4,
    },
    19: {
        "name": "foreign_worker",
        "dtype": "binary",
        "sensitivity": "known",
        "description": "Foreign worker status (public records)",
        "n_categories": 2,
    },
    # =========================================================================
    # SENSITIVE features - Adversary wants to infer these (financial privacy)
    # =========================================================================
    0: {
        "name": "checking_account",
        "dtype": "categorical",
        "sensitivity": "sensitive",
        "description": "Checking account status (financial, private)",
        "n_categories": 4,
    },
    2: {
        "name": "credit_history",
        "dtype": "categorical",
        "sensitivity": "sensitive",
        "description": "Credit history (financial, private)",
        "n_categories": 5,
    },
    5: {
        "name": "savings_account",
        "dtype": "categorical",
        "sensitivity": "sensitive",
        "description": "Savings account balance (financial, private)",
        "n_categories": 5,
    },
    6: {
        "name": "employment_duration",
        "dtype": "categorical",
        "sensitivity": "irrelevant",
        "description": "Employment duration (hard to infer, moved to irrelevant)",
        "n_categories": 5,
    },
    4: {
        "name": "credit_amount",
        "dtype": "integer",
        "sensitivity": "sensitive",
        "description": "Credit amount in DM (financial, private)",
        "min_value": 250,
        "max_value": 18424,
    },
    # =========================================================================
    # IRRELEVANT features - Adversary doesn't know AND doesn't care about
    # =========================================================================
    1: {
        "name": "duration",
        "dtype": "integer",
        "sensitivity": "irrelevant",
        "description": "Loan duration in months",
        "min_value": 4,
        "max_value": 72,
    },
    3: {
        "name": "purpose",
        "dtype": "categorical",
        "sensitivity": "irrelevant",
        "description": "Loan purpose",
        "n_categories": 11,
    },
    7: {
        "name": "installment_rate",
        "dtype": "integer",
        "sensitivity": "irrelevant",
        "description": "Installment rate as % of disposable income",
        "min_value": 1,
        "max_value": 4,
    },
    9: {
        "name": "other_debtors",
        "dtype": "categorical",
        "sensitivity": "irrelevant",
        "description": "Other debtors/guarantors",
        "n_categories": 3,
    },
    10: {
        "name": "present_residence",
        "dtype": "integer",
        "sensitivity": "irrelevant",
        "description": "Present residence duration",
        "min_value": 1,
        "max_value": 4,
    },
    11: {
        "name": "property",
        "dtype": "categorical",
        "sensitivity": "irrelevant",
        "description": "Property type",
        "n_categories": 4,
    },
    13: {
        "name": "other_installments",
        "dtype": "categorical",
        "sensitivity": "irrelevant",
        "description": "Other installment plans",
        "n_categories": 3,
    },
    15: {
        "name": "num_credits",
        "dtype": "integer",
        "sensitivity": "sensitive",
        "description": "Number of existing credits at this bank (financial, private)",
        "min_value": 1,
        "max_value": 4,
    },
    17: {
        "name": "num_dependents",
        "dtype": "integer",
        "sensitivity": "irrelevant",
        "description": "Number of dependents",
        "min_value": 1,
        "max_value": 2,
    },
    18: {
        "name": "telephone",
        "dtype": "binary",
        "sensitivity": "irrelevant",
        "description": "Telephone registered",
        "n_categories": 2,
    },
}

GERMAN_CREDIT_TARGET = {
    "name": "credit_risk",
    "dtype": "binary",
    "description": "Credit risk (0=Good, 1=Bad)",
    "categories": ["Good", "Bad"],
}

GERMAN_CREDIT_ATTACK_CONFIGS: dict[str, dict] = {
    "default": {
        # Default: Financial sensitive features
        "description": "Standard - infer financial attributes (checking, credit_history, savings, num_credits)",
        "sensitive": [0, 2, 5, 15],  # checking_account, credit_history, savings_account, num_credits
        "known": [8, 12, 14, 16, 19],  # personal_status_sex, age, housing, job, foreign_worker
    },
    "hard": {
        # Hard: Add credit_amount (continuous)
        "description": "Hard - also infer credit amount",
        "sensitive": [0, 2, 4, 5, 15],  # + credit_amount
        "known": [8, 12, 14, 16, 19],
    },
    "minimal": {
        # Minimal known features
        "description": "Minimal known - only age and sex observable",
        "sensitive": [0, 2, 4, 5, 15],
        "known": [8, 12],  # personal_status_sex, age
    },
}


# =============================================================================
# Heart Disease Dataset Feature Metadata
# =============================================================================
# Features from processed.cleveland.data after preprocessing
# Supports multiple attack goal dimensions (medical/clinical/diagnostic)

HEART_DISEASE_FEATURES: dict[int, dict] = {
    # =========================================================================
    # Demographics - Known features
    # =========================================================================
    0: {
        "name": "age",
        "dtype": "integer",
        "sensitivity": "known",
        "description": "Age in years (observable)",
        "min_value": 29,
        "max_value": 77,
    },
    1: {
        "name": "sex",
        "dtype": "binary",
        "sensitivity": "known",
        "description": "Sex (0=female, 1=male) - observable",
        "n_categories": 2,
    },
    # =========================================================================
    # Clinical symptoms
    # =========================================================================
    2: {
        "name": "chest_pain_type",
        "dtype": "categorical",
        "sensitivity": "sensitive",
        "description": "Chest pain type (clinical symptom)",
        "n_categories": 4,
    },
    8: {
        "name": "exercise_angina",
        "dtype": "binary",
        "sensitivity": "sensitive",
        "description": "Exercise induced angina (clinical symptom)",
        "n_categories": 2,
    },
    12: {
        "name": "thal",
        "dtype": "categorical",
        "sensitivity": "sensitive",
        "description": "Thalassemia type (clinical)",
        "n_categories": 3,
    },
    # =========================================================================
    # Medical test results
    # =========================================================================
    3: {
        "name": "resting_bp",
        "dtype": "integer",
        "sensitivity": "sensitive",
        "description": "Resting blood pressure (medical test)",
        "min_value": 94,
        "max_value": 200,
    },
    4: {
        "name": "cholesterol",
        "dtype": "integer",
        "sensitivity": "sensitive",
        "description": "Serum cholesterol (medical test)",
        "min_value": 126,
        "max_value": 564,
    },
    5: {
        "name": "fasting_blood_sugar",
        "dtype": "binary",
        "sensitivity": "sensitive",
        "description": "Fasting blood sugar > 120 (medical test)",
        "n_categories": 2,
    },
    7: {
        "name": "max_heart_rate",
        "dtype": "integer",
        "sensitivity": "sensitive",
        "description": "Maximum heart rate achieved (medical test)",
        "min_value": 71,
        "max_value": 202,
    },
    # =========================================================================
    # Diagnostic indicators
    # =========================================================================
    9: {
        "name": "oldpeak",
        "dtype": "continuous",
        "sensitivity": "sensitive",
        "description": "ST depression induced by exercise (diagnostic)",
        "min_value": 0.0,
        "max_value": 6.2,
    },
    10: {
        "name": "slope",
        "dtype": "categorical",
        "sensitivity": "sensitive",
        "description": "Slope of peak exercise ST segment (diagnostic)",
        "n_categories": 3,
    },
    11: {
        "name": "num_vessels",
        "dtype": "categorical",
        "sensitivity": "sensitive",
        "description": "Number of major vessels colored (diagnostic)",
        "n_categories": 4,
    },
    # =========================================================================
    # Irrelevant (derived/less targeted)
    # =========================================================================
    6: {
        "name": "resting_ecg",
        "dtype": "categorical",
        "sensitivity": "irrelevant",
        "description": "Resting ECG results",
        "n_categories": 3,
    },
}

HEART_DISEASE_TARGET = {
    "name": "target",
    "dtype": "binary",
    "description": "Heart disease presence (0=No, 1=Yes)",
    "categories": ["No Disease", "Disease"],
}

# Heart Disease has two dimensions: attack goal type AND difficulty level
HEART_DISEASE_ATTACK_CONFIGS: dict[str, dict] = {
    # Dimension 1: Attack goal type
    "medical": {
        # Infer medical test results
        "description": "Medical goal - infer test results (cholesterol, blood sugar, BP, HR)",
        "sensitive": [3, 4, 5, 7],  # resting_bp, cholesterol, fasting_blood_sugar, max_heart_rate
        "known": [0, 1],  # age, sex
    },
    "clinical": {
        # Infer clinical symptoms
        "description": "Clinical goal - infer symptoms (chest pain, angina, thal)",
        "sensitive": [2, 8, 12],  # chest_pain_type, exercise_angina, thal
        "known": [0, 1, 3],  # age, sex, resting_bp
    },
    "diagnostic": {
        # Infer diagnostic indicators
        "description": "Diagnostic goal - infer advanced indicators (vessels, oldpeak, slope)",
        "sensitive": [9, 10, 11],  # oldpeak, slope, num_vessels
        "known": [0, 1, 2],  # age, sex, chest_pain_type
    },
    # Dimension 2: Difficulty levels (combined targets)
    "default": {
        # Default: Mix of medical and clinical (4 sensitive features)
        "description": "Default - infer cholesterol, blood sugar, chest pain, angina",
        "sensitive": [2, 4, 5, 8],  # chest_pain_type, cholesterol, fasting_blood_sugar, exercise_angina
        "known": [0, 1],  # age, sex
    },
    "hard": {
        # Hard: More sensitive features (6 features)
        "description": "Hard - also infer BP, max HR, thal",
        "sensitive": [2, 3, 4, 5, 7, 8],  # + resting_bp, max_heart_rate
        "known": [0, 1],  # age, sex
    },
    "minimal": {
        # Minimal known - hardest
        "description": "Minimal known - only age observable",
        "sensitive": [2, 4, 5, 8, 12],
        "known": [0],  # only age
    },
}


# =============================================================================
# Diabetes Pima Dataset Feature Metadata
# =============================================================================
# Features from pima-indians-diabetes dataset
# All features are continuous/integer

DIABETES_PIMA_FEATURES: dict[int, dict] = {
    # =========================================================================
    # KNOWN features - Demographics/observable
    # =========================================================================
    7: {
        "name": "age",
        "dtype": "integer",
        "sensitivity": "known",
        "description": "Age in years (observable)",
        "min_value": 21,
        "max_value": 81,
    },
    # =========================================================================
    # SENSITIVE features - Medical history (private)
    # =========================================================================
    0: {
        "name": "pregnancies",
        "dtype": "integer",
        "sensitivity": "sensitive",
        "description": "Number of pregnancies (medical history, private)",
        "min_value": 0,
        "max_value": 17,
    },
    6: {
        "name": "diabetes_pedigree",
        "dtype": "continuous",
        "sensitivity": "sensitive",
        "description": "Diabetes pedigree function (genetic/family history, private)",
        "min_value": 0.078,
        "max_value": 2.42,
    },
    # =========================================================================
    # Health metrics (can be sensitive or irrelevant based on config)
    # =========================================================================
    1: {
        "name": "glucose",
        "dtype": "integer",
        "sensitivity": "irrelevant",
        "description": "Plasma glucose concentration (health metric)",
        "min_value": 44,
        "max_value": 199,
    },
    2: {
        "name": "blood_pressure",
        "dtype": "integer",
        "sensitivity": "irrelevant",
        "description": "Diastolic blood pressure (health metric)",
        "min_value": 24,
        "max_value": 122,
    },
    3: {
        "name": "skin_thickness",
        "dtype": "integer",
        "sensitivity": "irrelevant",
        "description": "Triceps skin fold thickness (clinical)",
        "min_value": 7,
        "max_value": 99,
    },
    4: {
        "name": "insulin",
        "dtype": "integer",
        "sensitivity": "irrelevant",
        "description": "2-Hour serum insulin (health metric)",
        "min_value": 14,
        "max_value": 846,
    },
    5: {
        "name": "bmi",
        "dtype": "continuous",
        "sensitivity": "irrelevant",
        "description": "Body mass index (health metric)",
        "min_value": 18.2,
        "max_value": 67.1,
    },
}

DIABETES_PIMA_TARGET = {
    "name": "outcome",
    "dtype": "binary",
    "description": "Diabetes (0=No, 1=Yes)",
    "categories": ["No Diabetes", "Diabetes"],
}

DIABETES_PIMA_ATTACK_CONFIGS: dict[str, dict] = {
    "default": {
        # Default: Medical history sensitive
        "description": "Standard - infer medical history (pregnancies, pedigree, age)",
        "sensitive": [0, 6, 7],  # pregnancies, diabetes_pedigree, age
        "known": [1, 5],  # glucose, bmi (commonly measured)
    },
    "hard": {
        # Hard: Add more sensitive features
        "description": "Hard - also infer insulin and blood pressure",
        "sensitive": [0, 2, 4, 6, 7],  # + blood_pressure, insulin
        "known": [1, 5],  # glucose, bmi
    },
    "health_metrics": {
        # Health metrics as target
        "description": "Health goal - infer glucose, insulin, BP, BMI",
        "sensitive": [1, 2, 4, 5],  # glucose, blood_pressure, insulin, bmi
        "known": [7],  # only age
    },
    "minimal": {
        # Minimal known
        "description": "Minimal known - only BMI observable",
        "sensitive": [0, 1, 2, 4, 6, 7],  # everything except bmi
        "known": [5],  # only bmi
    },
}


# =============================================================================
# Dynamic Config Utility
# =============================================================================

def make_attack_config(
    dataset: str,
    sensitive_features: list[str],
    known_features: list[str],
) -> dict:
    """Create a custom attack config from feature names.

    This utility allows flexible ablation studies by specifying which
    features should be sensitive vs known by name.

    Args:
        dataset: Dataset name (e.g., "german_credit")
        sensitive_features: List of feature names to treat as sensitive
        known_features: List of feature names to treat as known

    Returns:
        Dictionary with 'sensitive' and 'known' index lists

    Example:
        >>> config = make_attack_config(
        ...     "german_credit",
        ...     sensitive_features=["credit_history", "savings_account"],
        ...     known_features=["age", "job"],
        ... )
    """
    metadata = get_feature_metadata(dataset)

    # Build name -> index mapping
    name_to_idx = {info["name"]: idx for idx, info in metadata.items()}

    # Convert names to indices
    sensitive_indices = []
    for name in sensitive_features:
        if name not in name_to_idx:
            raise ValueError(f"Unknown feature '{name}' in dataset '{dataset}'")
        sensitive_indices.append(name_to_idx[name])

    known_indices = []
    for name in known_features:
        if name not in name_to_idx:
            raise ValueError(f"Unknown feature '{name}' in dataset '{dataset}'")
        known_indices.append(name_to_idx[name])

    # Validate no overlap
    overlap = set(sensitive_indices) & set(known_indices)
    if overlap:
        overlap_names = [metadata[idx]["name"] for idx in overlap]
        raise ValueError(f"Features cannot be both sensitive and known: {overlap_names}")

    return {
        "description": f"Custom config: sensitive={sensitive_features}, known={known_features}",
        "sensitive": sorted(sensitive_indices),
        "known": sorted(known_indices),
    }


DATASET_ATTACK_CONFIGS: dict[str, dict[str, dict]] = {
    "adult": ADULT_ATTACK_CONFIGS,
    "german_credit": GERMAN_CREDIT_ATTACK_CONFIGS,
    "heart_disease": HEART_DISEASE_ATTACK_CONFIGS,
    "diabetes_pima": DIABETES_PIMA_ATTACK_CONFIGS,
}


# =============================================================================
# Registry of all dataset feature metadata
# =============================================================================

DATASET_FEATURE_METADATA: dict[str, dict[int, dict]] = {
    "adult": ADULT_FEATURES,
    "german_credit": GERMAN_CREDIT_FEATURES,
    "heart_disease": HEART_DISEASE_FEATURES,
    "diabetes_pima": DIABETES_PIMA_FEATURES,
}

DATASET_TARGET_METADATA: dict[str, dict] = {
    "adult": ADULT_TARGET,
    "german_credit": GERMAN_CREDIT_TARGET,
    "heart_disease": HEART_DISEASE_TARGET,
    "diabetes_pima": DIABETES_PIMA_TARGET,
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_feature_metadata(dataset: str) -> dict[int, dict]:
    """Get feature metadata for a dataset.

    Args:
        dataset: Dataset name (e.g., "adult")

    Returns:
        Dictionary mapping feature index to metadata dict

    Raises:
        ValueError: If dataset not found
    """
    if dataset not in DATASET_FEATURE_METADATA:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {list(DATASET_FEATURE_METADATA.keys())}"
        )
    return DATASET_FEATURE_METADATA[dataset]


def get_feature_indices_by_sensitivity(
    dataset: str,
    sensitivity: SensitivityLevel,
) -> list[int]:
    """Get feature indices by sensitivity level.

    Args:
        dataset: Dataset name
        sensitivity: One of "sensitive", "known", "irrelevant"

    Returns:
        List of feature indices with the specified sensitivity (sorted)
    """
    metadata = get_feature_metadata(dataset)
    return sorted([
        idx for idx, info in metadata.items()
        if info.get("sensitivity") == sensitivity
    ])


def get_feature_indices_by_dtype(
    dataset: str,
    dtype: FeatureDtype,
) -> list[int]:
    """Get feature indices by data type.

    Args:
        dataset: Dataset name
        dtype: One of "integer", "categorical", "continuous", "binary"

    Returns:
        List of feature indices with the specified data type
    """
    metadata = get_feature_metadata(dataset)
    return [
        idx for idx, info in metadata.items()
        if info.get("dtype") == dtype
    ]


def get_sensitive_indices(dataset: str, config: str | None = None) -> list[int]:
    """Get indices of sensitive features (targets for inference attack).

    Args:
        dataset: Dataset name
        config: Optional attack config name. If provided, uses config's sensitive list.
                If None, uses base metadata sensitivity labels.
    """
    if config is not None:
        return get_config_sensitive_indices(dataset, config)
    return get_feature_indices_by_sensitivity(dataset, "sensitive")


def get_known_indices(dataset: str, config: str | None = None) -> list[int]:
    """Get indices of known features (adversary has access to these).

    Args:
        dataset: Dataset name
        config: Optional attack config name. If provided, uses config's known list.
                If None, uses base metadata sensitivity labels.
    """
    if config is not None:
        return get_config_known_indices(dataset, config)
    return get_feature_indices_by_sensitivity(dataset, "known")


def get_irrelevant_indices(dataset: str, config: str | None = None) -> list[int]:
    """Get indices of irrelevant features (adversary doesn't know or care).

    Args:
        dataset: Dataset name
        config: Optional attack config name. If provided, computes from config.
                If None, uses base metadata sensitivity labels.
    """
    if config is not None:
        return get_config_irrelevant_indices(dataset, config)
    return get_feature_indices_by_sensitivity(dataset, "irrelevant")


def get_categorical_indices(dataset: str) -> list[int]:
    """Get indices of categorical features (including binary)."""
    cat_idx = get_feature_indices_by_dtype(dataset, "categorical")
    bin_idx = get_feature_indices_by_dtype(dataset, "binary")
    return sorted(cat_idx + bin_idx)


def get_feature_name(dataset: str, idx: int) -> str:
    """Get feature name by index."""
    metadata = get_feature_metadata(dataset)
    if idx not in metadata:
        raise ValueError(f"Feature index {idx} not found in {dataset}")
    return metadata[idx]["name"]


def get_feature_dtype(dataset: str, idx: int) -> FeatureDtype:
    """Get feature dtype by index."""
    metadata = get_feature_metadata(dataset)
    if idx not in metadata:
        raise ValueError(f"Feature index {idx} not found in {dataset}")
    return metadata[idx]["dtype"]


def get_feature_n_categories(dataset: str, idx: int) -> int | None:
    """Get number of categories for a categorical feature."""
    metadata = get_feature_metadata(dataset)
    if idx not in metadata:
        raise ValueError(f"Feature index {idx} not found in {dataset}")
    return metadata[idx].get("n_categories")


def get_adult_feature_metadata() -> dict[int, dict]:
    """Shorthand for Adult dataset metadata."""
    return ADULT_FEATURES


def print_feature_summary(dataset: str, config: str | None = None) -> None:
    """Print a summary of feature metadata for a dataset."""
    metadata = get_feature_metadata(dataset)

    # Get config-specific indices if config is provided
    sensitive = get_sensitive_indices(dataset, config)
    known = get_known_indices(dataset, config)
    irrelevant = get_irrelevant_indices(dataset, config)

    # Build sensitivity lookup from config
    sensitivity_lookup = {}
    for idx in sensitive:
        sensitivity_lookup[idx] = "sensitive"
    for idx in known:
        sensitivity_lookup[idx] = "known"
    for idx in irrelevant:
        sensitivity_lookup[idx] = "irrelevant"

    print(f"\n{'='*70}")
    print(f"Feature Summary: {dataset}")
    if config:
        print(f"Attack Config: {config}")
    print(f"{'='*70}")

    print(f"\n{'Idx':<4} {'Name':<20} {'Type':<12} {'Sensitivity':<16}")
    print("-" * 56)

    for idx, info in sorted(metadata.items()):
        # Use config-specific sensitivity if available, else base metadata
        sensitivity = sensitivity_lookup.get(idx, info.get('sensitivity', 'unknown'))
        print(
            f"{idx:<4} {info['name']:<20} {info['dtype']:<12} "
            f"{sensitivity:<16}"
        )

    print(f"\nSensitive features (to infer):   {sensitive}")
    print(f"Known features (adversary has):  {known}")
    print(f"Irrelevant features (ignored):   {irrelevant}")


# =============================================================================
# Attack Configuration Functions
# =============================================================================

def get_attack_config(dataset: str, config: str) -> dict:
    """Get attack configuration by name.

    Args:
        dataset: Dataset name (e.g., "adult")
        config: Config name (e.g., "default", "hard", "minimal")

    Returns:
        Dictionary with 'sensitive', 'known' keys containing feature indices

    Raises:
        ValueError: If dataset or config not found
    """
    if dataset not in DATASET_ATTACK_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {list(DATASET_ATTACK_CONFIGS.keys())}"
        )
    configs = DATASET_ATTACK_CONFIGS[dataset]
    if config not in configs:
        raise ValueError(
            f"Unknown config: {config}. "
            f"Available for {dataset}: {list(configs.keys())}"
        )
    return configs[config]


def list_attack_configs(dataset: str) -> list[str]:
    """List available attack configurations for a dataset."""
    if dataset not in DATASET_ATTACK_CONFIGS:
        return []
    return list(DATASET_ATTACK_CONFIGS[dataset].keys())


def get_config_sensitive_indices(dataset: str, config: str) -> list[int]:
    """Get sensitive feature indices for a specific attack config."""
    cfg = get_attack_config(dataset, config)
    return sorted(cfg["sensitive"])


def get_config_known_indices(dataset: str, config: str) -> list[int]:
    """Get known feature indices for a specific attack config."""
    cfg = get_attack_config(dataset, config)
    return sorted(cfg["known"])


def get_config_irrelevant_indices(dataset: str, config: str) -> list[int]:
    """Get irrelevant feature indices for a specific attack config.

    Irrelevant = all features - sensitive - known
    """
    cfg = get_attack_config(dataset, config)
    metadata = get_feature_metadata(dataset)
    all_indices = set(metadata.keys())
    return sorted(all_indices - set(cfg["sensitive"]) - set(cfg["known"]))


# =============================================================================
# K-S Configuration Support
# =============================================================================
# Priority lists for dynamic K-S configuration (from exp_plan.md Section 2.2)
# - Public Priority: Features ordered by public availability (most public first)
# - Sensitive Priority: Features ordered by sensitivity (most sensitive first)

# Adult Census Income (13 features)
# Feature indices: 0=age, 1=workclass, 2=education, 3=education-num, 4=marital-status,
#                  5=occupation, 6=relationship, 7=race, 8=sex, 9=capital-gain,
#                  10=capital-loss, 11=hours-per-week, 12=native-country
# Priority lists aligned with original attack configs:
# - K=3 -> known=[7,8,12]=[race, sex, native-country] (matches "harder" config)
# - S=5 -> sensitive=[5,0,2,4,6]=[occupation, age, education, marital, relationship]
# Age (0) and education (2) are both sensitive for "harder" config
ADULT_PUBLIC_PRIORITY = [7, 8, 12, 2, 3, 1, 11, 9, 10, 4, 5, 6]
ADULT_SENSITIVE_PRIORITY = [5, 0, 2, 4, 6, 9, 10, 11, 1, 3, 12, 7, 8]

# German Credit (20 features)
# Feature indices: 0=checking_account, 1=duration, 2=credit_history, 3=purpose,
#                  4=credit_amount, 5=savings_account, 6=employment_since, 7=installment_rate,
#                  8=personal_status_sex, 9=other_debtors, 10=present_residence, 11=property,
#                  12=age, 13=other_installment_plans, 14=housing, 15=num_credits, 16=job,
#                  17=num_dependents, 18=telephone, 19=foreign_worker
# Priority lists aligned with original attack configs:
# - K=5 -> known=[19,12,8,14,16]=[foreign_worker, age, personal_status, housing, job] (matches "hard" config)
# - S=5 -> sensitive=[4,0,5,2,15]=[credit_amount, checking, savings, credit_history, num_credits]
GERMAN_CREDIT_PUBLIC_PRIORITY = [19, 12, 8, 14, 16, 18, 17, 10, 11, 13, 9, 6, 15, 7, 0, 5, 2, 3, 1, 4]
GERMAN_CREDIT_SENSITIVE_PRIORITY = [4, 0, 5, 2, 15, 3, 1, 7, 6, 9, 13, 11, 10, 17, 16, 14, 8, 12, 18, 19]

# Heart Disease Cleveland (13 features)
# Feature indices: 0=age, 1=sex, 2=chest_pain_type, 3=resting_bp, 4=cholesterol,
#                  5=fasting_blood_sugar, 6=rest_ecg, 7=max_heart_rate, 8=exercise_angina,
#                  9=oldpeak, 10=slope, 11=num_major_vessels, 12=thal
# Priority lists aligned with original attack configs:
# - K=2 -> known=[1,0]=[sex, age] (matches "hard" config)
# UPDATED (Issue #44): Reordered to put binary/categorical features first,
# continuous/large-range features last (cholesterol R²=-0.52, resting_bp R²=-0.25)
# New order: binary -> categorical -> small integer -> large integer -> continuous
HEART_DISEASE_PUBLIC_PRIORITY = [1, 0, 2, 5, 6, 8, 10, 12, 11, 9, 7, 3, 4]
HEART_DISEASE_SENSITIVE_PRIORITY = [5, 8, 2, 10, 12, 11, 6, 7, 3, 4, 9, 0, 1]
# 5=fasting_blood_sugar (binary), 8=exercise_angina (binary), 2=chest_pain_type (cat 4),
# 10=slope (cat 3), 12=thal (cat 3), 11=num_vessels (0-3), 6=resting_ecg (cat 3),
# 7=max_heart_rate (71-202), 3=resting_bp (94-200), 4=cholesterol (126-564),
# 9=oldpeak (continuous 0-6.2), 0=age, 1=sex

# Diabetes Pima (8 features)
# Feature indices: 0=pregnancies, 1=glucose, 2=blood_pressure, 3=skin_thickness,
#                  4=insulin, 5=bmi, 6=diabetes_pedigree, 7=age
# Priority lists aligned with original attack configs:
# - K=2 -> known=[1,5]=[glucose, bmi] (matches "hard" config)
# UPDATED (Issue #44): Reordered to put small-range integers first,
# continuous features last (diabetes_pedigree R²=0.18 was poor)
# New order: small integer -> medium integer -> large integer -> continuous
DIABETES_PIMA_PUBLIC_PRIORITY = [1, 5, 7, 0, 6, 3, 2, 4]
DIABETES_PIMA_SENSITIVE_PRIORITY = [0, 7, 2, 1, 3, 4, 6, 5]
# 0=pregnancies (0-17, R²=0.68), 7=age (21-81, R²=0.93), 2=blood_pressure (24-122, R²=0.58),
# 1=glucose (44-199), 3=skin_thickness (7-99), 4=insulin (14-846, R²=0.76),
# 6=diabetes_pedigree (continuous, R²=0.18), 5=bmi (continuous)

# Registry of priority lists
DATASET_PUBLIC_PRIORITY: dict[str, list[int]] = {
    "adult": ADULT_PUBLIC_PRIORITY,
    "german_credit": GERMAN_CREDIT_PUBLIC_PRIORITY,
    "heart_disease": HEART_DISEASE_PUBLIC_PRIORITY,
    "diabetes_pima": DIABETES_PIMA_PUBLIC_PRIORITY,
}

DATASET_SENSITIVE_PRIORITY: dict[str, list[int]] = {
    "adult": ADULT_SENSITIVE_PRIORITY,
    "german_credit": GERMAN_CREDIT_SENSITIVE_PRIORITY,
    "heart_disease": HEART_DISEASE_SENSITIVE_PRIORITY,
    "diabetes_pima": DIABETES_PIMA_SENSITIVE_PRIORITY,
}

# Number of features per dataset
DATASET_N_FEATURES: dict[str, int] = {
    "adult": 13,
    "german_credit": 20,
    "heart_disease": 13,
    "diabetes_pima": 8,
}


def get_ks_config(dataset: str, k: int, s: int) -> dict:
    """Generate attack configuration from K-S values using priority lists.

    The K-S configuration system allows flexible specification of attack difficulty:
    - K = number of known (public) features the adversary has access to
    - S = number of sensitive features the adversary wants to infer

    Args:
        dataset: Dataset name (e.g., "adult", "german_credit")
        k: Number of known features (taken from public priority list)
        s: Number of sensitive features (taken from sensitive priority list)

    Returns:
        Dictionary with:
        - "known": list of known feature indices
        - "sensitive": list of sensitive feature indices
        - "irrelevant": list of irrelevant feature indices
        - "description": human-readable description

    Raises:
        ValueError: If k + s > total features or invalid dataset

    Example:
        >>> config = get_ks_config("adult", k=3, s=3)
        >>> config["known"]
        [7, 8, 12]  # race, sex, native-country
        >>> config["sensitive"]
        [5, 4, 6]   # occupation, marital-status, relationship
    """
    if dataset not in DATASET_PUBLIC_PRIORITY:
        raise ValueError(
            f"Unknown dataset: {dataset}. "
            f"Available: {list(DATASET_PUBLIC_PRIORITY.keys())}"
        )

    n_features = DATASET_N_FEATURES[dataset]
    if k + s > n_features:
        raise ValueError(
            f"K + S ({k} + {s} = {k+s}) exceeds total features ({n_features}) "
            f"for dataset '{dataset}'"
        )

    public_priority = DATASET_PUBLIC_PRIORITY[dataset]
    sensitive_priority = DATASET_SENSITIVE_PRIORITY[dataset]

    # Get top K from public priority as known features
    known_indices = public_priority[:k]

    # Get top S from sensitive priority, excluding features already in known
    sensitive_indices = []
    for idx in sensitive_priority:
        if idx not in known_indices:
            sensitive_indices.append(idx)
            if len(sensitive_indices) >= s:
                break

    # Remaining features are irrelevant
    all_indices = set(range(n_features))
    irrelevant_indices = sorted(all_indices - set(known_indices) - set(sensitive_indices))

    # Get feature names for description
    metadata = get_feature_metadata(dataset)
    known_names = [metadata[idx]["name"] for idx in known_indices]
    sensitive_names = [metadata[idx]["name"] for idx in sensitive_indices]

    return {
        "description": f"K{k}-S{s}: known={known_names}, sensitive={sensitive_names}",
        "known": known_indices,
        "sensitive": sensitive_indices,
        "irrelevant": irrelevant_indices,
        "k": k,
        "s": s,
    }


def get_ks_feature_names(dataset: str, k: int, s: int) -> tuple[list[str], list[str]]:
    """Get feature names for a K-S configuration.

    Args:
        dataset: Dataset name
        k: Number of known features
        s: Number of sensitive features

    Returns:
        Tuple of (known_feature_names, sensitive_feature_names)
    """
    config = get_ks_config(dataset, k, s)
    metadata = get_feature_metadata(dataset)
    known_names = [metadata[idx]["name"] for idx in config["known"]]
    sensitive_names = [metadata[idx]["name"] for idx in config["sensitive"]]
    return known_names, sensitive_names


def get_max_ks_values(dataset: str) -> tuple[int, int]:
    """Get maximum K and S values for a dataset.

    Returns:
        Tuple of (max_k, max_s) where max_k + max_s <= n_features
    """
    n_features = DATASET_N_FEATURES.get(dataset, 0)
    # Reasonable maxes: K up to 5, S up to 4 for most datasets
    # Diabetes (8 features): K up to 3, S up to 4
    if dataset == "diabetes_pima":
        return (3, 4)
    else:
        return (5, 4)
