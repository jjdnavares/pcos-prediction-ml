"""
Feature engineering and preprocessing utilities
"""
import numpy as np
import pandas as pd
from typing import Dict

def engineer_features(patient_data: Dict) -> Dict:
    """
    Apply domain-specific feature engineering.

    Creates derived features that were used during model training:
    - Total Follicle Count
    - Symptom Burden

    Args:
        patient_data: Dictionary of raw patient features

    Returns:
        Dictionary with engineered features added
    """

    # Make a copy to avoid modifying original
    features = patient_data.copy()

    # Total Follicle Count
    features['Total_Follicle_Count'] = features['follicle_no_l'] + features['follicle_no_r']

    # Symptom Burden (count of physical symptoms)
    symptom_count = (
        features.get('skin_darkening', 0) +
        features.get('hair_growth', 0) +
        features.get('weight_gain', 0) +
        features.get('hair_loss', 0)
    )
    features['Symptom_Burden'] = symptom_count

    return features

def prepare_model_input(patient_data: Dict, feature_names: list) -> np.ndarray:
    """
    Convert patient data to model input format.

    Args:
        patient_data: Dictionary of patient features (with engineered features)
        feature_names: List of feature names in the correct order

    Returns:
        numpy array ready for model.predict()
    """

    # Map API field names to model feature names
    # This mapping handles differences between API naming (snake_case)
    # and model training naming (original dataset format)

    field_mapping = {
        'age': 'Age (yrs)',
        'vit_d3': 'Vit D3 (ng/mL)',
        'follicle_no_l': 'Follicle No. (L)',
        'follicle_no_r': 'Follicle No. (R)',
        'avg_f_size_r': 'Avg. F size (R) (mm)',
        'endometrium': 'Endometrium (mm)',
        'skin_darkening': 'Skin darkening (Y/N)',
        'hair_growth': 'hair growth(Y/N)',
        'weight_gain': 'Weight gain(Y/N)',
        'hair_loss': 'Hair loss(Y/N)',
        'fast_food': 'Fast food (Y/N)',
        'cycle_regularity': 'Cycle(R/I)',
        'cycle_length': 'Cycle length(days)',
        'Total_Follicle_Count': 'Total_Follicle_Count',
        'Symptom_Burden': 'Symptom_Burden'
    }

    # Create feature vector in the correct order
    feature_vector = []
    for feature_name in feature_names:
        # Find the API field that maps to this model feature
        api_field = None
        for api_name, model_name in field_mapping.items():
            if model_name == feature_name:
                api_field = api_name
                break

        if api_field and api_field in patient_data:
            feature_vector.append(patient_data[api_field])
        else:
            raise ValueError(f"Missing feature: {feature_name}")

    # Convert to numpy array and reshape for single prediction
    return np.array(feature_vector).reshape(1, -1)
