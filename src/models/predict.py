import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
from src.config import ROOT_DIR, MODELS_DIR, TARGET_NAMES, TARGET_MAPPING, ORDINAL_ORDERS
from src.utils.validation import validate_single_input, validate_batch_dataframe
from src.logger import get_logger
logger = get_logger('predict')
MODEL_FEATURES = ['Year_of_Study', 'Sleep_Hours', 'GPA', 'Social_Media_Hours', 'Screen_Time_Hours', 'Financial_Stress', 'Family_Support', 'Attendance_Rate', 'Non_Social_Screen_Time', 'Stress_to_Support_Ratio', 'Academic_Effort_Index', 'Physical_Activity', 'Part_Time_Job', 'Diet_Quality', 'Previous_Mental_Health_History', 'Gender_Male', 'Gender_Non-binary']
RISK_METADATA = {'Low': {'color': '#2ecc71', 'badge': 'Low Risk', 'description': 'Student demonstrates well-balanced lifestyle patterns, manageable stress levels, and stable academic engagement.', 'action': 'Routine wellness check-in and general student life balance resources.'}, 'Medium': {'color': '#f39c12', 'badge': 'Moderate Risk', 'description': 'Student exhibits elevated stress, suboptimal sleep, or academic pressure that could escalate if unaddressed.', 'action': 'Offer optional peer mentoring, time management workshops, and sleep hygiene advisory.'}, 'High': {'color': '#e74c3c', 'badge': 'High Risk', 'description': 'Critical convergence of acute stress, low family/social support, severe sleep deprivation, or prior history.', 'action': 'Urgent proactive counselor outreach, mental health support consultation, and academic accommodation review.'}}

def load_prediction_pipeline(pipeline_path: Optional[Path]=None):
    path = pipeline_path or MODELS_DIR / 'best_pipeline.joblib'
    if not Path(path).exists():
        raise FileNotFoundError(f'Model pipeline not found at {path}. Run training first.')
    pipe = joblib.load(path)
    if hasattr(pipe, 'named_steps') and 'scaler' in pipe.named_steps:
        if hasattr(pipe.named_steps['scaler'], 'set_output'):
            pipe.named_steps['scaler'].set_output(transform='pandas')
    return pipe

def load_prediction_model(model_path: Optional[Path]=None):
    path = model_path or MODELS_DIR / 'best_model.joblib'
    if not Path(path).exists():
        raise FileNotFoundError(f'Model not found at {path}. Run training first.')
    return joblib.load(path)

def preprocess_raw_input(df: pd.DataFrame) -> pd.DataFrame:
    df_in = df.copy()
    if all((col in df_in.columns for col in MODEL_FEATURES)):
        return df_in[MODEL_FEATURES].astype(float)
    if 'Social_Media_Hours' in df_in.columns and 'Screen_Time_Hours' in df_in.columns:
        invalid_mask = df_in['Social_Media_Hours'] > df_in['Screen_Time_Hours']
        if invalid_mask.any():
            df_in.loc[invalid_mask, 'Social_Media_Hours'] = df_in.loc[invalid_mask, 'Screen_Time_Hours']
    screen_time = df_in['Screen_Time_Hours'].astype(float)
    social_time = df_in['Social_Media_Hours'].astype(float)
    non_social = np.maximum(0.0, screen_time - social_time)
    fin_stress = df_in['Financial_Stress'].astype(float)
    fam_support = df_in['Family_Support'].astype(float)
    stress_to_support = fin_stress / (fam_support + 1e-05)
    gpa = df_in['GPA'].astype(float)
    study_hours = df_in['Study_Hours_Per_Week'].astype(float)
    academic_effort = gpa * (study_hours / 10.0)
    phys_map = {'low': 0.0, 'moderate': 1.0, 'high': 2.0}
    phys_encoded = df_in['Physical_Activity'].astype(str).str.strip().str.lower().map(phys_map).fillna(1.0)
    diet_map = {'poor': 0.0, 'average': 1.0, 'good': 2.0}
    diet_encoded = df_in['Diet_Quality'].astype(str).str.strip().str.lower().map(diet_map).fillna(1.0)
    part_time = (df_in['Part_Time_Job'].astype(str).str.strip().str.lower() == 'yes').astype(float)
    prev_mh = (df_in['Previous_Mental_Health_History'].astype(str).str.strip().str.lower() == 'yes').astype(float)
    gender_str = df_in['Gender'].astype(str).str.strip().str.lower()
    gender_male = (gender_str == 'male').astype(float)
    gender_nonbinary = ((gender_str == 'non-binary') | (gender_str == 'nonbinary')).astype(float)
    transformed = pd.DataFrame({'Year_of_Study': df_in['Year_of_Study'].astype(float), 'Sleep_Hours': df_in['Sleep_Hours'].astype(float), 'GPA': gpa, 'Social_Media_Hours': social_time, 'Screen_Time_Hours': screen_time, 'Financial_Stress': fin_stress, 'Family_Support': fam_support, 'Attendance_Rate': df_in['Attendance_Rate'].astype(float), 'Non_Social_Screen_Time': non_social, 'Stress_to_Support_Ratio': stress_to_support, 'Academic_Effort_Index': academic_effort, 'Physical_Activity': phys_encoded, 'Part_Time_Job': part_time, 'Diet_Quality': diet_encoded, 'Previous_Mental_Health_History': prev_mh, 'Gender_Male': gender_male, 'Gender_Non-binary': gender_nonbinary}, index=df_in.index)
    return transformed[MODEL_FEATURES]

def predict_single(sample: Dict[str, Any], pipeline=None, validate: bool=True) -> Dict[str, Any]:
    if validate:
        is_valid, errors = validate_single_input(sample)
        if not is_valid:
            return {'success': False, 'errors': errors, 'prediction': None}
    if pipeline is None:
        pipeline = load_prediction_pipeline()
    df_sample = pd.DataFrame([sample])
    features_df = preprocess_raw_input(df_sample)
    pred_idx = int(pipeline.predict(features_df)[0])
    risk_label = TARGET_NAMES[pred_idx]
    probabilities = None
    if hasattr(pipeline, 'predict_proba'):
        probas = pipeline.predict_proba(features_df)[0]
        probabilities = {TARGET_NAMES[i]: round(float(p), 4) for i, p in enumerate(probas)}
    meta = RISK_METADATA[risk_label]
    return {'success': True, 'prediction': risk_label, 'class_index': pred_idx, 'badge': meta['badge'], 'color': meta['color'], 'description': meta['description'], 'recommended_action': meta['action'], 'probabilities': probabilities, 'confidence': round(float(probabilities[risk_label]) * 100, 2) if probabilities else None}

def predict_batch(df: pd.DataFrame, pipeline=None, validate: bool=True) -> pd.DataFrame:
    if validate:
        is_valid, errors = validate_batch_dataframe(df)
        if not is_valid:
            raise ValueError(f"Batch validation failed: {'; '.join(errors)}")
    if pipeline is None:
        pipeline = load_prediction_pipeline()
    features_df = preprocess_raw_input(df)
    preds = pipeline.predict(features_df)
    df_out = df.copy()
    df_out['Predicted_Risk'] = [TARGET_NAMES[int(i)] for i in preds]
    if hasattr(pipeline, 'predict_proba'):
        probas = pipeline.predict_proba(features_df)
        for i, name in enumerate(TARGET_NAMES):
            df_out[f'Probability_{name}'] = probas[:, i].round(4)
    return df_out
if __name__ == '__main__':
    from src.data.load_data import load_data
    raw_df = load_data('raw')
    sample_row = raw_df.iloc[0].drop('Mental_Health_Risk').to_dict()
    print('Testing predict_single on raw sample row 0:')
    result = predict_single(sample_row)
    print(result)
