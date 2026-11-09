from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
REQUIRED_RAW_COLUMNS = ['Age', 'Gender', 'Year_of_Study', 'Sleep_Hours', 'Study_Hours_Per_Week', 'GPA', 'Social_Media_Hours', 'Screen_Time_Hours', 'Physical_Activity', 'Financial_Stress', 'Family_Support', 'Attendance_Rate', 'Part_Time_Job', 'Diet_Quality', 'Previous_Mental_Health_History']
VALID_GENDERS = ['Male', 'Female', 'Non-binary']
VALID_PHYSICAL_ACTIVITY = ['Low', 'Moderate', 'High']
VALID_DIET_QUALITY = ['Poor', 'Average', 'Good']
VALID_BINARY = ['Yes', 'No']
RANGE_CONSTRAINTS = {'Age': (15, 100), 'Year_of_Study': (1, 6), 'Sleep_Hours': (0.0, 24.0), 'Study_Hours_Per_Week': (0.0, 100.0), 'GPA': (0.0, 4.0), 'Social_Media_Hours': (0.0, 24.0), 'Screen_Time_Hours': (0.0, 24.0), 'Financial_Stress': (1.0, 10.0), 'Family_Support': (1.0, 10.0), 'Attendance_Rate': (0.0, 100.0)}

def validate_single_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    for col in REQUIRED_RAW_COLUMNS:
        if col not in data or data[col] is None or data[col] == '':
            errors.append(f"Missing required field: '{col}'")
    if errors:
        return (False, errors)
    for col, (min_val, max_val) in RANGE_CONSTRAINTS.items():
        try:
            val = float(data[col])
            if val < min_val or val > max_val:
                errors.append(f"'{col}' value {val} is outside valid range [{min_val}, {max_val}].")
        except (ValueError, TypeError):
            errors.append(f"'{col}' must be a numeric value, got '{data[col]}'.")
    try:
        social_h = float(data['Social_Media_Hours'])
        screen_h = float(data['Screen_Time_Hours'])
        if social_h > screen_h:
            errors.append(f'Social Media Hours ({social_h}h) cannot exceed total Screen Time Hours ({screen_h}h).')
    except (ValueError, TypeError):
        pass
    gender = str(data.get('Gender', '')).strip().capitalize()
    if gender not in VALID_GENDERS and str(data.get('Gender', '')).strip() != 'Non-binary':
        errors.append(f"Invalid Gender: '{data.get('Gender')}'. Allowed: {VALID_GENDERS}")
    phys = str(data.get('Physical_Activity', '')).strip().capitalize()
    if phys not in VALID_PHYSICAL_ACTIVITY:
        errors.append(f"Invalid Physical_Activity: '{data.get('Physical_Activity')}'. Allowed: {VALID_PHYSICAL_ACTIVITY}")
    diet = str(data.get('Diet_Quality', '')).strip().capitalize()
    if diet not in VALID_DIET_QUALITY and diet not in ['Nan', 'None', '']:
        errors.append(f"Invalid Diet_Quality: '{data.get('Diet_Quality')}'. Allowed: {VALID_DIET_QUALITY}")
    for bin_col in ['Part_Time_Job', 'Previous_Mental_Health_History']:
        val = str(data.get(bin_col, '')).strip().capitalize()
        if val not in VALID_BINARY:
            errors.append(f"Invalid {bin_col}: '{data.get(bin_col)}'. Allowed: {VALID_BINARY}")
    return (len(errors) == 0, errors)

def validate_batch_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if df.empty:
        return (False, ['Uploaded dataset is empty.'])
    missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing_cols:
        errors.append(f'Uploaded CSV is missing required columns: {missing_cols}')
        return (False, errors)
    missing_counts = df[REQUIRED_RAW_COLUMNS].isnull().sum()
    cols_with_many_nulls = missing_counts[missing_counts > len(df) * 0.5].index.tolist()
    if cols_with_many_nulls:
        errors.append(f'Columns have more than 50% missing values: {cols_with_many_nulls}')
    return (len(errors) == 0, errors)
