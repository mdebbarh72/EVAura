import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.features.outliers import IQROutlierCapper
from src.features.preprocessing import engineer_candidate_features, encode_categorical_features, scale_features

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.df_train = pd.DataFrame({'Age': np.random.randint(18, 25, size=n).astype(float), 'Sleep_Hours': np.random.normal(7.0, 1.0, size=n), 'Study_Hours_Per_Week': np.random.normal(20.0, 5.0, size=n), 'GPA': np.random.uniform(2.0, 3.8, size=n), 'Social_Media_Hours': np.random.uniform(2.0, 6.0, size=n), 'Screen_Time_Hours': np.random.uniform(6.5, 12.0, size=n), 'Financial_Stress': np.random.uniform(2.0, 8.0, size=n), 'Family_Support': np.random.uniform(3.0, 9.0, size=n), 'Attendance_Rate': np.random.uniform(70.0, 100.0, size=n), 'Physical_Activity': np.random.choice(['Low', 'Moderate', 'High'], size=n), 'Diet_Quality': np.random.choice(['Poor', 'Average', 'Good'], size=n), 'Gender': np.random.choice(['Female', 'Male', 'Non-binary'], size=n), 'Part_Time_Job': np.random.choice(['Yes', 'No'], size=n), 'Previous_Mental_Health_History': np.random.choice(['Yes', 'No'], size=n), 'Year_of_Study': np.random.choice([1, 2, 3, 4], size=n).astype(float)})

    def test_outlier_capper(self):
        self.df_train.loc[0, 'Sleep_Hours'] = 25.0
        capper = IQROutlierCapper(columns=['Sleep_Hours'], factor=1.5)
        df_capped = capper.fit_transform(self.df_train)
        upper_bound = capper.upper_bounds_['Sleep_Hours']
        self.assertAlmostEqual(df_capped.loc[0, 'Sleep_Hours'], upper_bound, places=4)
        self.assertLess(df_capped.loc[0, 'Sleep_Hours'], 25.0)

    def test_engineer_candidate_features(self):
        engineered = engineer_candidate_features(self.df_train)
        expected_cols = ['Sleep_Deficit', 'Study_to_Sleep_Ratio', 'Social_Media_Share', 'Non_Social_Screen_Time', 'Stress_to_Support_Ratio', 'Academic_Effort_Index']
        for col in expected_cols:
            self.assertIn(col, engineered.columns)
        expected_non_social = np.maximum(0.0, self.df_train['Screen_Time_Hours'] - self.df_train['Social_Media_Hours'])
        np.testing.assert_allclose(engineered['Non_Social_Screen_Time'], expected_non_social)

    def test_scaling(self):
        x_tr_scaled, x_te_scaled, scaler = scale_features(self.df_train[['GPA', 'Sleep_Hours']], self.df_train[['GPA', 'Sleep_Hours']].iloc[:20])
        self.assertAlmostEqual(x_tr_scaled['GPA'].mean(), 0.0, places=5)
        self.assertAlmostEqual(x_tr_scaled['GPA'].std(ddof=0), 1.0, places=5)
if __name__ == '__main__':
    unittest.main()
