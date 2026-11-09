import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.clean_data import clean_raw_data, check_missing_summary

class TestDataCleaning(unittest.TestCase):

    def setUp(self):
        self.sample_raw = pd.DataFrame({'Student_ID': ['STU001', 'STU002', 'STU003'], 'Gender': [' male ', 'FEMALE', 'non-binary'], 'Social_Media_Hours': [8.0, 4.0, 9.5], 'Screen_Time_Hours': [7.0, 6.0, 8.0], 'Physical_Activity': ['low', 'MODERATE', 'high'], 'Mental_Health_Risk': ['High', 'Low', 'Medium']})

    def test_drops_identifier(self):
        cleaned = clean_raw_data(self.sample_raw, identifier='Student_ID')
        self.assertNotIn('Student_ID', cleaned.columns)
        self.assertEqual(len(cleaned), 3)

    def test_standardizes_casing(self):
        cleaned = clean_raw_data(self.sample_raw, identifier='Student_ID')
        self.assertEqual(cleaned['Gender'].tolist(), ['Male', 'Female', 'Non-binary'])
        self.assertEqual(cleaned['Physical_Activity'].tolist(), ['Low', 'Moderate', 'High'])

    def test_handles_impossible_values(self):
        cleaned = clean_raw_data(self.sample_raw, identifier='Student_ID')
        self.assertTrue(np.isnan(cleaned.loc[0, 'Social_Media_Hours']))
        self.assertEqual(cleaned.loc[1, 'Social_Media_Hours'], 4.0)
        self.assertTrue(np.isnan(cleaned.loc[2, 'Social_Media_Hours']))

    def test_missing_summary(self):
        cleaned = clean_raw_data(self.sample_raw, identifier='Student_ID')
        summary = check_missing_summary(cleaned)
        self.assertIn('Social_Media_Hours', summary.index)
        self.assertEqual(summary.loc['Social_Media_Hours', 'Missing_Count'], 2)
if __name__ == '__main__':
    unittest.main()
