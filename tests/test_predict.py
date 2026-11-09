import sys
import unittest
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.predict import predict_single, predict_batch, load_prediction_pipeline
from src.utils.validation import validate_single_input

class TestPredictPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_valid = {'Age': 21, 'Gender': 'Male', 'Year_of_Study': 3, 'Sleep_Hours': 7.0, 'Study_Hours_Per_Week': 20.0, 'GPA': 3.2, 'Social_Media_Hours': 4.5, 'Screen_Time_Hours': 7.0, 'Physical_Activity': 'Moderate', 'Financial_Stress': 5.0, 'Family_Support': 7.0, 'Attendance_Rate': 88.0, 'Part_Time_Job': 'No', 'Diet_Quality': 'Good', 'Previous_Mental_Health_History': 'No'}

    def test_single_prediction_valid(self):
        result = predict_single(self.sample_valid)
        self.assertTrue(result['success'])
        self.assertIn(result['prediction'], ['Low', 'Medium', 'High'])
        self.assertIn('probabilities', result)
        self.assertIsNotNone(result['probabilities'])
        prob_sum = sum(result['probabilities'].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_single_prediction_invalid_ranges(self):
        sample_invalid = self.sample_valid.copy()
        sample_invalid['Sleep_Hours'] = 35.0
        is_valid, errors = validate_single_input(sample_invalid)
        self.assertFalse(is_valid)
        self.assertTrue(any(('Sleep_Hours' in err for err in errors)))

    def test_batch_prediction(self):
        df = pd.DataFrame([self.sample_valid, self.sample_valid])
        df['Student_ID'] = ['STU_TEST1', 'STU_TEST2']
        preds = predict_batch(df)
        self.assertEqual(len(preds), 2)
        self.assertIn('Predicted_Risk', preds.columns)
        self.assertIn('Probability_Low', preds.columns)
        self.assertIn('Probability_Medium', preds.columns)
        self.assertIn('Probability_High', preds.columns)
if __name__ == '__main__':
    unittest.main()
