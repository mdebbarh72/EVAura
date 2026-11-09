import sys
import io
from pathlib import Path
from typing import Dict, Any

for candidate in [Path.cwd(), Path(__file__).resolve().parents[2], Path('/app')]:
    if (candidate / 'src').exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import streamlit as st
import pandas as pd
import numpy as np

from src.models.predict import predict_single, predict_batch, load_prediction_pipeline
from src.utils.validation import validate_single_input, validate_batch_dataframe
from app.components.charts import plot_probability_bars, plot_batch_risk_pie

st.set_page_config(page_title='EVAura - Risk Assessment', layout='wide')

st.title('Student Mental Health Risk Assessment')
st.markdown('Single student risk triage and batch cohort screening.')

@st.cache_resource
def get_cached_pipeline():
    return load_prediction_pipeline()

try:
    pipeline = get_cached_pipeline()
    pipeline_ready = True
except Exception as e:
    pipeline_ready = False
    st.error(f'Model pipeline could not be loaded: {e}. Run training pipeline first.')

tab_single, tab_batch = st.tabs(['Single Student Assessment', 'Batch Cohort Prediction (CSV)'])

with tab_single:
    st.subheader('Student Profile Input')

    with st.form('single_prediction_form'):
        col_acad, col_screen, col_psycho = st.columns(3)

        with col_acad:
            st.markdown('#### Academic Profile')
            year_of_study = st.selectbox('Year of Study', [1, 2, 3, 4, 5], index=1)
            gpa = st.slider('Current GPA (0.00 - 4.00)', min_value=1.0, max_value=4.0, value=2.9, step=0.01)
            study_hours = st.slider('Study Hours / Week', min_value=0.0, max_value=60.0, value=20.0, step=0.5)
            attendance = st.slider('Attendance Rate (%)', min_value=40.0, max_value=100.0, value=85.0, step=0.5)
            part_time = st.selectbox('Part-Time Job', ['No', 'Yes'], index=0)

        with col_screen:
            st.markdown('#### Sleep and Screen Time')
            sleep_hours = st.slider('Sleep Hours / Night', min_value=3.0, max_value=12.0, value=6.5, step=0.1)
            screen_time = st.slider('Screen Time / Day (Hours)', min_value=2.0, max_value=16.0, value=8.5, step=0.5)
            social_media = st.slider('Social Media / Day (Hours)', min_value=0.5, max_value=14.0, value=5.5, step=0.5)
            phys_activity = st.selectbox('Physical Activity Level', ['Low', 'Moderate', 'High'], index=1)
            diet_quality = st.selectbox('Diet Quality', ['Poor', 'Average', 'Good'], index=1)

        with col_psycho:
            st.markdown('#### Stress and Social Support')
            fin_stress = st.slider('Financial Stress (1 - 10)', min_value=1.0, max_value=10.0, value=5.5, step=0.5)
            fam_support = st.slider('Family Support (1 - 10)', min_value=1.0, max_value=10.0, value=6.0, step=0.5)
            gender = st.selectbox('Gender', ['Female', 'Male', 'Non-binary'], index=0)
            age = st.number_input('Age', min_value=16, max_value=60, value=21, step=1)
            prev_mh = st.selectbox('Previous Mental Health History', ['No', 'Yes'], index=0)

        submit_btn = st.form_submit_button('Evaluate Risk Level', use_container_width=True)

    if submit_btn and pipeline_ready:
        if social_media > screen_time:
            social_media = screen_time

        sample_input = {
            'Age': age, 'Gender': gender, 'Year_of_Study': year_of_study,
            'Sleep_Hours': sleep_hours, 'Study_Hours_Per_Week': study_hours,
            'GPA': gpa, 'Social_Media_Hours': social_media,
            'Screen_Time_Hours': screen_time, 'Physical_Activity': phys_activity,
            'Financial_Stress': fin_stress, 'Family_Support': fam_support,
            'Attendance_Rate': attendance, 'Part_Time_Job': part_time,
            'Diet_Quality': diet_quality, 'Previous_Mental_Health_History': prev_mh
        }

        with st.spinner('Calculating prediction...'):
            result = predict_single(sample_input, pipeline=pipeline)

        if not result.get('success', False):
            st.error(f"Inference error: {result.get('errors')}")
        else:
            st.markdown('---')
            st.subheader('Assessment Results')
            confidence = result['confidence']
            probabilities = result['probabilities']

            res_c1, res_c2 = st.columns([1.1, 1])

            with res_c1:
                bg_color = result['color']
                st.markdown(f"""
                <div style="background-color: {bg_color}15; border: 2px solid {bg_color}; border-radius: 6px; padding: 14px; margin-bottom: 10px;">
                    <div style="font-size: 1.4rem; font-weight: 700; color: {bg_color}; margin-bottom: 4px;">
                        {result['badge']} ({confidence}% Confidence)
                    </div>
                    <div style="font-size: 1rem; color: #2c3e50;">
                        {result['description']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('#### Action Plan')
                st.info(result['recommended_action'])

                if sleep_hours < 6.0:
                    st.warning(f'Sleep Warning: {sleep_hours:.1f}h per night. Substantial sleep deficit detected.')
                if fin_stress >= 7.5:
                    st.warning(f'Financial Strain: Score of {fin_stress:.1f}/10. High financial burden.')
                if fam_support <= 3.5:
                    st.warning(f'Support Alert: Low family support ({fam_support:.1f}/10).')

            with res_c2:
                st.markdown('#### Risk Probability Breakdown')
                fig_bars = plot_probability_bars(probabilities)
                st.pyplot(fig_bars, use_container_width=True)

                prob_df = pd.DataFrame([{'Risk Category': k, 'Probability': f'{v * 100:.2f}%'} for k, v in probabilities.items()])
                st.table(prob_df)

with tab_batch:
    st.subheader('Batch Student Cohort Screening')

    sample_path = Path('data/samples/sample_students.csv')
    col_demo1, col_demo2 = st.columns([1, 1])

    with col_demo1:
        if sample_path.exists():
            with open(sample_path, 'rb') as f:
                st.download_button(
                    label='Download Sample CSV Template',
                    data=f.read(),
                    file_name='sample_students_template.csv',
                    mime='text/csv',
                    use_container_width=True
                )

    with col_demo2:
        use_sample_btn = st.button('Load Sample 10-Student Cohort', use_container_width=True)

    uploaded_file = st.file_uploader('Upload Student Survey CSV', type=['csv'])

    df_to_process = None
    if use_sample_btn and sample_path.exists():
        df_to_process = pd.read_csv(sample_path)
        st.success('Loaded sample cohort with 10 records.')
    elif uploaded_file is not None:
        try:
            df_to_process = pd.read_csv(uploaded_file)
            st.success(f'Uploaded file loaded: {len(df_to_process)} rows.')
        except Exception as e:
            st.error(f'Error parsing CSV: {e}')

    if df_to_process is not None and pipeline_ready:
        st.markdown('---')
        st.markdown('#### Data Preview')
        st.dataframe(df_to_process.head(5), use_container_width=True)

        is_valid, validation_errors = validate_batch_dataframe(df_to_process)
        if not is_valid:
            st.error('Validation failed:')
            for err in validation_errors:
                st.write(f'- {err}')
        elif st.button('Run Batch Predictions', type='primary', use_container_width=True):
            with st.spinner(f'Analyzing {len(df_to_process)} records...'):
                pred_df = predict_batch(df_to_process, pipeline=pipeline)

            st.markdown('---')
            st.subheader('Cohort Screening Summary')

            risk_counts = pred_df['Predicted_Risk'].value_counts().reindex(['Low', 'Medium', 'High']).fillna(0).astype(int)
            total_students = len(pred_df)

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric('Cohort Size', f'{total_students} Students')
            with k2:
                st.metric('Low Risk', f"{risk_counts['Low']} ({risk_counts['Low'] / total_students * 100:.1f}%)")
            with k3:
                st.metric('Medium Risk', f"{risk_counts['Medium']} ({risk_counts['Medium'] / total_students * 100:.1f}%)")
            with k4:
                st.metric('High Risk', f"{risk_counts['High']} ({risk_counts['High'] / total_students * 100:.1f}%)")

            c_chart, c_table = st.columns([1, 1.5])
            with c_chart:
                fig_pie = plot_batch_risk_pie(risk_counts)
                st.pyplot(fig_pie, use_container_width=True)

            with c_table:
                st.markdown('#### High Risk Students')
                high_risk_students = pred_df[pred_df['Predicted_Risk'] == 'High']
                if not high_risk_students.empty:
                    disp_cols = [c for c in ['Student_ID', 'Sleep_Hours', 'Financial_Stress', 'Family_Support', 'Probability_High'] if c in high_risk_students.columns]
                    st.dataframe(high_risk_students[disp_cols].sort_values('Probability_High', ascending=False), use_container_width=True)
                else:
                    st.info('No high-risk students detected in cohort.')

            st.markdown('---')
            st.markdown('#### Annotated Results')
            st.dataframe(pred_df, use_container_width=True)

            csv_buffer = io.StringIO()
            pred_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label='Download Predictions CSV',
                data=csv_buffer.getvalue(),
                file_name='student_mental_health_predictions.csv',
                mime='text/csv',
                use_container_width=True
            )
