import numpy as np
import pandas as pd
import pickle
import streamlit as st

from notebooks import visualization
from notebooks.feature_extractor import extract_features

def load_model():
    with open('resources/model_logreg.pkl', 'rb') as f:
        res = pickle.load(f)
    with open('data/feature/feature_selection.pkl', 'rb') as f:
        feature_col = pickle.load(f)
    return res['mdl'], res['scaler'], feature_col


def load_sample_features(test_file):
    test_df = pd.read_csv('data/feature/' + test_file + '.csv')
    test_df.drop(['apn', 'group', 'file'], axis=1, inplace=True)
    return test_df


def apnea_diagnose(y_pred):
    # Total minute
    apnea_total = sum(y_pred)

    # Hourly Apnea Index (AI)
    total_hour = int(len(y_pred) / 60)
    y_pred_hourly = np.reshape(y_pred[ : total_hour * 60], (total_hour, 60))
    AI_hourly = y_pred_hourly.sum(axis=1)
    y_pred_left = y_pred[420 : ]
    if len(y_pred_left) >= 30:
        AI_hourly = np.append(AI_hourly, sum(y_pred_left) * 60 / len(y_pred_left))
        total_hour += 1
    
    # Maximum Apnea Index
    AI_max = AI_hourly.max()

    return AI_max, apnea_total


def check_data(t_hr):
    # t_hr: numpy array in minutes

    # Check validity
    if np.any(np.diff(t_hr) < 0):
        st.warning(
            'Heart rate timestamp should be monotonically increasing.')
        return False

    # Check duration (should between 4 and 12 hours)
    duration = (t_hr[-1] - t_hr[0]) / 60
    if duration > 12 or duration < 4:
        st.warning(
            'A typical recording of heart rate should be around 8 hours. ' +\
            'Please make sure the heart rate data is in minutes.')
        return False
    return True


# Initialize
mdl, scaler, feature_col = load_model()
show_result = False

# Show introduction
st.title('Sleep Apnea Evaluation')
st.markdown(
    '''
    <font size="4">More than 18 million Americans suffer from [Sleep Apnea]
    (https://en.wikipedia.org/wiki/Sleep_apnea), which causes hypertension, heart disease, and 
    even depression and stroke in the long term. 

    Many people are not aware of this disease, thinking that they just snore a lot (Normal snore
    might progress into Sleep Apnea as well). If wearable devices
    can provide early warning, people with sleep apnea can seek for treatment before their health is
    compromised. This app is reaching towards this goal. It detect apnea with more than 80% accuracy just
    from heart rate measurements! 
    
    Curious about how it works? Check out [here]
    (https://docs.google.com/presentation/d/1WwZyvJ4VLjRcUPeKftsnVOTlXbZ1NYcIuLxvsKsN9ew/edit?usp=sharing)
     for more details.

    _(For self-evaluation only. Please confirm results with your health care provider.)_</font>
    ''',
    unsafe_allow_html=True)
st.header('Upload heart rate data')
a = st.empty() # Place holder to be either file uploader or selectbox
from_sample = st.checkbox('Just show me some samples')

# Load data
if from_sample:
    options = ('Sample 1', 'Sample 2', 'Sample 3')
    option = a.radio('', options, index=0)
    if option != 'Select one':
        dict_data = {'Sample 1': 'b09', 'Sample 2': 'c07', 'Sample 3': 'a12'}
        # Load features
        features_df = load_sample_features(dict_data[option])
        # Load heart rate data
        with open(f'data/raw/{dict_data[option]}.pkl', 'rb') as f:
            hr_data = pickle.load(f)
        show_result = True
else:
    uploaded_file = a.file_uploader(
        'format requirement: time of each heart beat in minutes, ' +\
        'starting from 0, single column csv file', type='csv')
    if uploaded_file is not None:
        t_hr = np.loadtxt(uploaded_file)
        show_result = check_data(t_hr)
        if show_result:
            with st.spinner('Extracting features...'):
                hr = 1 / (np.diff(t_hr * 60))
                t_hr = t_hr[1: ]
                hr_data = {'t': t_hr, 'hr': hr}
                features_df = extract_features(hr_data)

# Make prediction and plots
if show_result:
    with st.spinner('Generating diagnosis report...'):
        # Make prediction
        y_pred = mdl.predict_proba(scaler.transform(features_df[feature_col]))[:, 1] > 0.61
        AI_max, apnea_total = apnea_diagnose(y_pred)

        # Plot minute-wise prediction
        st.header('')
        st.header('Minute-wise evaluation')
        st.markdown('''
            <font size="4">Apnea is first evluated for each minute based on the heart rate.</font>
            ''', unsafe_allow_html=True)
        st.plotly_chart(visualization.plot_hr(hr_data['t'], hr_data['hr'], y_pred))

        # Plot severity diagnosis
        st.header('')
        st.header('Severity diagnosis')
        st.markdown('''
            <font size="4">The severity is determined by: <br />1) the highest Apnea Index (apnea minutes per hour), and 
            <br />2) total minutes of apnea during the sleep.</font>
            ''', unsafe_allow_html=True)
        st.plotly_chart(visualization.plot_apnea_diagnosis(AI_max, apnea_total, y_pred), config={'displayModeBar': False, 'staticPlot': True})
        st.plotly_chart(visualization.plot_diagnosis_result(AI_max, apnea_total), config={'displayModeBar': False, 'staticPlot': True})

