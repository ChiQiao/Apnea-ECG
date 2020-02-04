import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objs as go
import streamlit as st

from scripts import plot

def load_model():
    with open('resources/model_logreg.pkl', 'rb') as f:
        res = pickle.load(f)
    return res


def load_sample_features(test_file):
    test_df = pd.read_csv('resources/feature_' + test_file + '.csv')
    test_df.drop(['apn', 'group', 'file'], axis=1, inplace=True)
    return test_df


def apnea_diagnose(y_pred):
    # Total minute
    apnea_total = sum(y_pred)

    # Hourly AI 
    total_hour = int(len(y_pred) / 60)
    y_pred_hourly = np.reshape(y_pred[ : total_hour * 60], (total_hour, 60))
    AI_hourly = y_pred_hourly.sum(axis=1)
    y_pred_left = y_pred[420 : ]
    if len(y_pred_left) >= 30:
        AI_hourly = np.append(AI_hourly, sum(y_pred_left) * 60 / len(y_pred_left))
        total_hour += 1
    AI_max = AI_hourly.max()
    return AI_max, apnea_total

mdl = load_model()
dict_data = {'Sample 1': 'c20', 'Sample 2': 'b05', 'Sample 3': 'a04'}
show_result = False

st.title('Sleep Apnea Evaluation')

st.header('Heart rate data')
a = st.empty()
b = st.empty()
options = ('Select one', 'Sample 1', 'Sample 2', 'Sample 3')
text_upload = 'Or upload your own heart rate data (Format: time of heart beat in minutes, single column csv file)'

option = a.selectbox('From a sample', options)
uploaded_file = b.file_uploader(text_upload, type='csv')

if option != 'Select one':
    features_df = load_sample_features(dict_data[option])
    # a.empty()
    # b.empty()
    show_result = True

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # a.empty()
    # b.empty()
    show_result = True

if show_result:
    # st.header('How is the diagnosis made?')
    # st.subheader('Usually this is what we need:')
    # st.markdown('* Electrocardiogram')
    # st.markdown('* Lung and brain activities')
    # st.markdown('* Breathing patterns')
    # st.markdown('* Blood oxygen levels')

    with open(f'resources/{dict_data[option]}.pkl', 'rb') as f:
        data = pickle.load(f)
    y_pred = mdl['mdl'].predict(mdl['scaler'].transform(features_df))
    AI_max, apnea_total = apnea_diagnose(y_pred)

    st.header('')
    st.header('Minute-wise evaluation')
    st.subheader('based on the heart rate data you uploaded')
    st.plotly_chart(plot.plot_hr(data['t'], data['hr'], y_pred))

    st.header('')
    st.header('Severity diagnosis')
    st.subheader('according to max[apnea minutes per hour] and total apnea minutes')
    st.plotly_chart(plot.plot_apnea_diagnosis(AI_max, apnea_total, y_pred))
    st.plotly_chart(plot.plot_diagnosis_result(AI_max, apnea_total))

    # st.subheader('1. Diagnose Apnea for each minute (red below)')
    # st.plotly_chart(plot.plot_hourly_apnea(y_pred))

    # st.subheader('2. Calculate Apnea Index (minutes of Apnea per hour)')
    # fig = plot_hourly_AI(y_pred, AI_hourly)
    # st.plotly_chart(fig)

    # st.subheader('3. Determine severity of Apnea')


