import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objs as go
import pickle

def load_model():
    with open('resources/logreg_time.pkl', 'rb') as f:
        res = pickle.load(f)
    return res

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

    return AI_max, apnea_total, AI_hourly


def plot_diagnosis_result(AI_max, apnea_total):
    if AI_max >= 10 and apnea_total >= 100:
        img_path = 'warning.png'
        text = 'Severe Apnea'
    elif AI_max >= 5 and apnea_total >= 5:
        img_path = 'attention.png'
        text = 'Moderate Apnea'
    else:
        img_path = 'good.png'
        text = 'You are doing well!'
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.text(2000, 750, text, 
        verticalalignment='center', fontsize=30)
    plt.xlim(0, 5000)
    plt.tight_layout()
    plt.axis('off')
    st.pyplot()


def plot_hourly_apnea(y_pred):
    total_hour = np.ceil(len(y_pred) / 60)
    y_segs = np.hstack((y_pred, 2 * np.ones(int(total_hour * 60 - len(y_pred)))))
    y_segs = y_segs.reshape(int(total_hour), 60)

    fig = go.Figure()
    for hour in range(len(y_segs)):
        data = y_segs[hour, :]

        # Find starting and ending index for each chunck of result (apnea or non-apnea)
        s_idx, e_idx = [], []
        s_idx.append(0)
        for i in range(1, len(data)):
            if data[i] != data[i-1]:
                e_idx.append(i-1)
                s_idx.append(i)

        e_idx.append(len(data) - 1)

        # Plot blocks
        for s_idx, e_idx in zip(s_idx, e_idx):
            color = 'red' if data[s_idx] else 'lightgreen'
            plot_apnea_block(fig, hour, s_idx, e_idx+1, color)

    fig.update_layout(
        xaxis=dict(
            range=[0, 60],
            title='Minute'
        ),
        yaxis=dict(
            range=[0, total_hour],
            tick0=1,
            dtick=1,
            title='Hour'
        ),
        title={
            'text':'Apnea based on minutes', 
            'x':0.5, 
            'xanchor': 'center',
            'y':0.85,
            'yanchor': 'top',
            },
        font={'size': 15},
    )
    st.plotly_chart(fig)


def plot_apnea_block(fig, hour, s_min, e_min, color):
    fig.add_shape(go.layout.Shape(
        type="rect",
        x0=s_min,
        y0=hour,
        x1=e_min,
        y1=hour+1,
        fillcolor=color,
        opacity=0.5,
        layer="below",
        line_width=0,
    ))


def plot_hourly_AI(y_pred, AI_hourly):
    fig = go.Figure()
    # t_minute = np.arange(len(y_pred)) / 60
    fig.add_trace(go.Scatter(
        x=np.arange(len(AI_hourly)) + 0.5, 
        y=AI_hourly,
        name='Apnea Index',
    ))
    # fig.add_trace(go.Bar(
    #     x=t_minute[y_pred==1], 
    #     y=np.ones(t_minute[y_pred==1].shape) * AI_max / 10,
    #     width=1/60,
    #     name='Apnea'
    # ))
    # fig.add_trace(go.Bar(
    #     x=t_minute[y_pred==0], 
    #     y=np.ones(t_minute[y_pred==0].shape) * AI_max / 10,
    #     width=1/60,
    #     name='Non-apnea',
    # ))
    fig.update_layout(
        title={
            'text':'Apnea Index', 
            'x':0.5, 
            'xanchor': 'center',
            'y':0.9,
            'yanchor': 'top',
            },
        xaxis_title='Hour',
        yaxis_title='Apnes Index',
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 1,
            dtick = 1,
            range=[0, len(AI_hourly)]
        ),
        xaxis_showgrid=True,
        yaxis = dict(
            range=[0, AI_max * 1.2],
        ),
        # showlegend=True,
        # legend=dict(x=0, y=1.12, orientation='h'),
        font={'size': 15},
    )
    st.plotly_chart(fig)


def plot_apnea_diagnosis(AI_max, apnea_total):
    fig = go.Figure()
    fig.add_shape(go.layout.Shape(
        type="rect",
        x0=0,
        y0=0,
        x1=5,
        y1=len(y_pred) * 1.2,
        fillcolor='lightgreen',
        opacity=0.5,
        layer="below",
        line_width=0,
    ))
    fig.add_shape(go.layout.Shape(
        type="rect",
        x0=5,
        y0=0,
        x1=60,
        y1=5,
        fillcolor='lightgreen',
        opacity=0.5,
        layer="below",
        line_width=0,
        name='Safe',
    ))
    fig.add_shape(go.layout.Shape(
        type="rect",
        x0=5,
        y0=5,
        x1=10,
        y1=len(y_pred) * 1.2,
        fillcolor='yellow',
        opacity=0.5,
        layer="below",
        line_width=0,
    ))
    fig.add_shape(go.layout.Shape(
        type="rect",
        x0=10,
        y0=5,
        x1=60,
        y1=100,
        fillcolor='yellow',
        opacity=0.5,
        layer="below",
        line_width=0,
    ))
    fig.add_shape(go.layout.Shape(
        type="rect",
        x0=10,
        y0=100,
        x1=60,
        y1=len(y_pred) * 1.2,
        fillcolor='red',
        opacity=0.5,
        layer="below",
        line_width=0,
    ))
    fig.add_trace(go.Scatter(
        x=[AI_max],
        y=[apnea_total],
        mode='markers',
        marker=dict(
            size=[20], 
            color='blueviolet',
            opacity=0.9,
            ),
    ))
    y_ub = np.min([np.max([120, apnea_total * 1.8]), len(y_pred)])
    x_ub = np.min([np.max([12, AI_max * 1.8]), 60])
    fig.update_layout(
        title={
            'text':'OSA Diagnosis', 
            'x':0.5, 
            'xanchor': 'center',
            'y':0.85,
            'yanchor': 'top',
            },
        xaxis_title='Max. Apnea Index',
        yaxis_title='Total Apnea Minutes',
        xaxis = dict(
            range=[0, x_ub]
        ),
        yaxis = dict(
            range=[0, y_ub],
        ),
        showlegend=False,
        annotations=[
            go.layout.Annotation(
                x=0,
                y=0,
                text="Safe",
                font={'size': 17}, 
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
            ),
            go.layout.Annotation(
                x=5,
                y=5,
                text="Attention<br>Needed",
                font={'size': 17}, 
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
            ),
            go.layout.Annotation(
                x=10,
                y=100,
                text="Treatment<br>Needed",
                font={'size': 17}, 
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
            ),
        ],
        font={'size': 15},
    )
    st.plotly_chart(fig)

st.title('Apnea prediction')

option = st.selectbox(
    'Sample ECG recording', ('Select one', 'Normal', 'Moderate Apnea', 'Severe Apnea'))
dict_data = {'Normal': 'X_C', 'Moderate Apnea': 'X_B', 'Severe Apnea': 'X_A'}

if option != 'Select one':
    res = load_model()
    y_pred = res['mdl'].predict(res[dict_data[option]])
    AI_max, apnea_total, AI_hourly = apnea_diagnose(y_pred)

    st.header('Diagnosis result')
    plot_diagnosis_result(AI_max, apnea_total)

    st.header('How is the diagnosis made?')
    st.subheader('1. Apnea is first diagnosed for each minute')
    plot_hourly_apnea(y_pred)

    st.subheader('2. Apnea Index is calculated as minutes of Apnea per hour')
    plot_hourly_AI(y_pred, AI_hourly)

    st.subheader('3. Final diagnosis is based on max Apnea Index and total minutes of Apnea')
    plot_apnea_diagnosis(AI_max, apnea_total)


