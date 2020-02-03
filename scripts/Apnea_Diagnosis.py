import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objs as go
import streamlit as st

import plot

def load_model(test_file):
    with open('../resources/model_logreg.pkl', 'rb') as f:
        res = pickle.load(f)
    test_df = pd.read_csv('../resources/feature_' + test_file + '.csv')
    test_df.drop(['apn', 'group', 'file'], axis=1, inplace=True)
    return res, test_df

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
        img_path = 'https://raw.githubusercontent.com/ChiQiao/Apnea-ECG/master/resources/warning.png'
        text = 'Severe Apnea<br ><span style="font-size:0.6em;">Snoring is jeopardizing your health'\
            '<br >Strongly recommend to see a doctor</span>'
    elif AI_max >= 5 and apnea_total >= 5:
        img_path = 'https://raw.githubusercontent.com/ChiQiao/Apnea-ECG/master/resources/attention.png'
        text = 'Moderate Apnea<br ><span style="font-size:0.6em;">Snoring is becoming a problem'\
            '<br >Recommend to see a doctor</span>'
    else:
        img_path = 'https://raw.githubusercontent.com/ChiQiao/Apnea-ECG/master/resources/good.png'
        text = 'You are doing well!'

    # with open(img_path, "rb") as image_file:
    #     encoded_string = base64.b64encode(image_file.read()).decode()
    # # Add the prefix that plotly will want when using the string as source
    # encoded_image = "data:image/png;base64," + encoded_string

    fig = go.Figure()
    fig.add_layout_image(
        go.layout.Image(
            source=img_path, #encoded_image,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=1.5,
            sizey=1.5,
            xanchor='right',
            yanchor='middle',
            # sizing="stretch",
            # opacity=0.5,
            layer="above",
        )
    )
    fig.update_layout(
        xaxis = dict(
            range=[-2, 3],
            visible=False,
        ),
        yaxis = dict(
            range=[-1, 1],
            visible=False,
        ),
        showlegend=False,
        annotations=[
            go.layout.Annotation(
                x=0.2,
                y=0,
                text=text,
                font={'size': 30}, 
                showarrow=False,
                xanchor='left',
                yanchor='middle',
                align="left",
            ),
        ],
        height=200,
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=go.layout.Margin(
            b=0,
            t=0,
        ),
    )
    st.plotly_chart(fig)

    # img = mpimg.imread(img_path)
    # fig = plt.figure()
    # plt.imshow(img)
    # plt.text(2000, 750, text, 
    #     verticalalignment='center', fontsize=30)
    # plt.xlim(0, 5000)
    # plt.tight_layout()
    # plt.axis('off')
    # st.pyplot(fig)

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
            if data[s_idx] == 1:
                color = 'red'
            elif data[s_idx] == 2:
                color = 'grey'
            else:
                color = 'lightgreen'
            plot_apnea_block(fig, hour, s_idx, e_idx+1, color)

        # Plot separation lines
        fig.add_trace(go.Scatter(
            x=[0, 60],
            y=[hour+0.5, hour+0.5],
            mode='lines',
            line={'color':'white'},
        ))

    fig.update_layout(
        xaxis=dict(
            range=[0, 60],
            title='Minute',
            showgrid=False,
        ),
        yaxis=dict(
            range=[total_hour+0.5, 0.5],
            tick0=1,
            dtick=1,
            title='Hour',
            showgrid=False,
        ),
        # title={
        #     'text':'Apnea based on minutes', 
        #     'x':0.5, 
        #     'xanchor': 'center',
        #     'y':0.85,
        #     'yanchor': 'top',
        #     },
        height=300,
        showlegend=False,
        margin=go.layout.Margin(
            b=0,
            t=10,
        ),
        font={'size': 15},
    )
    st.plotly_chart(fig)

def plot_apnea_block(fig, hour, s_min, e_min, color):
    fig.add_shape(go.layout.Shape(
        type="rect",
        x0=s_min,
        y0=hour+0.5,
        x1=e_min,
        y1=hour+1.5,
        fillcolor=color,
        opacity=0.5,
        layer="below",
        line_width=0,
    ))

def plot_hr(t_hr, hr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_hr / 60, 
        y=hr,
    ))
    fig.update_layout(
        # title={
        #     'text':'First minute of heart rate', 
        #     'x':0.5, 
        #     'xanchor': 'center',
        #     'y':0.9,
        #     'yanchor': 'top',
        #     },
        xaxis=dict(
            # range=[0, 60],
            title='Hour',
        ),
        yaxis=dict(
            range=[0, 2],
            title='Heart rate (bps)',
        ),
        font={'size': 15},
        height=200,
        margin=go.layout.Margin(
            b=0,
            t=10,
        ),
        )
    st.plotly_chart(fig)    

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
        # title={
        #     'text':'Apnea Index', 
        #     'x':0.5, 
        #     'xanchor': 'center',
        #     'y':0.9,
        #     'yanchor': 'top',
        #     },
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
        height=300,
        margin=go.layout.Margin(
            b=0,
            t=10,
        ),
        # showlegend=True,
        # legend=dict(x=0, y=1.12, orientation='h'),
        font={'size': 15},
    )
    st.plotly_chart(fig)


    


st.title('Sleep Apnea Evaluation')

option = st.selectbox(
    'Sample ECG recording', ('Select one', 'Normal', 'Moderate Apnea', 'Severe Apnea'))
dict_data = {'Normal': 'c06', 'Moderate Apnea': 'x03', 'Severe Apnea': 'x05'}

if option != 'Select one':
    mdl, data = load_model(dict_data[option])
    y_pred = mdl['mdl'].predict(mdl['scaler'].transform(data))
    AI_max, apnea_total, AI_hourly = apnea_diagnose(y_pred)

    st.header('Diagnosis result')
    plot_diagnosis_result(AI_max, apnea_total)

    st.header('How is the diagnosis made?')
    
    st.subheader('Usually this is what we need:')
    st.markdown('* Electrocardiogram')
    st.markdown('* Lung and brain activities')
    st.markdown('* Breathing patterns')
    st.markdown('* Blood oxygen levels')

    st.subheader('Predictions here, however, are based on the heart rate data you uploaded.')
    with open('../resources/HR_' + dict_data[option] + '.pkl', 'rb') as f:
        data = pickle.load(f)
    # st.dataframe(pd.DataFrame(data['t'] * 60, columns=['Heart beat time (s)']))
    plot_hr(data['t'], data['hr'])

    st.subheader('1. Diagnose Apnea for each minute (red below)')
    plot_hourly_apnea(y_pred)

    # st.subheader('2. Calculate Apnea Index (minutes of Apnea per hour)')
    # plot_hourly_AI(y_pred, AI_hourly)

    st.subheader('3. Determine severity of Apnea')
    fig = plot.plot_apnea_diagnosis(AI_max, apnea_total, y_pred)
    st.plotly_chart(fig)


