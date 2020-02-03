import numpy as np
import plotly.graph_objects as go
# import util

fs = 100


def plot_raw_qrs_ann(ecg, r_peaks, atfs):
    # Plot original QRS annotation
    t = np.arange(len(ecg)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ecg, mode='lines', name='ECG'))
    fig.add_trace(go.Scatter(
        x=t[r_peaks], y=ecg[r_peaks], mode='markers', name='R'))
    fig.add_trace(go.Scatter(
        x=t[atfs], y=ecg[atfs], mode='markers', name='Artifacts'))
    fig.show()


# def plot_hr_apn(res, minute_start, minute_end):
#     # Plot heart rate with color indicating apn
#     fig = go.Figure()
#     for minute in range(minute_start, minute_end):
#         t_hr, hr = util.get_heart_rate(res['ecg'][minute].flatten())
#         color = 'red' if res['apn'][minute] else 'green'
#         t_hr = t_hr / 60 + minute # Convert to minute
#         fig.add_trace(go.Scatter(
#             x=t_hr, 
#             y=hr,
#             mode='lines',
#             line={'color':color, 'width':1}
#         ))

#     fig.update_layout(
#         xaxis_title="Minute",
#         yaxis_title="Heart Rate (bps)",
#         showlegend=False,
#     )
#     fig.show()
#     return fig


def plot_apnea_diagnosis(AI_max, apnea_total, y_pred):
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
        # title={
        #     'text':'OSA Diagnosis', 
        #     'x':0.5, 
        #     'xanchor': 'center',
        #     'y':0.85,
        #     'yanchor': 'top',
        #     },
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
                text="Normal",
                font={'size': 17}, 
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
            ),
            go.layout.Annotation(
                x=5,
                y=5,
                text="Moderate<br>Apnea",
                font={'size': 17}, 
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
            ),
            go.layout.Annotation(
                x=10,
                y=100,
                text="Severe<br>Apnea",
                font={'size': 17}, 
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
            ),
        ],
        height=300,
        margin=go.layout.Margin(
            b=0,
            t=10,
        ),
        font={'size': 15},
    )

    return fig