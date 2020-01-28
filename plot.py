import numpy as np
import plotly.graph_objects as go
import util

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


def plot_hr_apn(res, minute_start, minute_end):
    # Plot heart rate with color indicating apn
    fig = go.Figure()
    for minute in range(minute_start, minute_end):
        t_hr, hr = util.get_heart_rate(res['ecg'][minute].flatten())
        color = 'red' if res['apn'][minute] else 'green'
        t_hr = t_hr / 60 + minute # Convert to minute
        fig.add_trace(go.Scatter(
            x=t_hr, 
            y=hr,
            mode='lines',
            line={'color':color, 'width':1}
        ))

    fig.update_layout(
        xaxis_title="Minute",
        yaxis_title="Heart Rate (bps)",
        showlegend=False,
    )
    fig.show()
    return fig