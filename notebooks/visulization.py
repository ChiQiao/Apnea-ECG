import numpy as np
import plotly.graph_objects as go


def plot_raw_qrs_ann(ecg, r_peaks, atfs, fs=100):
    # Plot original QRS annotation
    t = np.arange(len(ecg)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ecg, mode="lines", name="ECG"))
    fig.add_trace(go.Scatter(x=t[r_peaks], y=ecg[r_peaks], mode="markers", name="R"))
    fig.add_trace(go.Scatter(x=t[atfs], y=ecg[atfs], mode="markers", name="Artifacts"))
    fig.show()


def plot_diagnosis_result(AI_max, apnea_total):
    # Plot diagnosis result and recommendation
    if AI_max >= 10 and apnea_total >= 100:
        img_path = "https://raw.githubusercontent.com/ChiQiao/Apnea-ECG/master/resources/icon_warning.png"
        text = (
            'Severe Apnea<br ><span style="font-size:0.6em;">Snoring is jeopardizing your health'
            "<br >Sleep study strongly recommended</span>"
        )
    elif AI_max >= 5 and apnea_total >= 5:
        img_path = "https://raw.githubusercontent.com/ChiQiao/Apnea-ECG/master/resources/icon_attention.png"
        text = (
            'Moderate Apnea<br ><span style="font-size:0.6em;">Snoring is becoming a problem'
            "<br >Sleep study recommended</span>"
        )
    else:
        img_path = "https://raw.githubusercontent.com/ChiQiao/Apnea-ECG/master/resources/icon_good.png"
        text = "You are doing well!"

    fig = go.Figure()
    fig.add_layout_image(
        go.layout.Image(
            source=img_path,  # encoded_image,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=1.5,
            sizey=1.5,
            xanchor="right",
            yanchor="middle",
            # sizing="stretch",
            # opacity=0.5,
            layer="above",
        )
    )
    fig.update_layout(
        xaxis=dict(range=[-1.2, 3], visible=False,),
        yaxis=dict(range=[-1, 1], visible=False,),
        showlegend=False,
        annotations=[
            go.layout.Annotation(
                x=0.2,
                y=0,
                text=text,
                font={"size": 30},
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                align="left",
            ),
        ],
        height=200,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=go.layout.Margin(b=0, t=0,),
    )
    return fig


def plot_hourly_apnea(y_pred):
    # Plot grid (one row for one hour) indicating apnea (red) and non-apnea (green)
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
            if data[i] != data[i - 1]:
                e_idx.append(i - 1)
                s_idx.append(i)

        e_idx.append(len(data) - 1)

        # Plot blocks
        for s_idx, e_idx in zip(s_idx, e_idx):
            if data[s_idx] == 1:
                color = "red"
            elif data[s_idx] == 2:
                color = "grey"
            else:
                color = "lightgreen"
            plot_apnea_block(fig, hour, s_idx, e_idx + 1, color)

        # Plot separation lines
        fig.add_trace(
            go.Scatter(
                x=[0, 60],
                y=[hour + 0.5, hour + 0.5],
                mode="lines",
                line={"color": "white"},
            )
        )

    fig.update_layout(
        xaxis=dict(range=[0, 60], title="Minute", showgrid=False,),
        yaxis=dict(
            range=[total_hour + 0.5, 0.5],
            tick0=1,
            dtick=1,
            title="Hour",
            showgrid=False,
        ),
        height=300,
        showlegend=False,
        margin=go.layout.Margin(b=0, t=10,),
        font={"size": 15},
    )
    return fig


def plot_apnea_block(fig, hour, s_min, e_min, color):
    fig.add_shape(
        go.layout.Shape(
            type="rect",
            x0=s_min,
            y0=hour + 0.5,
            x1=e_min,
            y1=hour + 1.5,
            fillcolor=color,
            opacity=0.5,
            layer="below",
            line_width=0,
        )
    )


def plot_hr(t_hr, hr, y_pred):
    # Plot heart rate time history with red indicating apnea
    fig = go.Figure()

    # Find starting and ending index for each chunck of result (apnea or non-apnea)
    minute_start, minute_end = [], []
    minute_start.append(0)
    for i in range(1, len(y_pred)):
        if y_pred[i] != y_pred[i - 1]:
            minute_end.append(i)
            minute_start.append(i)

    minute_end.append(len(y_pred) - 1)

    # Plot segments
    neg_legend, pos_legend = True, False
    for minute_start, minute_end in zip(minute_start, minute_end):
        if y_pred[minute_start] == 1:
            color = "red"
            trace_legend = pos_legend
            seg_name = "Apnea"
            if pos_legend:
                pos_legend = False  # Shut down positive legend
        else:
            color = "green"
            trace_legend = neg_legend
            seg_name = "Non-apnea"
            if neg_legend:
                neg_legend, pos_legend = (
                    False,
                    True,
                )  # Shut down negtive legend, start positive legend
        idx = (t_hr > minute_start) & (t_hr < minute_end)
        fig.add_trace(
            go.Scatter(
                x=t_hr[idx] / 60,
                y=hr[idx],
                line={"color": color},
                name=seg_name,
                showlegend=trace_legend,
            )
        )
    fig.update_layout(
        xaxis=dict(title="Sleep hours", dtick=1, range=[0, t_hr[-1] / 60]),
        yaxis=dict(range=[0, 2], title="Heart rate (bps)",),
        font={"size": 15},
        height=200,
        margin=go.layout.Margin(b=0, t=30,),
    )
    return fig


def plot_hourly_AI(y_pred, AI_hourly, AI_max):
    # Plot time history of Apnea Index
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=np.arange(len(AI_hourly)) + 0.5, y=AI_hourly, name="Apnea Index",)
    )
    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Apnes Index",
        xaxis=dict(tickmode="linear", tick0=1, dtick=1, range=[0, len(AI_hourly)]),
        xaxis_showgrid=True,
        yaxis=dict(range=[0, AI_max * 1.2],),
        height=300,
        margin=go.layout.Margin(b=0, t=10,),
        # showlegend=True,
        # legend=dict(x=0, y=1.12, orientation='h'),
        font={"size": 15},
    )
    return fig


def plot_apnea_diagnosis(AI_max, apnea_total, y_pred):
    # Plot diagnosis of apnea severity
    fig = go.Figure()
    fig.add_shape(
        go.layout.Shape(
            type="rect",
            x0=0,
            y0=0,
            x1=60,
            y1=len(y_pred) * 1.2,
            fillcolor="rgb(187, 237, 195)",
            layer="below",
            name="Normal",
            line_width=0,
        )
    )
    fig.add_shape(
        go.layout.Shape(
            type="rect",
            x0=5,
            y0=5,
            x1=60,
            y1=len(y_pred) * 1.2,
            fillcolor="rgb(242, 245, 123)",
            layer="below",
            name="Moderate apnea",
            line_width=0,
        )
    )
    fig.add_shape(
        go.layout.Shape(
            type="rect",
            x0=10,
            y0=100,
            x1=60,
            y1=len(y_pred) * 1.2,
            fillcolor="rgb(242, 118, 123)",
            layer="below",
            name="Severe apnea",
            line_width=0,
        )
    )
    # Dummy bars for legend
    fig.add_trace(
        go.Bar(
            x=[-1],
            y=[0],
            name="Normal",
            marker=dict(opacity=0.5, color="rgb(187, 237, 195)",),
        )
    )
    fig.add_trace(
        go.Bar(
            x=[-1],
            y=[0],
            name="Moderate apnea",
            marker=dict(opacity=0.5, color="rgb(242, 245, 123)",),
        )
    )
    fig.add_trace(
        go.Bar(
            x=[-1],
            y=[0],
            name="Severe apnea",
            marker=dict(opacity=0.5, color="rgb(242, 118, 123)",),
        )
    )
    # Diagnosis result
    fig.add_trace(
        go.Scatter(
            x=[AI_max],
            y=[apnea_total],
            mode="markers",
            name="Your sleep",
            marker=dict(size=[20], color="blueviolet", opacity=0.9,),
        )
    )
    y_ub = np.min([np.max([120, apnea_total * 1.8]), len(y_pred)])
    x_ub = np.min([np.max([12, AI_max * 1.8]), 60])
    fig.update_layout(
        xaxis_title="Apnea Index",
        yaxis_title="Total apnea minutes",
        xaxis=dict(range=[0, x_ub], showgrid=True,),
        yaxis=dict(range=[0, y_ub],),
        showlegend=True,
        height=300,
        margin=go.layout.Margin(b=0, t=10,),
        font={"size": 15},
    )
    return fig
