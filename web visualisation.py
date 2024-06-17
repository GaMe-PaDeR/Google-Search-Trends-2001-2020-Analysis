import pandas as pd
import seaborn as sns
# Using plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
# pyo.init_notebook_mode()

df = pd.read_csv("Google-Search-Trends-2001-2020-Analysis/trends.csv")
# df =df[df['year']>2007]

df = df[df.location == 'Russia']
category = pd.DataFrame((df.category.value_counts() > 30)).head(8).index

fig = go.Figure()

for i in range(2008, 2021):
    fig.add_trace(
        go.Bar(
            visible=False,
            x=['Rank 1','Rank 2','Rank 3','Rank 4','Rank 5'], y=[7,4.5,2.5,1,0.5],
            text=list(df[(df['year'] == i) & (df['category']=='Люди')]['query']))
            )

fig.data[0].visible = True

steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Google Trends: " + str(i+2008)},
              ],  # layout attribute
        label = i + 2008
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Year: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_traces(textposition='outside',hoverinfo='text')
fig.update_traces(marker_color='white')
fig.update_layout(plot_bgcolor='white')
fig.update_traces(textfont_size=40)
fig.update_yaxes(showticklabels=False, range=[0,8])
fig.update_xaxes(showticklabels=False)

fig.update_layout(
    sliders=sliders,
)

button_dict =[]
for i in range(len(category)):
    button = dict(
        args=["text",[
                     list(df[(df['year'] == 2008) & (df['category']==category[i])]['query']),
                     list(df[(df['year'] == 2009) & (df['category']==category[i])]['query']),
                     list(df[(df['year'] == 2010) & (df['category']==category[i])]['query']),
                     list(df[(df['year'] == 2011) & (df['category']==category[i])]['query']),
                     list(df[(df['year'] == 2012) & (df['category']==category[i])]['query']),
                     list(df[(df['year'] == 2013) & (df['category']==category[i])]['query']),
                     list(df[(df['year'] == 2014) & (df['category']==category[i])]['query']),
                     list(df[(df['year'] == 2015) & (df['category']==category[i])]['query']),
                     list(df[(df['year'] == 2016) & (df['category']==category[i])]['query']),
                     list(df[(df['year'] == 2017) & (df['category']==category[i])]['query']),
                     list(df[(df['year'] == 2018) & (df['category']==category[i])]['query']),
                     list(df[(df['year'] == 2019) & (df['category']==category[i])]['query']),
                     list(df[(df['year'] == 2020) & (df['category']==category[i])]['query']),]],
        label=category[i],
        method="restyle"
    )
    button_dict.append(button)

fig.update_layout(
    updatemenus=[
        dict(
            buttons= button_dict,
            direction="down",
            pad={"l": 10, "t": 10},
            showactive=True,
            x=0.25,
            xanchor="left",
            y=1.15,
            yanchor="top"
            ),
    ],
)
fig.update_layout(annotations=[
                    dict(text='<It do not show if have no data>', x=3, y=6, font_size=20, showarrow=False),
                    dict(text='Rank 1', x=0, y=4, font_size=25, showarrow=False,bgcolor="#ff7f0e"),
                    dict(text='Rank 2', x=1, y=2.5, font_size=20, showarrow=False,bgcolor="#ff7f0e"),
                    dict(text='Rank 3', x=2, y=1.3, font_size=17, showarrow=False,bgcolor="#ff7f0e"),
                    dict(text='Rank 4', x=3, y=0.5, font_size=13, showarrow=False,bgcolor="#ff7f0e"),
                    dict(text='Rank 5', x=4, y=0.2, font_size=8, showarrow=False,bgcolor="#ff7f0e"),

])

fig.update_traces(opacity = .9)
fig.update_layout(title="Google Trends: 2008",height=800)

fig.show()