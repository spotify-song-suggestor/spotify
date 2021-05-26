import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
from tensorflow.keras.models import load_model
import numpy as np
from app import db, server, app
from models import spotify
import joblib

encodings = joblib.load(r'assets/encoded_data.joblib')
knn = joblib.load(r'assets/knn.joblib')
model = load_model(r'assets/ae4')

def recommend(index: int, n: int=5) -> 'tuple[np.ndarray]':
    '''
    ### Parameters
    index: index of song
    n: number of recommendations to pull

    returns: (dist, ind), array of distances, array of indeces for recommended songs. Includes
    original song.
    '''
    return knn.kneighbors([encodings[index]], n_neighbors=5)


def get_songs(indeces: 'list[int]') -> list:
    '''
    Uses SQLAlchemy queries to return track data from their indeces
    '''
    data = [spotify.query.filter(spotify.id == x).one() for x in indeces]
    return data

column1 = dbc.Col(
    [

        html.Br(),
        html.Br(),
        html.H6('Duration'),
        dcc.Slider(
            id='slider-1',
            min=90,
            max=100,
            value=45,
            step=.1,
            #             marks={i:str(i) for i in range(90,101)},

        ),
        html.H6('Explicit'),
        dcc.Slider(
            id='slider-2',
            min=0,
            max=1,
            step=1,
            value=0,
            #             marks={i:str(i) for i in range(1,11)},

        ),
        html.H6('Loudness'),
        dcc.Slider(
            id='slider-3',
            min=.01,
            max=1,
            step=.01,
            value=.5,
            #             marks={i:str(i) for i in range(1,11)},

        ),
        html.H6('Danceability'),
        dcc.Slider(
            id='slider-4',
            min=0.01,
            max=1,
            step=0.01,
            value=.5,
            #             marks={i:str(i) for i in range(0,2)},

        ),
        html.H6('Accousticness'),
        dcc.Slider(
            id='slider-5',
            min=0.01,
            max=38,
            step=0.01,
            value=19,
            #             marks={i:str(i) for i in range(0,39)},

        ),
        html.H6('Energy'),
        dcc.Slider(
            id='slider-6',
            min=0.01,
            max=1,
            step=0.01,
            value=.5,
            #             marks={i:str(i) for i in range(1,11)},

        ),
        html.H6('Instrumentalness'),
        dcc.Slider(
            id='slider-7',
            min=0.01,
            max=1,
            step=0.01,
            value=.5,
            #             marks={i:str(i) for i in range(0,1)},

        ),
        html.H6('Liveness'),
        dcc.Slider(
            id='slider-8',
            min=0.01,
            max=1,
            step=0.01,
            value=.5,
            #             marks={i:str(i) for i in range(1,9)},

        ),
        html.H6('Popularity'),
        dcc.Slider(
            id='slider-9',
            min=0.01,
            max=1,
            step=0.01,
            value=.5,
            #             marks={i:str(i) for i in range(1,9)},

        ),
        html.H6('Speechiness'),
        dcc.Slider(
            id='slider-10',
            min=0.1,
            max=1,
            step=0.01,
            value=.5,
            #             marks={i:str(i) for i in range(1,9)},

        ),
        html.H6('Tempo'),
        dcc.Slider(
            id='slider-11',
            min=60,
            max=240,
            step=5,
            value=92,
            marks={i: str(i) for i in range(60, 241, 20)},

        ),
        html.Br(),
        html.Br(),
        html.H6('Release_date'),
        dcc.Slider(
            id='slider-0',
            min=1960,
            max=2020,
            step=1,
            value=1990,
            marks={i: str(i) for i in range(1960, 2021, 10)},

        ),

        html.Br(),
        html.Br(),
        html.Br(),
    ],
    md=4,
)



column2 = dbc.Col(
    [html.Br(),
     html.Div(id='prediction-text1', children='output will go here', style={'color': 'green', 'fontSize': 16}),
     html.Br(),
     html.Div(id='prediction-text2', children='output will go here', style={'fontSize': 16}),
     html.Br(),
     html.Div(id='prediction-text3', children='output will go here', style={'color': 'green', 'fontSize': 16}),
     html.Div(id='prediction-text4', children='output will go here', style={'color': 'green', 'fontSize': 16}),
     html.Div(id='prediction-text5', children='output will go here', style={'color': 'green', 'fontSize': 16}),
     html.Br(),
     html.Div(id='prediction-text6', children='output will go here', style={'fontSize': 16}),
     html.Div(id='prediction-text7', children='output will go here', style={'color': 'green', 'fontSize': 16}),

     ],
    md=3,

)

column3 = dbc.Col(
    [
        # html.Div(id='prediction-text',children='output will go here'), 
        dcc.Markdown(
            """

            **Instructions**: Adjust the attribute sliders. Your prediction outcome will update dynamically. 
            Attribute Definitions:
            * **Duration** - Length of the song in ms
            * ** Explicit** - Has curse words or language or art that is sexual, violent, or offensive in nature. Left for none and right for has.
            * **Loudness** -How loud the songs are in dB ranging from -60 to 0
            * **Danceability** - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. 
            * **Acousticness** -Whether the tracks of artist are acoustic
            * **Energy** - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.
            * **Instrumentalness** - Predicts whether a track contains no vocals. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content.            
            * **Liveness** - how present audience are in artists songs
            * **US Popularity Level** - Highest point on Billboard  
            * **Speechiness** - Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value.
            * **Tempo** - The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. 
            * **Year of Release** - Year the track was released 
            """
        ),
        # html.Div(id='shapley',children='output will go here'),
        # dcc.Graph(id='my-graph-name', figure=plotly_figure)

    ]

)

layout = dbc.Row([column1, column2, column3])     
