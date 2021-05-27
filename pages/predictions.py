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
import matplotlib.pyplot as plt
import plotly.express as px
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


def get_songs(indeces: 'list[int]') -> 'list[spotify]':
    '''
    Uses SQLAlchemy queries to return track data from their indeces
    '''
    data = [spotify.query.filter(spotify.id == x).one() for x in indeces]
    return data


def plot_graph(data:list):
    g_name = [(x.name) for x in data]
    g_popularity = [(x.popularity) for x in data]
    g_artist = [(x.artists) for x in data]
    g_release_date = [(x.release_date) for x in data]
    g_valence = [(x.valence) for x in data]
    g_danceability = [(x.danceability) for x in data]
    g_energy = [(x.energy) for x in data]
    g_speechiness = [(x.speechiness) for x in data]
    g_acousticness = [(x.acousticness) for x in data]
    g_instrumentalness = [(x.instrumentalness) for x in data]
    fig = px.bar(data, x= g_name, y= g_popularity,
                hover_data=[g_artist, g_release_date,
                            g_valence, g_danceability, g_energy],
                color=g_popularity,
                labels={'x':'Song Name','y': 'Popularity',
                        'hover_data_0': 'Artist',
                        'hover_data_1': 'Release Date',
                        'hover_data_2': 'Valence',
                        'hover_data_3': 'Danceability',
                        'hover_data_4': 'Energy'})

    return fig

# TODO: Test this function
def get_songs_via_features(features: list, n_songs: int=5) -> 'tuple[np.ndarray]':
    '''
    Converts input into the model's encoding, then runs it through the
    K-NearestNeighbors models
    Returns: 

    ### Parameters
    features: A list of all features required to run the model.
    The model encoder expects these inputs in this order:
    duration_ms,
    explicit,
    release_date,
    danceability,
    energy,
    key,
    loudness,
    mode,
    speechiness,
    acousticness,
    instrumentalness,
    liveness,
    valence,
    tempo,
    time_signature,
    popularity

    n_songs: number of songs to return.
    '''
    vec = model.encoder(np.array(features))
    return knn([vec], n_songs)




@app.callback(
    Input('duration', 'value'),
    Input('explicit', 'value'),
    Input('release_date', 'value'),
    Input('danceability', 'value'),
    Input('energy', 'value'),
    Input('key', 'value'),
    Input('loudness', 'value'),
    Input('mode', 'value'),
    Input('speechiness', 'value'),
    Input('acousticness', 'value'),
    Input('instrumentalness', 'value'),
    Input('liveness', 'value'),
    Input('valence', 'value'),
    Input('tempo', 'value'),
    Input('time_signature', 'value'),
    Input('popularity', 'value'),
    Output('prediction-text1', 'children')
)
def update_list(duration_ms,
                explicit,
                release_date,
                danceability,
                energy,
                key,
                loudness,
                mode,
                speechiness,
                acousticness,
                instrumentalness,
                liveness,
                valence,
                tempo,
                time_signature,
                popularity):
    # TODO: Write this function
    # Utilize get_songs_via_features to grab songs to display for user to pick
    # from. Route those songs via clicks to get recommendations for those songs.
    # This is just one method of doing this, discussion surrounding this would
    # be great
    pass

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout

# TODO: These values must match the input range for the model, meaning between
# 0 and 1. There also must be the complete set of features the model uses, as
# described above.
column1 = dbc.Col(
    [
        html.Br(),
        html.Br(),
        html.H6('Duration'),
        dcc.Slider(
            id='duration',
            min=90,
            max=100,
            value=45,
            step=.1,
            #             marks={i:str(i) for i in range(90,101)},
        ),
        html.H6('Explicit'),
        dcc.Slider(
            id='explicit',
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

        #Sanity Test
        dcc.Graph(figure=plot_graph(data=get_songs([0,6,1609,34455]))),
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
