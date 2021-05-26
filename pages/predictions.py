# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import joblib
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.python.training.tracking.util import list_objects
from app import db, server, app
from models import spotify
import matplotlib.pyplot as plt
import plotly.express as px

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
                hover_data=[g_artist,g_release_date,
                g_valence,g_danceability,g_energy],
                color=g_popularity,
                labels={'x':'Song Name','y': 'Popularity',
                        'hover_data_0': 'Artist',
                        'hover_data_1': 'Release Date',
                        'hover_data_2': 'Valence',
                        'hover_data_3': 'Danceability',
                        'hover_data_4': 'Energy'})

    return fig


# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout

column1 = dbc.Col(
    [
        dcc.Markdown(
        ),
        # toy tests with no meaning other than seeing what works
        F"{get_songs([0, 1,2,3])}"
        
    ],
    md=4
)

column2 = dbc.Col(
    [
        #Sanity Test
        dcc.Graph(figure=plot_graph(data=get_songs([0,6,1609,34455])))
       
    ]
)

layout = dbc.Row([column1, column2])
