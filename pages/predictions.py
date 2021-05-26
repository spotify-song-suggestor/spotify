# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import joblib
from tensorflow.keras.models import load_model
import numpy as np
from app import db, server, app
from models import spotify
import matplotlib.pyplot as plt

encodings = joblib.load(r'assets\encoded_data.joblib')
knn = joblib.load(r'assets\knn.joblib')
model = load_model(r'assets\ae4')



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


# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
        ),
        # toy tests with no meaning other than seeing what works
        F"{get_songs([0, 0])}"
    ],
    md=4
)

column2 = dbc.Col(
    [

    ]
)

layout = dbc.Row([column1, column2])
