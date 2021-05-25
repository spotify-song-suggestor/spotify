# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import joblib
from tensorflow.keras.models import load_model
import numpy as np
# Imports from this application
from app import app


encodings = joblib.load(r'app\dash-template\assets\encoded_data.joblib')
knn = joblib.load(r'app\dash-template\assets\knn.joblib')
model = load_model(r'app\dash-template\assets\ae4')


def recommend(index: int, n: int=5):
    '''
    index: index of song
    n: number of recommendations to pull
    returns: array of distances, array of indexes for recommended songs. Includes
        original song.
    '''
    return knn.kneighbors([encodings[index]], n_neighbors=5)


def get_songs(indeces: 'list[int]'):
    ''' TODO
    Uses SQLAlchemy queries to return track data from their indeces
    '''
    return None


# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
        ),
    ],
    md=4,
)

column2 = dbc.Col(
    [

    ]
)

layout = dbc.Row([column1, column2])
