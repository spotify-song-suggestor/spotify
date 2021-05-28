import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash_table import DataTable
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

def get_songs(indeces: 'list[int]') -> 'list[spotify]':
    '''
    Uses SQLAlchemy queries to return track data from their indeces
    '''
    print('indeces:',indeces)
    data = [spotify.query.filter(spotify.id == x).one() for x in indeces]
    print('Data:', data)
    return data


def recommend(index: int, n: int=5) -> 'tuple[np.ndarray]':
    '''
    ### Parameters
    index: index of song
    n: number of recommendations to pull
    returns: (dist, ind), array of distances, array of indeces for recommended songs. Includes
    original song.
    '''
    return knn.kneighbors([encodings[index]], n_neighbors=5)


def plot_graph(data: 'list[spotify]'):
    '''Function to plot bar graph
       Input: Data objects returned by get_song()
       Output: Plotly Bar Graph
    '''   
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
    fig = px.bar(x=g_name, y=g_popularity,
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
def get_songs_via_features(features: list, n_songs: int=5) -> 'list[int]':
    '''
    Converts input into the model's encoding, then runs it through the
    K-NearestNeighbors models
    Returns: 
    ### Parameters
    features: A list of all features required to run the model.
    The model encoder expects these inputs in this order:
    duration_ms,
    explicit,
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
    vec = model.encoder(np.array(features).reshape(1, -1))
    print('vec:',vec)
    _, indeces = knn.kneighbors(vec, n_songs)
    indeces = indeces.reshape(-1).tolist()
    print('reshaped indeces in get_songs_via_features: ', indeces)
    return indeces



# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout

# TODO: These values must match the input range for the model, meaning between
# 0 and 1. There also must be the complete set of features the model uses, as
# described above.
column2 = dbc.Col(
    [
        dcc.Markdown('''###### Duration'''),
        dcc.Slider(
            id='duration-slider',
            min=0,
            max=1,
            value=0.5,
            step=0.1,
        ),
        dcc.Markdown('', id='duration-slider-container'),

        dcc.Markdown('''###### Explicit'''),
        dcc.Slider(
            id='explicit-slider',
            min=0,
            max=1,
            step=1,
            value=0,
        ),
        dcc.Markdown('', id='explicit-slider-container'),

        dcc.Markdown('''###### Danceability'''),
        dcc.Slider(
            id='danceability-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.5,
        ),
        dcc.Markdown('', id='danceability-slider-container'),

        dcc.Markdown('''###### Energy'''),
        dcc.Slider(
            id='energy-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.5,
        ),
        dcc.Markdown('', id='energy-slider-container'),

        dcc.Markdown('''###### Key'''),
        dcc.Slider(
            id='key-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.5,
        ),
        dcc.Markdown('', id='key-slider-container'),

        dcc.Markdown('''###### Loudness'''),
        dcc.Slider(
            id='loudness-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.5,
        ),
        dcc.Markdown('', id='loudness-slider-container'),        

        dcc.Markdown('''###### Mode'''),
        dcc.Slider(
            id='mode-slider',
            min=0,
            max=1,
            step=1,
            value=0.5,
        ),
        dcc.Markdown('', id='mode-slider-container'),

        

    ],
    
)

row1 = dbc.Row(
    [
        dcc.Markdown(
            """
            **Instructions**: Adjust the attribute sliders. Your prediction outcome will update dynamically. 
            Attribute Definitions:
            * **Duration** - Length of the song in ms
            * **Loudness** -How loud the songs are
            * **Danceability** - Danceability describes how suitable a track is for dancing based on a combination of musical elements.
            * **Energy** - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.
            * **Instrumentalness** - The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content.            
            * **Liveness** - how present audience are in artists songs
            * **US Popularity Level** - Highest point on Billboard  
            * **Speechiness** - The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value.
            * **Tempo** - The overall estimated tempo of a track in beats per minute (BPM).
            * **Year of Release** - Year the track was released 
            """
        ),

     ],
    

)



# Takes inputs from user and returning to show their selection
@app.callback(
    dash.dependencies.Output('duration-slider-container', 'children'),
    [dash.dependencies.Input('duration-slider', 'value')])
def update_output(value):
    return 'Duration = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('explicit-slider-container', 'children'),
    [dash.dependencies.Input('explicit-slider', 'value')])
def update_output(value):
    return 'Explicit = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('danceability-slider-container', 'children'),
    [dash.dependencies.Input('danceability-slider', 'value')])
def update_output(value):
    return 'Danceability = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('energy-slider-container', 'children'),
    [dash.dependencies.Input('energy-slider', 'value')])
def update_output(value):
    return 'Energy = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('key-slider-container', 'children'),
    [dash.dependencies.Input('key-slider', 'value')])
def update_output(value):
    return 'Key = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('loudness-slider-container', 'children'),
    [dash.dependencies.Input('loudness-slider', 'value')])
def update_output(value):
    return 'Loudness = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('mode-slider-container', 'children'),
    [dash.dependencies.Input('mode-slider', 'value')])
def update_output(value):
    return 'Mode = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('speechiness-slider-container', 'children'),
    [dash.dependencies.Input('speechiness-slider', 'value')])
def update_output(value):
    return 'Speechiness = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('acousticness-slider-container', 'children'),
    [dash.dependencies.Input('acousticness-slider', 'value')])
def update_output(value):
    return 'Acousticness = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('instrumentalness-slider-container', 'children'),
    [dash.dependencies.Input('instrumentalness-slider', 'value')])
def update_output(value):
    return 'Instrumentalness = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('liveness-slider-container', 'children'),
    [dash.dependencies.Input('liveness-slider', 'value')])
def update_output(value):
    return 'Liveness = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('valence-slider-container', 'children'),
    [dash.dependencies.Input('valence-slider', 'value')])
def update_output(value):
    return 'Valence = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('tempo-slider-container', 'children'),
    [dash.dependencies.Input('tempo-slider', 'value')])
def update_output(value):
    return 'Tempo = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('time-signature-slider-container', 'children'),
    [dash.dependencies.Input('time-signature-slider', 'value')])
def update_output(value):
    return 'Time Signature = "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('popularity-slider-container', 'children'),
    [dash.dependencies.Input('popularity-slider', 'value')])
def update_output(value):
    return 'Popularity = "{}"'.format(value)

# Uses the inputs to the user to generate the recommendation
@app.callback(
    Output('memory', 'data'),
    [
        Input('duration-slider', 'value'),
        Input('explicit-slider', 'value'),
        Input('danceability-slider', 'value'),
        Input('energy-slider', 'value'),
        Input('key-slider', 'value'),
        Input('loudness-slider', 'value'),
        Input('mode-slider', 'value'),
        Input('speechiness-slider', 'value'),
        Input('acousticness-slider', 'value'),
        Input('instrumentalness-slider', 'value'),
        Input('liveness-slider', 'value'),
        Input('valence-slider', 'value'),
        Input('tempo-slider', 'value'),
        Input('time-signature-slider', 'value'),
        Input('popularity-slider', 'value')])
def update_list(duration_ms,
                explicit,
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
    # Utilize get_songs_via_features to grab songs to display for user to pick
    # from. Route those songs via clicks to get recommendations for those songs.
    # This is just one method of doing this, discussion surrounding this would
    # be great
    features = [duration_ms,
                explicit,
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
                popularity]

    indeces = get_songs_via_features(features)
    print('update_list indeces: ', indeces)
    songs = get_songs(indeces)
    print('update_list songs:', songs)
    return [song.id for song in songs]

rec_col = dbc.Col(
    [
        
        # Container to display recommendations
        dcc.Markdown('#### Recommended Songs:', style={'textAlign': 'center'}),
        dcc.Markdown('', id='recommendation-content', style={
        'textAlign':'center',
        'font-size':20}),
        #Sanity Test
        #dcc.Graph(figure=plot_graph(data=get_songs([0,6,1609,34455]))),
        dcc.Graph(id = 'my-graph', style={"height": "50%", "width" : "80%", "align": "center", "margin": "auto"}),
        dcc.Store(id='memory')
    ]
)
column3 = dbc.Col([
    dcc.Markdown('''###### Speechiness'''),
        dcc.Slider(
            id='speechiness-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.5,
        ),
        dcc.Markdown('', id='speechiness-slider-container'),

        dcc.Markdown('''###### Acousticness'''),
        dcc.Slider(
            id='acousticness-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.5,
        ),
        dcc.Markdown('', id='acousticness-slider-container'),

        dcc.Markdown('''###### Instrumentalness'''),
        dcc.Slider(
            id='instrumentalness-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.5,
        ),
        dcc.Markdown('', id='instrumentalness-slider-container'),

        dcc.Markdown('''###### Liveness'''),
        dcc.Slider(
            id='liveness-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.5,
        ),
        dcc.Markdown('', id='liveness-slider-container'),

        dcc.Markdown('''###### Valence'''),
        dcc.Slider(
            id='valence-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.5,
        ),
        dcc.Markdown('', id='valence-slider-container'),

        dcc.Markdown('''###### Tempo'''),
        dcc.Slider(
            id='tempo-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.5,
        ),
        dcc.Markdown('', id='tempo-slider-container'),

        dcc.Markdown('''###### Time Signature'''),
        dcc.Slider(
            id='time-signature-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.5,
        ),
        dcc.Markdown('', id='time-signature-slider-container'),

        dcc.Markdown('''######  Popularity'''),
        dcc.Slider(
            id='popularity-slider',
            min=0,
            max=1,
            step=0.1,
            value=0.5,
        ),
        dcc.Markdown('', id='popularity-slider-container'),
    ],
    md=3,
)

@app.callback(
             Output('my-graph', 'figure'),
             Input('memory', 'data'))
def update_figure(data):
    return plot_graph(data=get_songs(data))


@app.callback(
    Output('recommendation-content', 'children'),
    Input('memory', 'data'))
def update_recommended(data):
    return [song.name + '\n' for song in get_songs(data)]


layout = [dbc.Row([column3, column2,rec_col]), dbc.Row(row1)]
