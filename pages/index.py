# Imports from 3rd party libraries
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# Imports from this application
from app import app

# 2 column layout. 1st column width = 4/12
# https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout
column1 = dbc.Col(
    [
        dcc.Markdown(
            """

            ## What shall you listen to?

            This app will help you select what to listen next. Be impressed!


            From over a hundred thousand song's to select from. This running app with a sophisticated deep neural net machine will let you discover new favorite songs!


            """
        ),
        dcc.Link(dbc.Button('Try it!', color='success', outline=True), href='/predictions')
    ],
    md=4,
)

gapminder = px.data.gapminder()
fig = px.scatter(gapminder.query("year==2007"), x="gdpPercap", y="lifeExp", size="pop", color="continent",
           hover_name="country", log_x=True, size_max=60)

column2 = dbc.Col(
    [
        #dcc.Graph(figure=fig),
        
    ]
)

column3 = dbc.Col(
    [
        #dcc.Graph(figure=fig),
        html.Div(html.Img(src=app.get_asset_url('cloud.png')), style={'height':'100%', 'width':'100%'})
    ]
)

layout = dbc.Row([column1, column2, column3])