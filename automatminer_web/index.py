import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from automatminer_web.app import app

external_font = \
    html.Link(
        href="https://fonts.googleapis.com/css?family=Ubuntu&display=swap",
        rel="stylesheet",
        className="is-hidden"
    )

external_bulma = \
    html.Link(
        href="https://cdn.jsdelivr.net/npm/bulma@0.8.0/css/bulma.css",
        rel="stylesheet",
        className="is-hidden"
    )
external_stylesheets = html.Div(children=[external_bulma, external_font])
location = dcc.Location(id="url", refresh=False)
app_container = html.Div(id="app_container")


app.layout = html.Div(
    [
        external_stylesheets,
        location,
        app_container
    ],
    className="ammw-dark-bg"
)


# Top level callbacks
#######################
# callbacks for loading different apps or are present on every page

@app.callback(
    Output('app_container', 'children'),
    [Input('url', 'pathname')]
)
def display_page(path):
    if str(path).strip() in ["/", "/search"] or not path:
        return html.Div("404", className="has-text-centered")
