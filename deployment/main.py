import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])

# Define the list of image options
image_options = [
    {'label': 'Style 1', 'value': '1'},
    {'label': 'Style 2', 'value': '2'},
]

app.layout = html.Div(
    dbc.Container(
        [
            html.Div(
                [
                    html.H1("Available styles"),
                    html.Div(
                        [
                            html.Label("Select the style:"),
                            dcc.Dropdown(
                                id='image-dropdown',
                                options=image_options,
                                value=image_options[0]['value']
                            )
                        ]
                    ),
                    html.Div(
                        [
                            html.Img(id='image1', style={'width': '300px', 'height': '300px', 'margin-right': '10px'}),
                            html.Img(id='image2', style={'width': '300px', 'height': '300px', 'margin-right': '10px'}),
                            html.Img(id='image3', style={'width': '300px', 'height': '300px', 'margin-right': '10px'})
                        ],
                    )
                ]
            )
        ]
    )
)

@app.callback(
    dash.dependencies.Output('image1', 'src'),
    dash.dependencies.Output('image2', 'src'),
    dash.dependencies.Output('image3', 'src'),
    [dash.dependencies.Input('image-dropdown', 'value')]
)
def update_images(number):
    # Update the source of the three images based on the selected image
    image_src1 = f'/assets/content_images/content_image{number}.jpg'
    image_src2 = f'/assets/style_images/style_image{number}.jpg'
    image_src3 = f'/assets/output_images/output_image{number}.jpg'

    return image_src1, image_src2, image_src3

if __name__ == '__main__':
    app.run_server(debug=True)