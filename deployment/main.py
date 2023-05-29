import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime
import base64
import io
import PIL.Image as Image
import dash
from dash import dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from src.model import StyleTransfer

st = StyleTransfer(total_variation_weight= 0,style_weight=1e-1,
                   content_weight=1e4, steps_per_epoch = 100,
                   learning_rate=0.05, max_dim=1200)

parent = "deployment"

output_path = "/assets/trymodel/output.jpg"

style_path = "deployment/assets/trymodel/style_image.png"

content_path = "deployment/assets/trymodel/content_image.png"

app = dash.Dash(external_stylesheets=[dbc.themes.SKETCHY])

server = app.server
# Define the list of image options
image_options = [
    {'label': 'Starry night', 'value': '1'},
    {'label': 'Wheat Field', 'value': '2'},
    {'label': 'Pink Peach Tree', 'value': '3'},
    {'label': 'The scream', 'value': '4'},
]

app.layout = html.Div(
    dbc.Container(
        [
            html.Br(),
            html.Div(
                [
                    html.H2("Style Transfer Gallery", style={'text-align': 'center'}),
                    html.Div(
                        [
                            html.Label("Select the style:"),
                            dcc.Dropdown(
                                id='image-dropdown',
                                options=image_options,
                                value=image_options[0]['value']
                            ,style={'width': '100%'})
                        ]
                    ),
                    html.Br(),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H4('Content Image', style={'text-align': 'center'}),
                                    html.Div(
                                        html.Img(id='image1', style={'max-width': '320px', 'max-height': '320px', 'object-fit': 'contain'}),
                                        style={'width': '320px', 'height': '320px', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin': '10px'}
                                    ),
                                ],
                                style={'display': 'inline-block', 'vertical-align': 'top'}
                            ),
                            html.Div([
                                html.Br(),
                                html.Br(),
                                html.Br(),
                                "+"],
                                style={'display': 'inline-block', 'vertical-align': 'top', 'font-size': '40px'}
                            ),
                            html.Div(
                                [
                                    html.H4('Style Image', style={'text-align': 'center'}),
                                    html.Div(
                                        html.Img(id='image2', style={'max-width': '320px', 'max-height': '320px', 'object-fit': 'contain'}),
                                        style={'width': '320px', 'height': '320px', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin': '10px'}
                                    ),
                                ],
                                style={'display': 'inline-block', 'vertical-align': 'top'}
                            ),
                            html.Div([
                                html.Br(),
                                html.Br(),
                                html.Br(),
                                "\u2192"],
                                style={'display': 'inline-block', 'vertical-align': 'top', 'font-size': '40px'}
                            ),
                            html.Div(
                                [
                                    html.H4('Generated Image', style={'text-align': 'center'}),
                                    html.Div(
                                        html.Img(id='image3', style={'max-width': '320px', 'max-height': '320px', 'object-fit': 'contain'}),
                                        style={'width': '320px', 'height': '320px', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin': '10px'}
                                    ),
                                ],
                                style={'display': 'inline-block', 'vertical-align': 'top'}
                            ),
                        ],
                        style={'text-align': 'center'}
                    )
                ]
            ),html.Div([
    html.Hr(),
    html.H2("Use the model", style={'text-align': 'center'}),
    html.Div([
            html.Div(
                [   
                     dcc.Upload(
                        id='upload-content',
                        children=html.Div([
                            html.A('Drag or Select Content Image')
                        ]),
                        style={
                            'height': '50px',
                            'lineHeight': '50px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px',
                        },
                        multiple=False
                    ),
                    html.H4('Content Image', style={'text-align': 'center'}),
                    html.Div(
                        html.Img(id='content-image', style={'max-width': '320px', 'max-height': '320px', 'object-fit': 'contain'}),
                        style={'width': '320px', 'height': '320px', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin': '10px'}
                    ),
                ],
                style={'display': 'inline-block', 'vertical-align': 'top'}
            ),
            html.Div(
                [   dcc.Upload(
                            id='upload-style',
                            children=html.Div([
                                html.A('Drag or Select Style Image')
                            ]),
                            style={
                                'height': '50px',
                                'lineHeight': '50px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            multiple=False
                        ),
                    html.H4('Style Image', style={'text-align': 'center'}),
                    html.Div(
                        html.Img(id='style-image', style={'max-width': '320px', 'max-height': '320px', 'object-fit': 'contain'}),
                        style={'width': '320px', 'height': '320px', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin': '10px'}
                    ),
                ],
                style={'display': 'inline-block', 'vertical-align': 'top'}
            ),
            html.Div(
                [   dbc.Row([html.Div(
                            [
                                dbc.Input(type="number", min=1, max=20, step=1,placeholder="n epochs",style={'height': '50px'},id="epochs"),
                            ],
                        style={'width': '40%', 'height': '50px'  ,'margin': '10px'}),
                             dbc.Button("Generate Image", color="primary", className="me-1",id="button",style={ 'width': '40%', 'height': '50px' ,'margin': '10px'})]),
                    html.H4('Generated Image', style={'text-align': 'center'}),
                    html.Div(
                        html.Img(id='generated-image', style={'max-width': '320px', 'max-height': '320px', 'object-fit': 'contain'}),
                        style={'width': '320px', 'height': '320px', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center', 'margin': '10px'}
                    ),
                ],
                style={'display': 'inline-block', 'vertical-align': 'top'}
            ),
    ],style={'text-align': 'center'})
]),
        html.Br(),
        html.Div([
        html.Small(f'© {datetime.now().year}. Created by: '),
        html.A('Osama Fayez', href='https://www.linkedin.com/in/osama-oun/', target='_blank', style={ "color" : "#7d2b29"}),
        ' and ', html.A('Israa Okil', href='https://www.linkedin.com/in/israa-okil/', target='_blank', style={ "color" : "#7d2b29"})
    ],
        id='footer',style={"text-align" : "center"}
    ),
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


# Callback to handle the uploaded style image
@app.callback(Output('style-image', 'src'),
              [Input('upload-style', 'contents')],
              [State('upload-style', 'filename')])
def update_style_image(content, filename):
    if content is not None:
        _, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        # Save the image
        image.save(style_path)
        # Return the image source
        return 'data:image/png;base64,{}'.format(content_string)
    else:
        raise PreventUpdate


# Callback to handle the uploaded content image
@app.callback(Output('content-image', 'src'),
              [Input('upload-content', 'contents')],
              [State('upload-content', 'filename')])
def update_content_image(content, filename):
    if content is not None:
        _, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        # Save the image
        image.save(content_path)
        # Return the image source
        return 'data:image/png;base64,{}'.format(content_string)
    else:
        raise PreventUpdate
    
# Callback to generate image
@app.callback(Output('generated-image', 'src'),
              [Input('button', 'n_clicks')],
              [Input('epochs', 'value')])
def generate_image(n_clicks, epochs):
    if n_clicks:
        print(f"n of epochs = {epochs}")
        st.style_transfer(content_path, style_path, parent + output_path, epochs)
        return output_path
    else:
        raise PreventUpdate
    
app.run_server(debug=False)
