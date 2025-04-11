import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import json

from models.knn_classifier import KNNClassifierModel, generate_sample_data as generate_classifier_data
from models.knn_regression import KNNRegressionModel, generate_sample_data as generate_regression_data
from models.logistic_regression import LogisticRegressionModel, generate_sample_data as generate_logistic_data

# Initialize the models and generate data
def initialize_data(seed=42):
    np.random.seed(seed)
    
    # Generate fresh data for each model
    X_class, y_class = generate_classifier_data(n_samples=100)
    X_reg, y_reg = generate_regression_data(n_samples=100)
    X_log, y_log = generate_logistic_data(n_samples=100)
    
    # Initialize models
    knn_classifier = KNNClassifierModel()
    knn_regression = KNNRegressionModel()
    logistic_regression = LogisticRegressionModel()
    
    # Fit models with default parameters
    knn_classifier.fit(X_class, y_class)
    knn_regression.fit(X_reg, y_reg)
    logistic_regression.fit(X_log, y_log)
    
    return (knn_classifier, X_class, y_class,
            knn_regression, X_reg, y_reg,
            logistic_regression, X_log, y_log)

# Initialize all models and data
(knn_classifier, X_class, y_class,
 knn_regression, X_reg, y_reg,
 logistic_regression, X_log, y_log) = initialize_data()

def format_c_value(c_exp):
    """Format C value from exponential to decimal with proper formatting"""
    value = 10**c_exp
    if value >= 1:
        return f'{value:,.0f}'
    else:
        # For values less than 1, show appropriate decimal places
        decimal_places = abs(int(c_exp))
        return f'{value:.{decimal_places}f}'

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div(className='container', children=[
    # Store components for caching data
    dcc.Store(id='classifier-data', data={
        'X': X_class.tolist(),
        'y': y_class.tolist()
    }),
    dcc.Store(id='regression-data', data={
        'X': X_reg.tolist(),
        'y': y_reg.tolist()
    }),
    dcc.Store(id='logistic-data', data={
        'X': X_log.tolist(),
        'y': y_log.tolist()
    }),
    
    # Header Section
    html.Div([
        html.H1("Interactive Machine Learning Models", 
                className='text-center'),
        html.P([
            "Explore three fundamental machine learning algorithms through interactive visualizations. ",
            "Each tab demonstrates a different type of machine learning task and lets you adjust key parameters ",
            "to understand how they affect the model's behavior."
        ], className='header-description text-center')
    ]),
    
    # Main Content
    dcc.Tabs(className='dash-tabs', children=[
        # KNN Classifier Tab
        dcc.Tab(label='KNN Classifier', className='dash-tab', selected_className='dash-tab--selected', children=[
            html.Div(className='model-section', children=[
                html.H2("K-Nearest Neighbors Classifier", className='model-title'),
                html.Div([
                    html.P([
                        "The K-Nearest Neighbors (KNN) Classifier is like asking your neighbors to vote on a decision. ",
                        "Each point in the plot represents a data point, colored by its class - blue for cat photos and red for dog photos. ",
                        "The background colors show which type of photo the model predicts for each region - light blue areas are where the model predicts 'cat', ",
                        "and light red areas are where it predicts 'dog'. ",
                        "When k=1, each point only looks at its single closest neighbor. ",
                        "As k increases, each point considers more neighbors, leading to smoother decision boundaries."
                    ], className='model-description'),
                    html.P([
                        "Try it: Move the slider to see how the decision boundary changes. ",
                        "Notice how small k values create detailed, complex boundaries, ",
                        "while larger k values create smoother, more general boundaries."
                    ], className='model-interaction-hint')
                ]),
                html.Div(className='slider-container', children=[
                    html.Label("Number of Neighbors (k)", className='parameter-label'),
                    dcc.Slider(
                        id='knn-classifier-k',
                        min=1,
                        max=20,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(1, 21, 2)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode='drag'
                    ),
                    html.P("Drag the slider to adjust how many neighbors influence the classification.",
                          className='parameter-description')
                ]),
                dcc.Graph(id='knn-classifier-plot', className='dash-graph')
            ])
        ]),
        
        # KNN Regression Tab
        dcc.Tab(label='KNN Regression', className='dash-tab', selected_className='dash-tab--selected', children=[
            html.Div(className='model-section', children=[
                html.H2("K-Nearest Neighbors Regression", className='model-title'),
                html.Div([
                    html.P([
                        "KNN Regression predicts numerical values by averaging its neighbors. ",
                        "The blue dots show actual data points, and the red line shows the model's predictions. ",
                        "Instead of voting like in classification, KNN Regression takes the average of nearby points."
                    ], className='model-description'),
                    html.P([
                        "Try it: Watch how the prediction line (red) changes with different k values. ",
                        "Small k values will create a wiggly line that closely follows individual points, ",
                        "while larger k values create smoother predictions by averaging more points."
                    ], className='model-interaction-hint')
                ]),
                html.Div(className='slider-container', children=[
                    html.Label("Number of Neighbors (k)", className='parameter-label'),
                    dcc.Slider(
                        id='knn-regression-k',
                        min=1,
                        max=20,
                        step=1,
                        value=5,
                        marks={i: str(i) for i in range(1, 21, 2)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode='drag'
                    ),
                    html.P("Adjust k to control how smooth the prediction line becomes.",
                          className='parameter-description')
                ]),
                dcc.Graph(id='knn-regression-plot', className='dash-graph')
            ])
        ]),
        
        # Logistic Regression Tab
        dcc.Tab(label='Logistic Regression', className='dash-tab', selected_className='dash-tab--selected', children=[
            html.Div(className='model-section', children=[
                html.H2("Logistic Regression", className='model-title'),
                html.Div([
                    html.P([
                        "Logistic Regression finds a linear boundary to separate classes. ",
                        "Unlike KNN, it creates a straight(ish) boundary between classes. ",
                        "The C parameter controls how strictly the model fits the training data."
                    ], className='model-description'),
                    html.P([
                        "Try it: Adjust the C parameter to see how it affects the decision boundary. ",
                        "Small values (0.01) create simple, straight boundaries that might miss some points. ",
                        "Large values (100) create boundaries that try to perfectly separate all points, ",
                        "which might lead to overfitting."
                    ], className='model-interaction-hint')
                ]),
                html.Div(className='slider-container', children=[
                    html.Label("Regularization Strength (C)", className='parameter-label'),
                    dcc.Slider(
                        id='logistic-c',
                        min=-2,
                        max=2,
                        step=0.1,
                        value=0,
                        marks={i: f'C={format_c_value(i)}' for i in range(-2, 3)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode='drag'
                    ),
                    html.P([
                        "Slide to adjust the model's flexibility. ",
                        "Small values create simpler boundaries, large values create more complex boundaries."
                    ], className='parameter-description')
                ]),
                dcc.Graph(id='logistic-regression-plot', className='dash-graph')
            ])
        ])
    ])
])

# Callback for KNN Classifier
@app.callback(
    Output('knn-classifier-plot', 'figure'),
    [Input('knn-classifier-k', 'value'),
     Input('classifier-data', 'data')]
)
def update_knn_classifier(k, data):
    if k is None:
        k = 5
        
    # Get data from store
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    # Fit model and get prediction
    knn_classifier.fit(X, y, n_neighbors=k)
    xx, yy, Z = knn_classifier.get_decision_boundary(X)
    
    fig = {
        'data': [
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=Z,
                colorscale=[[0, '#e3f2fd'], [1, '#ffebee']],  # Light blue for cats (0), light red for dogs (1)
                showscale=False
            ),
            go.Scatter(
                x=X[y == 0, 0],
                y=X[y == 0, 1],
                mode='markers',
                name='Cat Photos',
                marker=dict(size=10, color='#3498db', symbol='circle')  # Darker blue for cat dots
            ),
            go.Scatter(
                x=X[y == 1, 0],
                y=X[y == 1, 1],
                mode='markers',
                name='Dog Photos',
                marker=dict(size=10, color='#e74c3c', symbol='circle')  # Darker red for dog dots
            )
        ],
        'layout': go.Layout(
            title=f'Pet Photo Classifier (k={k} neighbors)',
            xaxis_title='Brightness',  # Changed from Feature 1
            yaxis_title='Color Saturation',  # Changed from Feature 2
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            height=500,
            width=None,
            margin=dict(l=50, r=50, t=50, b=50),
            uirevision=True
        )
    }
    return fig

# Callback for KNN Regression
@app.callback(
    Output('knn-regression-plot', 'figure'),
    [Input('knn-regression-k', 'value'),
     Input('regression-data', 'data')]
)
def update_knn_regression(k, data):
    if k is None:
        k = 5
        
    # Get data from store
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    knn_regression.fit(X, y, n_neighbors=k)
    X_line, y_pred = knn_regression.get_prediction_line(X)
    
    return {
        'data': [
            go.Scatter(
                x=X.ravel(),
                y=y,
                mode='markers',
                name='Past House Sales',  # Changed from Training Data
                marker=dict(size=10, color='#3498db')
            ),
            go.Scatter(
                x=X_line.ravel(),
                y=y_pred,
                mode='lines',
                name='Predicted Prices',  # Changed from Prediction
                line=dict(color='#e74c3c', width=3)
            )
        ],
        'layout': go.Layout(
            title=f'House Price Predictor (k={k} neighbors)',
            xaxis_title='House Size (sq ft)',  # Changed from X
            yaxis_title='Price (thousands $)',  # Changed from y
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            height=500,
            width=None,
            margin=dict(l=50, r=50, t=50, b=50),
            uirevision=True
        )
    }

# Callback for Logistic Regression
@app.callback(
    Output('logistic-regression-plot', 'figure'),
    [Input('logistic-c', 'value'),
     Input('logistic-data', 'data')]
)
def update_logistic_regression(c_exp, data):
    if c_exp is None:
        c_exp = 0
        
    # Get data from store
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    C = 10**c_exp
    logistic_regression.fit(X, y, C=C)
    xx, yy, Z = logistic_regression.get_decision_boundary(X)
    
    return {
        'data': [
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=Z,
                colorscale=[[0, '#ffebee'], [1, '#e3f2fd']],  # Very light red to very light blue
                showscale=False
            ),
            go.Scatter(
                x=X[y == 0, 0],
                y=X[y == 0, 1],
                mode='markers',
                name='Spam Emails',  # Changed from Class 0
                marker=dict(size=10, color='#3498db', symbol='circle')
            ),
            go.Scatter(
                x=X[y == 1, 0],
                y=X[y == 1, 1],
                mode='markers',
                name='Regular Emails',  # Changed from Class 1
                marker=dict(size=10, color='#e74c3c', symbol='circle')
            )
        ],
        'layout': go.Layout(
            title=f'Email Spam Filter (C={format_c_value(c_exp)})',
            xaxis_title='Word Frequency',  # Changed from Feature 1
            yaxis_title='Email Length',  # Changed from Feature 2
            hovermode='closest',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            height=500,
            width=None,
            margin=dict(l=50, r=50, t=50, b=50),
            uirevision=True
        )
    }

if __name__ == '__main__':
    app.run(debug=True)
