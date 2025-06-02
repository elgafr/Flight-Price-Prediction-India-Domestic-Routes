import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import plotly.graph_objects as go
import base64
import io

# Initialize Dash app
app = dash.Dash(__name__)

# Load initial files
unique_values = joblib.load('models/unique_values.joblib')
airline_flight_map = joblib.load('models/airline_flight_map.joblib')

class SFSSelector:
    def __init__(self, selected_indices):
        self.selected_indices = selected_indices
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, self.selected_indices]

# Load model results and handle duplicates
try:
    results_df = pd.read_csv('models/model_results.csv')
    print("Contents of results_df:")
    print(results_df)
    results_df['Feature Selection'] = results_df['Feature Selection'].fillna('None')
    results_df = results_df.groupby(['Model', 'Feature Selection'], as_index=False)['RMSE'].mean()
except FileNotFoundError:
    results_df = pd.DataFrame(columns=['Model', 'Feature Selection', 'RMSE'])
    print("Warning: models/model_results.csv not found. Using empty DataFrame.")

# Define feature lists
categorical_columns = ['airline', 'flight', 'source_city', 'departure_time', 'stops', 
                      'arrival_time', 'destination_city', 'class']
numeric_columns = ['duration', 'days_left']

# Get list of models from the models folder
model_files = [os.path.splitext(f)[0] for f in os.listdir('models') if f.endswith('.joblib') and ('DecisionTree' in f or 'RandomForest' in f or 'XGBoost' in f)]
default_model = model_files[0] if model_files else 'DecisionTree_None'

# Function to calculate confidence scores
def get_confidence_scores(model, X_test, model_type):
    if model_type == "DecisionTree":
        n_bootstraps = 100
        predictions = []
        for _ in range(n_bootstraps):
            idx = np.random.choice(X_test.shape[0], size=X_test.shape[0], replace=True)
            X_boot = X_test[idx]
            predictions.append(model.predict(X_boot))
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        confidence_scores = 1 / (1 + std_pred / (mean_pred + 1e-10))
    elif model_type == "RandomForest":
        tree_predictions = np.array([tree.predict(X_test) for tree in model.estimators_])
        mean_pred = tree_predictions.mean(axis=0)
        std_pred = tree_predictions.std(axis=0)
        confidence_scores = 1 / (1 + std_pred / (mean_pred + 1e-10))
    elif model_type == "XGBoost":
        n_samples = 100
        predictions = []
        for _ in range(n_samples):
            noise = np.random.normal(0, 0.01, X_test.shape)
            X_noisy = X_test + noise
            predictions.append(model.predict(X_noisy))
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        confidence_scores = 1 / (1 + std_pred / (mean_pred + 1e-10))
    return confidence_scores

# Parse uploaded CSV file
def parse_csv(contents, filename):
    if contents is None:
        return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
        else:
            return None
    except Exception as e:
        print(f"Error parsing CSV: {str(e)}")
        return None

# Validate CSV data
def validate_csv_data(df):
    required_columns = categorical_columns + numeric_columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {', '.join(missing_cols)}"
    
    # Validate categorical values
    for col in categorical_columns:
        valid_values = unique_values.get(col, [])
        invalid_rows = df[~df[col].isin(valid_values)]
        if not invalid_rows.empty:
            return False, f"Invalid values in {col}: {invalid_rows[col].unique()}"
    
    # Validate numeric values
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column {col} must be numeric"
        if df[col].isna().any():
            return False, f"Column {col} contains missing values"
    
    # Validate flight-airline mapping
    for _, row in df.iterrows():
        if row['flight'] not in airline_flight_map.get(row['airline'], []):
            return False, f"Flight {row['flight']} is not valid for airline {row['airline']}"
    
    return True, ""

# If no models are found, display an error
if not model_files:
    app.layout = html.Div([
        html.H1("Flight Price (₹) Prediction – India Domestic Routes", className="text-3xl font-bold text-center text-blue-700 mb-6"),
        html.Div("Error: No models found in 'models/' folder. Please run the training script first.", className="text-red-600 text-center")
    ], className="container mx-auto p-6 w-full")
else:
    # Layout with upload component
    app.layout = html.Div([
        html.Link(
            rel='stylesheet',
            href='https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css'
        ),
        html.H1("Flight Price (₹) Prediction – India Domestic Routes", className="text-4xl font-extrabold text-center text-blue-700 mb-8"),
        html.Div([
            # Model dropdown
            html.Div([
                html.Label("Select Model:", className="block text-sm font-medium text-blue-700"),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[{'label': model, 'value': model} for model in model_files],
                    value=default_model,
                    className="mt-1 block w-full p-2 border border-blue-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                ),
            ], className="mb-4"),
            # File upload component
            html.Div([
                html.Label("Upload CSV File:", className="block text-sm font-medium text-blue-700"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Button('Upload CSV', className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"),
                    multiple=False,
                    className="mt-1 block w-full"
                ),
                html.Div(id='upload-error', className="text-red-600 mt-2")
            ], className="mb-4"),
            # Existing input fields (unchanged)
            html.Div([
                html.Div([
                    html.Label("Airline:", className="block text-sm font-medium text-blue-700"),
                    dcc.Dropdown(
                        id='airline-dropdown',
                        options=[{'label': airline, 'value': airline} for airline in unique_values['airline']],
                        value=unique_values['airline'][0],
                        className="mt-1 block w-full p-2 border border-blue-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    ),
                ], className="w-1/3 pr-2"),
                html.Div([
                    html.Label("Flight:", className="block text-sm font-medium text-blue-700"),
                    dcc.Dropdown(
                        id='flight-dropdown',
                        value=unique_values['flight'][0],
                        className="mt-1 block w-full p-2 border border-blue-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    ),
                ], className="w-1/3 px-2"),
                html.Div([
                    html.Label("Source City:", className="block text-sm font-medium text-blue-700"),
                    dcc.Dropdown(
                        id='source_city-dropdown',
                        options=[{'label': city, 'value': city} for city in unique_values['source_city']],
                        value=unique_values['source_city'][0],
                        className="mt-1 block w-full p-2 border border-blue-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    ),
                ], className="w-1/3 pl-2"),
            ], className="flex mb-4"),
            html.Div([
                html.Div([
                    html.Label("Departure Time:", className="block text-sm font-medium text-blue-700"),
                    dcc.Dropdown(
                        id='departure_time-dropdown',
                        options=[{'label': time, 'value': time} for time in unique_values['departure_time']],
                        value=unique_values['departure_time'][0],
                        className="mt-1 block w-full p-2 border border-blue-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    ),
                ], className="w-1/3 pr-2"),
                html.Div([
                    html.Label("Stops:", className="block text-sm font-medium text-blue-700"),
                    dcc.Dropdown(
                        id='stops-dropdown',
                        options=[{'label': stop, 'value': stop} for stop in unique_values['stops']],
                        value=unique_values['stops'][0],
                        className="mt-1 block w-full p-2 border border-blue-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    ),
                ], className="w-1/3 px-2"),
                html.Div([
                    html.Label("Arrival Time:", className="block text-sm font-medium text-blue-700"),
                    dcc.Dropdown(
                        id='arrival_time-dropdown',
                        options=[{'label': time, 'value': time} for time in unique_values['arrival_time']],
                        value=unique_values['arrival_time'][0],
                        className="mt-1 block w-full p-2 border border-blue-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    ),
                ], className="w-1/3 pl-2"),
            ], className="flex mb-4"),
            html.Div([
                html.Div([
                    html.Label("Destination City:", className="block text-sm font-medium text-blue-700"),
                    dcc.Dropdown(
                        id='destination_city-dropdown',
                        options=[{'label': city, 'value': city} for city in unique_values['destination_city']],
                        value=unique_values['destination_city'][0],
                        className="mt-1 block w-full p-2 border border-blue-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    ),
                ], className="w-1/2 pr-2"),
                html.Div([
                    html.Label("Class:", className="block text-sm font-medium text-blue-700"),
                    dcc.Dropdown(
                        id='class-dropdown',
                        options=[{'label': cls, 'value': cls} for cls in unique_values['class']],
                        value=unique_values['class'][0],
                        className="mt-1 block w-full p-2 border border-blue-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    ),
                ], className="w-1/2 pl-2"),
            ], className="flex mb-4"),
            html.Div([
                html.Div([
                    html.Label("Duration (hours):", className="block text-sm font-medium text-blue-700"),
                    dcc.Input(
                        id='duration-input',
                        type='number',
                        value=10.0,
                        step=0.1,
                        className="mt-1 block w-full p-2 border border-blue-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    ),
                ], className="w-1/2 pr-2"),
                html.Div([
                    html.Label("Days Left:", className="block text-sm font-medium text-blue-700"),
                    dcc.Input(
                        id='days_left-input',
                        type='number',
                        value=30,
                        step=1,
                        className="mt-1 block w-full p-2 border border-blue-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                    ),
                ], className="w-1/2 pl-2"),
            ], className="flex mb-4"),
            html.Div([
                html.Button('Predict Price', id='predict-button', n_clicks=0, className="w-full bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"),
            ], className="flex justify-center"),
            html.Div(id='prediction-output', className="mt-6 p-4 rounded-md text-center"),
            dcc.Store(id='prediction-history', data=[]),
            dcc.Store(id='last-click-timestamp', data=None),
            html.Div([
                html.H2("Prediction History", className="text-2xl font-bold text-blue-700 mb-4 mt-24"),
                html.Div([
                    html.Button('Clear History', id='clear-history-button', n_clicks=0, className="bg-blue-400 text-white px-4 py-2 rounded-md hover:bg-blue-500 mb-4"),
                ], className="flex justify-end mb-2"),
                html.Div(id='history-table', className="overflow-x-auto")
            ]),
            html.Div([
                html.H2("Model RMSE Comparison", className="text-2xl font-bold text-blue-700 mt-24 mb-4"),
                dcc.Graph(id='metrics-chart')
            ])
        ], className="w-full mx-auto bg-white p-6 rounded-lg shadow-md")
    ], className="container mx-auto p-6 bg-blue-50 min-h-screen w-full")

    # Callback to update flight dropdown based on airline
    @app.callback(
        Output('flight-dropdown', 'options'),
        Input('airline-dropdown', 'value')
    )
    def update_flight_dropdown(airline):
        flights = airline_flight_map.get(airline, [])
        return [{'label': flight, 'value': flight} for flight in flights]

    # Callback to handle predictions and history (updated to include CSV upload)
    @app.callback(
        [Output('prediction-output', 'children'),
         Output('prediction-history', 'data'),
         Output('history-table', 'children'),
         Output('last-click-timestamp', 'data'),
         Output('upload-error', 'children')],
        [Input('predict-button', 'n_clicks'),
         Input('clear-history-button', 'n_clicks'),
         Input('upload-data', 'contents')],
        [State('model-dropdown', 'value'),
         State('airline-dropdown', 'value'),
         State('flight-dropdown', 'value'),
         State('source_city-dropdown', 'value'),
         State('departure_time-dropdown', 'value'),
         State('stops-dropdown', 'value'),
         State('arrival_time-dropdown', 'value'),
         State('destination_city-dropdown', 'value'),
         State('class-dropdown', 'value'),
         State('duration-input', 'value'),
         State('days_left-input', 'value'),
         State('prediction-history', 'data'),
         State('last-click-timestamp', 'data'),
         State('upload-data', 'filename')]
    )
    def predict_price_and_update_history(n_clicks_predict, n_clicks_clear, upload_contents, selected_model, airline, flight, source_city, 
                                        departure_time, stops, arrival_time, destination_city, class_, duration, days_left, history, last_click, filename):
        ctx = dash.callback_context
        if not ctx.triggered:
            return [""], history, generate_history_table(history), None, ""

        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        current_time = datetime.now().timestamp()

        # Debounce: Prevent rapid clicks
        if last_click is not None and (current_time - last_click) < 1:
            return [""], history, generate_history_table(history), last_click, ""

        # Handle clear history
        if triggered_id == 'clear-history-button' and n_clicks_clear > 0:
            return [""], [], [], current_time, ""

        # Handle CSV upload
        if triggered_id == 'upload-data' and upload_contents is not None:
            df = parse_csv(upload_contents, filename)
            if df is None:
                return [""], history, generate_history_table(history), current_time, "Error: Please upload a valid CSV file."
            
            is_valid, error_msg = validate_csv_data(df)
            if not is_valid:
                return [""], history, generate_history_table(history), current_time, f"Error: {error_msg}"
            
            try:
                model = joblib.load(f'models/{selected_model}.joblib')
                predictions = model.predict(df)
                X_processed = model.named_steps['preprocessor'].transform(df)
                if 'selector' in model.named_steps:
                    X_processed = model.named_steps['selector'].transform(X_processed)
                model_type = selected_model.split('_')[0]
                conf_scores = get_confidence_scores(model.named_steps['model'], X_processed, model_type)
                
                new_entries = []
                for i, row in df.iterrows():
                    new_entry = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'model': selected_model,
                        'airline': row['airline'],
                        'flight': row['flight'],
                        'source_city': row['source_city'],
                        'departure_time': row['departure_time'],
                        'stops': row['stops'],
                        'arrival_time': row['arrival_time'],
                        'destination_city': row['destination_city'],
                        'class': row['class'],
                        'duration': float(row['duration']),
                        'days_left': int(row['days_left']),
                        'prediction': float(predictions[i]),
                        'confidence': float(conf_scores[i])
                    }
                    if not history or history[-1] != new_entry:
                        new_entries.append(new_entry)
                
                history.extend(new_entries)
                return [html.Div(f"Processed {len(new_entries)} predictions from CSV.", className="text-green-600")], history, generate_history_table(history), current_time, ""
            except Exception as e:
                return [""], history, generate_history_table(history), current_time, f"Error in prediction: {str(e)}"

        # Handle single prediction (original functionality)
        if triggered_id == 'predict-button' and n_clicks_predict > 0:
            if None in [airline, flight, source_city, departure_time, stops, 
                        arrival_time, destination_city, class_, duration, days_left]:
                return [html.Div("Error: All fields must be filled.", className="text-red-600")], history, generate_history_table(history), current_time, ""
            
            try:
                model = joblib.load(f'models/{selected_model}.joblib')
                input_data = pd.DataFrame({
                    'airline': [airline],
                    'flight': [flight],
                    'source_city': [source_city],
                    'departure_time': [departure_time],
                    'stops': [stops],
                    'arrival_time': [arrival_time],
                    'destination_city': [destination_city],
                    'class': [class_],
                    'duration': [duration],
                    'days_left': [days_left]
                })
                prediction = model.predict(input_data)[0]
                X_processed = model.named_steps['preprocessor'].transform(input_data)
                if 'selector' in model.named_steps:
                    X_processed = model.named_steps['selector'].transform(X_processed)
                model_type = selected_model.split('_')[0]
                conf_score = get_confidence_scores(model.named_steps['model'], X_processed, model_type)[0]
                
                new_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'model': selected_model,
                    'airline': airline,
                    'flight': flight,
                    'source_city': source_city,
                    'departure_time': departure_time,
                    'stops': stops,
                    'arrival_time': arrival_time,
                    'destination_city': destination_city,
                    'class': class_,
                    'duration': float(duration),
                    'days_left': int(days_left),
                    'prediction': float(prediction),
                    'confidence': float(conf_score)
                }
                
                if history and history[-1] == new_entry:
                    return [html.Div("This prediction is already in history.", className="text-blue-600")], history, generate_history_table(history), current_time, ""
                
                history.append(new_entry)
                prediction_output = html.Div([
                    html.H3(f"Predicted Price: ₹{prediction:,.2f}", className="text-2xl font-semibold"),
                    html.P(f"Confidence Score: {conf_score:.2f}", className="text-green-600")
                ])
                return prediction_output, history, generate_history_table(history), current_time, ""
            except Exception as e:
                return [html.Div(f"Error in prediction: {str(e)}", className="text-red-600")], history, generate_history_table(history), current_time, ""
        
        return [""], history, generate_history_table(history), current_time, ""

    # Function to generate history table
    def generate_history_table(history):
        if not history:
            return html.P("No prediction history yet.", className="italic")
        
        column_mapping = {
            'Timestamp': 'timestamp',
            'Model': 'model',
            'Airline': 'airline',
            'Flight': 'flight',
            'Source City': 'source_city',
            'Dep. Time': 'departure_time',
            'Stops': 'stops',
            'Arr. Time': 'arrival_time',
            'Dest. City': 'destination_city',
            'Class': 'class',
            'Duration': 'duration',
            'Days Left': 'days_left',
            'Prediction': 'prediction',
            'Confidence': 'confidence'
        }
        
        return html.Table([
            html.Thead(
                html.Tr([
                    html.Th(col, className="px-4 py-2 border bg-blue-100 text-left text-sm font-medium text-blue-700")
                    for col in column_mapping.keys()
                ])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(f"{entry.get(column_mapping[col], ''):.2f}" if col in ['Duration', 'Prediction', 'Confidence'] else entry.get(column_mapping[col], ''),
                            className="px-4 py-2 border text-sm")
                    for col in column_mapping.keys()
                ])
                for entry in history
            ])
        ], className="min-w-full divide-y divide-blue-200 mb-30")

    # Callback for metrics chart
    @app.callback(
        Output('metrics-chart', 'figure'),
        Input('predict-button', 'n_clicks')
    )
    def update_metrics_chart(n_clicks):
        if results_df.empty:
            return {
                'data': [],
                'layout': {
                    'title': {'text': 'Model RMSE Comparison', 'x': 0.5, 'xanchor': 'center', 'font': {'color': "#000000", 'size': 20}},
                    'xaxis': {'title': 'Model'},
                    'yaxis': {'title': 'RMSE'},
                    'annotations': [{
                        'text': 'No model data available',
                        'xref': 'paper',
                        'yref': 'paper',
                        'showarrow': False,
                        'font': {'size': 16, 'color': '#dc2626'}
                    }]
                }
            }
        
        fig = go.Figure(data=[
            go.Bar(
                name='RMSE',
                x=results_df['Model'] + ' (' + results_df['Feature Selection'] + ')',
                y=results_df['RMSE'],
                marker_color='#3b82f6',
                marker_line_color='#1e40af',
                marker_line_width=1.5,
                opacity=0.8
            )
        ])
        fig.update_layout(
            title={'text': 'Model RMSE Comparison', 'x': 0.5, 'xanchor': 'center', 'font': {'color': "#000000", 'size': 14}},
            xaxis_title='Model',
            yaxis_title='RMSE',
            xaxis_tickangle=-45,
            xaxis_title_font={'color': "#030303", 'size': 14},
            yaxis_title_font={'color': "#000000", 'size': 14},
            xaxis_tickfont={'color': '#1e40af', 'size': 12},
            yaxis_tickfont={'color': '#1e40af', 'size': 12},
            plot_bgcolor='rgba(219, 234, 254, 0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin={'t': 50, 'b': 100, 'l': 50, 'r': 50},
            showlegend=False
        )
        return fig

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)