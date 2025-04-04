from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Import your prediction function
from prediction import predict_best_time_to_buy_ticket

@app.route('/')
def home():
    return {
        'message': 'Welcome to the FlightSavvy API!',
        'List of endpoints': {
            '/api/predict': 'Predict the best time to buy a flight ticket.',
            '/api/help': 'Get help on how to use the API.'
        }
    }


@app.route('/api/help')
def help():
    return {
        'message': 'This is the help endpoint.',
        'usage': {
            'POST /api/predict': {
                'description': 'Predict the best time to buy a flight ticket.',
                'parameters': {
                    'origin': 'Origin airport code (e.g., "JFK").',
                    'destination': 'Destination airport code (e.g., "LAX").',
                    'granularity': {
                        'description': 'Granularity of the prediction.',
                        'options': ['day', 'week', 'month', 'quarter'],
                        'default': 'quarter'
                    },
                    'futureYear': '(optional) Future year for prediction.',
                    'weeksAhead': '(optional) Number of weeks ahead for prediction.',
                    'start_month': '(optional) Start month for travel period.',
                    'end_month': '(optional) End month for travel period.',
                    'carrier': '(optional) Airline carrier code.'
                }
            }
        }
    }


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Required parameters
    origin = data.get('origin')
    destination = data.get('destination')
    
    # Optional parameters with defaults
    granularity = data.get('granularity', 'quarter')
    future_year = data.get('futureYear')
    weeks_ahead = data.get('weeksAhead')
    
    # Travel period parameters (month numbers 1-12)
    start_month = data.get('start_month')
    end_month = data.get('end_month')
    
    # NEW: Add carrier parameter
    carrier = data.get('carrier')
    
    # Convert month names to month numbers if provided
    if isinstance(start_month, str) and not start_month.isdigit():
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
        try:
            start_month = months.index(start_month) + 1
        except ValueError:
            pass
    
    if isinstance(end_month, str) and not end_month.isdigit():
        months = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']
        try:
            end_month = months.index(end_month) + 1
        except ValueError:
            pass
    
    # Call prediction function
    try:
        result = predict_best_time_to_buy_ticket(
            origin=origin,
            destination=destination,
            granularity=granularity,
            future_year=future_year,
            weeks_ahead=weeks_ahead,
            start_month=start_month,
            end_month=end_month,
            carrier=carrier  # NEW: Pass carrier to prediction function
        )
        
        return jsonify(result)
    except Exception as e:
        import traceback
        print(f"API Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)