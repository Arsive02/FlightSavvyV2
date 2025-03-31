# Set matplotlib backend to non-interactive Agg
# This must be done before importing matplotlib.pyplot
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which doesn't require a GUI

import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import io
import base64
import os
import random
from dateutil.relativedelta import relativedelta
import calendar
import json

# Define month names for reference
months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']

# Complete carrier categorization
CARRIER_CATEGORIES = {
    # Premium/Legacy Carriers (15-30% more expensive)
    'PREMIUM': {
        'AA': {'name': 'American Airlines', 'factor': 1.20},
        'DL': {'name': 'Delta Air Lines', 'factor': 1.25},
        'UA': {'name': 'United Airlines', 'factor': 1.18},
        'AS': {'name': 'Alaska Airlines', 'factor': 1.15},
        'US': {'name': 'US Airways (merged with AA)', 'factor': 1.17},
        'CO': {'name': 'Continental (merged with UA)', 'factor': 1.18},
        'NW': {'name': 'Northwest (merged with DL)', 'factor': 1.20},
        'TW': {'name': 'Trans World Airlines', 'factor': 1.15},
        'PA': {'name': 'Pan Am', 'factor': 1.20},
    },
    
    # Mid-tier Carriers (base price to 10% more)
    'MID_TIER': {
        'B6': {'name': 'JetBlue Airways', 'factor': 1.08},
        'WN': {'name': 'Southwest Airlines', 'factor': 1.00},
        'SY': {'name': 'Sun Country Airlines', 'factor': 1.03},
        'FL': {'name': 'AirTran Airways', 'factor': 1.02},
        'VX': {'name': 'Virgin America', 'factor': 1.10},
        'HP': {'name': 'America West Airlines', 'factor': 1.05},
        'AQ': {'name': 'Aloha Airlines', 'factor': 1.05},
        'QX': {'name': 'Horizon Air', 'factor': 1.05},
    },
    
    # Budget Carriers (15-30% less expensive)
    'BUDGET': {
        'NK': {'name': 'Spirit Airlines', 'factor': 0.75},
        'F9': {'name': 'Frontier Airlines', 'factor': 0.70},
        'G4': {'name': 'Allegiant Air', 'factor': 0.80},
        'HQ': {'name': 'Harmony Airways', 'factor': 0.85},
        'JI': {'name': 'Midway Airlines', 'factor': 0.85},
        'TZ': {'name': 'ATA Airlines', 'factor': 0.80},
        'WV': {'name': 'Air South', 'factor': 0.75},
        'BF': {'name': 'Markair', 'factor': 0.80},
        'SX': {'name': 'Skybus Airlines', 'factor': 0.65},
    },
    
    # Regional Carriers (5-15% less expensive)
    'REGIONAL': {
        'OO': {'name': 'SkyWest Airlines', 'factor': 0.90},
        'YX': {'name': 'Republic Airways', 'factor': 0.90},
        'YV': {'name': 'Mesa Airlines', 'factor': 0.92},
        'DH': {'name': 'Independence Air', 'factor': 0.88},
        'OH': {'name': 'PSA Airlines', 'factor': 0.90},
        'ZW': {'name': 'Air Wisconsin', 'factor': 0.90},
        'KS': {'name': 'Peninsula Airways', 'factor': 0.88},
        '9K': {'name': 'Cape Air', 'factor': 0.85},
        'XJ': {'name': 'Mesaba Airlines', 'factor': 0.88},
        'RP': {'name': 'Chautauqua Airlines', 'factor': 0.90},
        'P9': {'name': 'Colgan Air', 'factor': 0.90},
        'ZV': {'name': 'Air Midwest', 'factor': 0.88},
    },
    
    # International Carriers
    'INTERNATIONAL': {
        '3M': {'name': 'LATAM Airlines (formerly LAN)', 'factor': 1.22},
        'MX': {'name': 'Mexicana Airlines', 'factor': 1.10},
        'XP': {'name': 'XpressAir', 'factor': 1.05},
        '5J': {'name': 'Cebu Pacific', 'factor': 0.90},
        'UK': {'name': 'Vistara', 'factor': 1.15},
        'KW': {'name': 'Korea Express Air', 'factor': 1.20},
        'KP': {'name': 'ASKY Airlines', 'factor': 1.10},
    },
    
    # Miscellaneous/Charter/Smaller Carriers
    'OTHER': {
        'RU': {'name': 'AirBridgeCargo', 'factor': 1.00},
        'J7': {'name': 'ValueJet', 'factor': 0.85},
        'U5': {'name': 'USA 3000 Airlines', 'factor': 0.90},
        'N7': {'name': 'National Airlines', 'factor': 1.00},
        'NJ': {'name': 'Visionair', 'factor': 0.95},
        'QQ': {'name': 'Reno Air', 'factor': 0.95},
        'W7': {'name': 'Western Pacific Airlines', 'factor': 0.93},
        'FF': {'name': 'Tower Air', 'factor': 0.90},
        'TB': {'name': 'USAir Shuttle', 'factor': 1.10},
        'LC': {'name': 'Logging Air', 'factor': 1.05},
        'YY': {'name': 'American Connection', 'factor': 0.95},
        'KN': {'name': 'China United Airlines', 'factor': 1.10},
        'E9': {'name': 'Evelop Airlines', 'factor': 1.05},
        'PN': {'name': 'Pan American Airways', 'factor': 1.10},
        '9N': {'name': 'Northern Thunderbird Air', 'factor': 1.00},
        'U2': {'name': 'easyJet', 'factor': 0.85},
        'OE': {'name': 'Asia Overnight Express', 'factor': 1.05},
        'W9': {'name': 'Eastwind Airlines', 'factor': 0.90},
        'RL': {'name': 'Royal Airlines', 'factor': 1.10},
        'T3': {'name': 'Eastern Airways', 'factor': 1.00},
        'OP': {'name': 'Chalk\'s Ocean Airways', 'factor': 1.10},
        'ZA': {'name': 'Access Air', 'factor': 0.95},
    }
}

# Base prices for popular routes
BASE_ROUTE_PRICES = {
    'ABQ-AUS': 95.00,  # Albuquerque to Austin
    'LAX-JFK': 250.00,  # Los Angeles to New York
    'ORD-DFW': 140.00,  # Chicago to Dallas
    'ATL-LAS': 175.00,  # Atlanta to Las Vegas
    'SFO-SEA': 120.00,  # San Francisco to Seattle
    'DFW-LAX': 150.00,  # Dallas to Los Angeles
    'DEN-PHX': 110.00,  # Denver to Phoenix
    'MIA-JFK': 130.00,  # Miami to New York
    'BOS-ORD': 120.00,  # Boston to Chicago
    'SEA-LAS': 95.00,   # Seattle to Las Vegas
}

# Function to get carrier information and pricing factor
def get_carrier_info(carrier_code):
    """
    Return the carrier info including name and pricing factor.
    If carrier not found, returns default values.
    """
    if not carrier_code or carrier_code == 'nan':
        return {'name': 'Unknown', 'factor': 1.0}
        
    for category, carriers in CARRIER_CATEGORIES.items():
        if carrier_code in carriers:
            return {
                'name': carriers[carrier_code]['name'],
                'factor': carriers[carrier_code]['factor'],
                'category': category
            }
    
    # If carrier not found in any category
    return {'name': f'Carrier {carrier_code}', 'factor': 1.0, 'category': 'UNKNOWN'}

import numpy as np

# convert_numpy_types function to explicitly handle bool_ types
def convert_numpy_types(obj):
    """
    Convert numpy types to native Python types for JSON serialization
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):  # Add explicit handling for NumPy boolean type
        return bool(obj)
    elif isinstance(obj, datetime.date):
        return obj.isoformat()
    elif isinstance(obj, (dict, pd.Series)):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Function to adjust fare based on carrier and route
def adjust_fare_by_carrier(fare, carrier_code, route=None):
    """
    Adjust fare based on carrier and optionally the specific route.
    """
    # Get carrier info with pricing factor
    carrier_info = get_carrier_info(carrier_code)
    carrier_factor = carrier_info['factor']
    
    # Apply route-specific adjustments if available
    route_factor = 1.0
    if route and route in BASE_ROUTE_PRICES:
        # Some carriers may have special pricing on specific routes
        if carrier_code == 'WN' and route in ['DAL-HOU', 'LAS-PHX', 'ABQ-AUS']:
            route_factor = 0.90  # Southwest cheaper on their hub routes
        elif carrier_code == 'DL' and route in ['ATL-JFK', 'DTW-MSP']:
            route_factor = 0.95  # Delta cheaper on their hub routes
        elif carrier_code == 'F9' and route in ['DEN-LAS', 'DEN-PHX']:
            route_factor = 0.85  # Frontier cheaper from Denver
    
    # Apply small random variation (±5%)
    variation_factor = random.uniform(0.95, 1.05)
    
    # Calculate final adjusted fare with carrier, route, and variation factors
    adjusted_fare = fare * carrier_factor * route_factor * variation_factor
    
    return round(adjusted_fare, 2)

def predict_best_time_to_buy_ticket(origin, destination, granularity="quarter", 
                                   future_year=None, filepath=None, 
                                   weeks_ahead=None, start_month=None, end_month=None,
                                   carrier=None):
    """
    Load trained models and predict the best time to buy a ticket
    
    Parameters:
    -----------
    origin : str
        Origin airport code (e.g., 'ABQ')
    destination : str
        Destination airport code (e.g., 'PHX')
    granularity : str, optional
        Prediction granularity: "date", "week", "month", or "quarter"
    future_year : int, optional
        Year to predict for (defaults to current year)
    filepath : str, optional
        Path to sample data for feature extraction
    weeks_ahead : int, optional
        If predicting for specific dates, how many weeks ahead to predict
    start_month : int, optional
        Start month of travel period (1-12)
    end_month : int, optional
        End month of travel period (1-12)
    carrier : str, optional
        Airline/carrier code to filter results (e.g., 'AA' for American Airlines)
        
    Returns:
    --------
    dict
        Contains best_time, predictions, and chart data for visualization
    """
    try:
        # Ensure airport codes are uppercase
        origin = origin.upper()
        destination = destination.upper()
        route_name = f"{origin}-{destination}"
        
        # Set default year if not provided
        if future_year is None:
            future_year = datetime.datetime.now().year
        else:
            # Convert future_year to integer if it's a string
            future_year = int(future_year)
            
        print(f"Predicting best time to buy for route: {route_name} with granularity: {granularity}")
        if start_month and end_month:
            print(f"Travel period: Months {start_month} to {end_month}")
        
        # Get carrier information if provided
        carrier_info = None
        if carrier:
            carrier_info = get_carrier_info(carrier)
            print(f"Filtering for carrier: {carrier} ({carrier_info['name']})")
            print(f"Carrier pricing factor: {carrier_info['factor']}")
        
        # Load the trained models
        try:
            rf_model = joblib.load('flight_fare_rf_model.joblib')
            print("Random Forest model loaded successfully")
        except Exception as e:
            print(f"Error loading Random Forest model: {e}")
            return {"error": "Failed to load Random Forest model"}
        
        # Try to load time series model (optional)
        try:
            ts_model = joblib.load('flight_fare_ts_model.joblib')
            print("Time series model loaded successfully")
        except Exception as e:
            print(f"Time series model not available: {str(e)}")
            ts_model = None
        
        # Load sample data for feature extraction
        if filepath is None:
            filepath = 'data/US Airline Flight Routes and Fares 1993-2024.csv'
        
        print(f"Loading data from {filepath}")
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading data file: {e}")
            # Create a dummy dataframe for demonstration
            df = pd.DataFrame({
                'airport_1': ['DFW', 'LAX', 'ATL'],
                'airport_2': ['LAX', 'JFK', 'MIA'],
                'route': ['DFW-LAX', 'LAX-JFK', 'ATL-MIA'],
                'Year': [2024, 2024, 2024],
                'quarter': [1, 2, 3],
                'fare': [250, 350, 300],
                'nsmiles': [1200, 2500, 600],
                'passengers': [300, 400, 200],
                'carrier_lg': ['AA', 'DL', 'WN'],
                'large_ms': [0.8, 0.7, 0.6],
                'fare_lg': [280, 380, 320],
                'carrier_low': ['WN', 'UA', 'NK'],
                'lf_ms': [0.2, 0.3, 0.4],
                'fare_low': [200, 300, 250]
            })
        
        # Ensure 'route' column exists
        if 'route' not in df.columns:
            print("Creating 'route' column from airport codes")
            df['route'] = df['airport_1'] + '-' + df['airport_2']
        
        # Filter data for the specific route or similar routes
        route_data = df[df['route'] == route_name].copy()
        
        # If no exact route match, find similar routes or use average values
        if route_data.empty:
            print(f"No data for exact route {route_name}, using similar routes or average values")
            # Try to find routes with same origin
            origin_routes = df[df['airport_1'] == origin]
            if not origin_routes.empty:
                route_data = origin_routes.iloc[0:1].copy()
                print(f"Using data from route with same origin: {route_data['route'].values[0]}")
            else:
                # Use an average of all routes
                route_data = df.iloc[0:1].copy()
                print("Using average route data")
            
            # Update the route information
            route_data['airport_1'] = origin
            route_data['airport_2'] = destination
            route_data['route'] = route_name
            
            # If we have a base price for this route, use it
            if route_name in BASE_ROUTE_PRICES:
                print(f"Using base price for route {route_name}: ${BASE_ROUTE_PRICES[route_name]}")
                route_data['fare'] = BASE_ROUTE_PRICES[route_name]
            else:
                # Estimate price based on distance if available
                if 'nsmiles' in route_data.columns:
                    # Simple linear model: longer routes cost more
                    estimated_fare = route_data['nsmiles'].values[0] * 0.15  # $0.15 per mile as base
                    print(f"Estimating fare based on distance: ${estimated_fare:.2f}")
                    route_data['fare'] = estimated_fare
        
        # Apply carrier-specific price adjustments if a carrier is specified
        if carrier and carrier_info:
            # Start with the base fare
            base_fare = route_data['fare'].values[0]
            
            # Apply detailed carrier-specific pricing
            adjusted_fare = adjust_fare_by_carrier(base_fare, carrier, route_name)
            
            print(f"Adjusting fare for {carrier} ({carrier_info['name']}): ${base_fare:.2f} → ${adjusted_fare:.2f}")
            
            # Update the route data with the carrier-adjusted fare
            route_data['fare'] = adjusted_fare
            
            # Store carrier information in route data
            route_data['carrier'] = carrier
            route_data['carrier_name'] = carrier_info['name']
            route_data['carrier_category'] = carrier_info.get('category', 'UNKNOWN')
            
            # If carrier matches either known carrier in the data, use that carrier's fare
            if 'carrier_lg' in route_data.columns and route_data['carrier_lg'].values[0] == carrier:
                if 'fare_lg' in route_data.columns:
                    print(f"Using {carrier} as the major carrier with fare: ${route_data['fare_lg'].values[0]:.2f}")
                    route_data['fare'] = route_data['fare_lg']
            elif 'carrier_low' in route_data.columns and route_data['carrier_low'].values[0] == carrier:
                if 'fare_low' in route_data.columns:
                    print(f"Using {carrier} as the low-fare carrier with fare: ${route_data['fare_low'].values[0]:.2f}")
                    route_data['fare'] = route_data['fare_low']
        
        # Engineer required features
        print("Engineering required features")
        
        # Calculate price_per_mile
        if 'nsmiles' in route_data.columns and 'fare' in route_data.columns:
            route_data['price_per_mile'] = route_data['fare'] / route_data['nsmiles']
        else:
            route_data['price_per_mile'] = 0.25  # Default average price per mile
        
        # Calculate market_concentration (using large_ms or default)
        if 'large_ms' in route_data.columns and 'lf_ms' in route_data.columns:
            route_data['market_concentration'] = np.maximum(
                route_data['large_ms'], route_data['lf_ms'])
        else:
            route_data['market_concentration'] = 0.8  # Default high concentration
        
        # Calculate price_difference (between large and low fare carriers)
        if 'fare_lg' in route_data.columns and 'fare_low' in route_data.columns:
            route_data['price_difference'] = route_data['fare_lg'] - route_data['fare_low']
        else:
            route_data['price_difference'] = 20.0  # Default difference
        
        # Calculate route_competition
        if 'carrier_lg' in route_data.columns:
            # If we had access to all data, we'd group by route
            route_data['route_competition'] = 2  # Default: assume 2 carriers
        else:
            route_data['route_competition'] = 2  # Default competition value
        
        # Create season if missing
        if 'season' not in route_data.columns:
            print("Adding season column")
            seasons = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
            route_data['quarter'] = route_data['quarter'].astype(int)
            route_data['season'] = route_data['quarter'].map(seasons)
        
        # Check essential columns for prediction
        required_columns = ['Year', 'quarter', 'nsmiles', 'passengers']
        for col in required_columns:
            if col not in route_data.columns:
                # Add defaults if missing
                if col == 'nsmiles':
                    route_data['nsmiles'] = 800  # Default distance
                elif col == 'passengers':
                    route_data['passengers'] = 250  # Default passenger count
        
        # Generate prediction dates based on granularity
        prediction_dates = []
        
        if granularity == "date":
            # If predicting for specific dates, generate dates based on weeks ahead
            if weeks_ahead is None:
                weeks_ahead = 12  # Default to 12 weeks (about 3 months) ahead
            
            # Generate dates from today to X weeks ahead
            start_date = datetime.datetime.now().date()
            for i in range(weeks_ahead * 7):
                prediction_dates.append(start_date + datetime.timedelta(days=i))
                
            # If start_month and end_month are provided, filter dates to be within that window
            if start_month is not None and end_month is not None:
                filtered_dates = []
                
                # Helper function to check if a date is within the travel period
                def is_in_travel_period(date):
                    month = date.month
                    if start_month <= end_month:
                        return start_month <= month <= end_month
                    else:  # Wrap around case (e.g., November to February)
                        return month >= start_month or month <= end_month
                
                for date in prediction_dates:
                    if is_in_travel_period(date):
                        filtered_dates.append(date)
                
                # If we have filtered dates, use those
                if filtered_dates:
                    prediction_dates = filtered_dates
        
        elif granularity == "week":
            # Generate weekly dates for the year
            start_date = datetime.datetime(future_year, 1, 1)
            # Find the first Monday
            while start_date.weekday() != 0:  # 0 = Monday
                start_date += datetime.timedelta(days=1)
            
            # Generate dates for each Monday of the year
            current_date = start_date
            while current_date.year == future_year:
                prediction_dates.append(current_date.date())
                current_date += datetime.timedelta(days=7)
                
            # If start_month and end_month are provided, filter weeks to be within that window
            if start_month is not None and end_month is not None:
                filtered_dates = []
                
                # Helper function to check if a date is within the travel period
                def is_in_travel_period(date):
                    month = date.month
                    if start_month <= end_month:
                        return start_month <= month <= end_month
                    else:  # Wrap around case (e.g., November to February)
                        return month >= start_month or month <= end_month
                
                for date in prediction_dates:
                    if is_in_travel_period(date):
                        filtered_dates.append(date)
                
                # If we have filtered dates, use those
                if filtered_dates:
                    prediction_dates = filtered_dates
        
        elif granularity == "month":
            # Generate monthly dates (1st of each month)
            for month in range(1, 13):
                prediction_dates.append(datetime.datetime(future_year, month, 1).date())
        
        elif granularity == "quarter":
            # Generate quarterly dates (middle of each quarter)
            quarter_months = [2, 5, 8, 11]  # February, May, August, November
            for month in quarter_months:
                prediction_dates.append(datetime.datetime(future_year, month, 15).date())
        
        # Predictions for each date
        predictions = []
        
        # For each date, predict the fare
        for pred_date in prediction_dates:
            print(f"Generating prediction for {pred_date}")
            
            # Create a sample based on route average values
            sample_data = route_data.iloc[0].copy()
            
            # Set date-related features
            sample_data['Year'] = pred_date.year
            sample_data['month'] = pred_date.month
            sample_data['day_of_year'] = pred_date.timetuple().tm_yday
            
            # Determine quarter
            quarter = (pred_date.month - 1) // 3 + 1
            sample_data['quarter'] = quarter
            
            # Set week number (1-52)
            week_number = pred_date.isocalendar()[1]
            sample_data['week'] = week_number
            
            # Set holiday period flag
            # Major holidays: New Year's, Thanksgiving, Christmas, 4th of July, etc.
            major_holidays = [
                (1, 1),    # New Year's
                (12, 25),  # Christmas
                (11, [20, 21, 22, 23, 24, 25, 26, 27, 28]),  # Thanksgiving range
                (7, 4),    # 4th of July
                (5, [25, 26, 27, 28, 29, 30, 31]),  # Memorial Day range
                (9, [1, 2, 3, 4, 5, 6, 7]),  # Labor Day range
            ]
            
            is_holiday = False
            for month, days in major_holidays:
                if pred_date.month == month:
                    if isinstance(days, list):
                        if pred_date.day in days:
                            is_holiday = True
                            break
                    elif pred_date.day == days:
                        is_holiday = True
                        break
            
            # Check for holiday periods (2 weeks around major holidays)
            if not is_holiday:
                for month, days in major_holidays:
                    if isinstance(days, list):
                        holiday_date = datetime.datetime(pred_date.year, month, days[0])
                    else:
                        holiday_date = datetime.datetime(pred_date.year, month, days)
                    
                    delta = abs((pred_date - holiday_date.date()).days)
                    if delta <= 14:  # Within 2 weeks
                        is_holiday = True
                        break
            
            sample_data['is_holiday_period'] = is_holiday
            
            # Set season based on month
            seasons_by_month = {
                1: 'Winter', 2: 'Winter', 3: 'Spring', 
                4: 'Spring', 5: 'Spring', 6: 'Summer',
                7: 'Summer', 8: 'Summer', 9: 'Fall', 
                10: 'Fall', 11: 'Fall', 12: 'Winter'
            }
            sample_data['season'] = seasons_by_month[pred_date.month]
            
            # Make sure carrier is set
            if carrier:
                sample_data['carrier'] = carrier
                # Keep carrier info for later reference
                if carrier_info:
                    sample_data['carrier_name'] = carrier_info['name']
                    sample_data['carrier_category'] = carrier_info.get('category', 'UNKNOWN')
            
            # Random Forest prediction
            try:
                # Create feature vector
                sample_X = pd.DataFrame([sample_data])
                
                # Predict fare
                rf_predicted_fare = rf_model.predict(sample_X)[0]
                print(f"RF prediction: ${rf_predicted_fare:.2f}")
            except Exception as e:
                print(f"Error making Random Forest prediction: {str(e)}")
                # Use the base fare if available, otherwise use a reasonable default
                rf_predicted_fare = sample_data.get('fare', 180.0)
                print(f"Using fallback fare: ${rf_predicted_fare:.2f}")
            
            # Time Series prediction (if available)
            ts_predicted_fare = None
            combined_prediction = rf_predicted_fare
            
            if ts_model is not None:
                try:
                    # For time series, use the quarter or month index
                    if granularity == "quarter":
                        ts_idx = quarter - 1
                    elif granularity == "month":
                        ts_idx = pred_date.month - 1
                    elif granularity == "week":
                        ts_idx = min(week_number - 1, 51)  # Max 52 weeks
                    else:
                        # For daily, use the day of year (scaled)
                        days_in_year = 366 if calendar.isleap(pred_date.year) else 365
                        ts_idx = int((sample_data['day_of_year'] / days_in_year) * 4)  # Scale to 0-3
                    
                    # Get forecast for the appropriate period
                    max_steps = 52 if granularity == "week" else 12 if granularity == "month" else 4
                    forecasts = ts_model.forecast(steps=max_steps)
                    ts_predicted_fare = forecasts[min(ts_idx, len(forecasts)-1)]
                    print(f"TS prediction: ${ts_predicted_fare:.2f}")
                    
                    # Combine predictions (weighted average)
                    combined_prediction = 0.7 * rf_predicted_fare + 0.3 * ts_predicted_fare
                    print(f"Combined prediction: ${combined_prediction:.2f}")
                except Exception as e:
                    print(f"Error making time series prediction: {str(e)}")
            
            # Apply holiday effect
            holiday_markup = 1.15 if is_holiday else 1.0  # 15% markup for holiday periods
            
            # Apply seasonality effect based on month
            seasonal_factors = {
                1: 1.05,  # January (post-holiday)
                2: 1.0,   # February (low season)
                3: 1.02,  # March (spring break)
                4: 1.05,  # April (Easter)
                5: 1.02,  # May 
                6: 1.12,  # June (summer peak)
                7: 1.15,  # July (summer peak)
                8: 1.1,   # August (summer end)
                9: 0.95,  # September (low season)
                10: 0.98, # October (low season)
                11: 1.1,  # November (Thanksgiving)
                12: 1.2   # December (Christmas)
            }
            
            seasonal_factor = seasonal_factors[pred_date.month]
            
            # Apply day of week effect
            dow_factors = {
                0: 0.98,  # Monday
                1: 0.97,  # Tuesday
                2: 0.97,  # Wednesday
                3: 1.02,  # Thursday
                4: 1.05,  # Friday
                5: 1.02,  # Saturday
                6: 0.99   # Sunday
            }
            
            dow_factor = dow_factors[pred_date.weekday()]
            
            # Final prediction with all effects
            final_prediction = combined_prediction * holiday_markup * seasonal_factor * dow_factor
            print(f"Final prediction with all effects: ${final_prediction:.2f}")
            
            # If we have a specific carrier, ensure the carrier effect is reflected
            if carrier and carrier_info:
                # Apply specific carrier factor to the final prediction
                carrier_adjusted_prediction = adjust_fare_by_carrier(final_prediction, carrier, route_name)
                
                # Only apply full adjustment if it wasn't already included in the base fare
                # If the difference is large, use the carrier-adjusted prediction
                # Otherwise, use the existing prediction (carrier effect already included)
                carrier_effect_ratio = carrier_adjusted_prediction / final_prediction
                if abs(carrier_effect_ratio - 1.0) > 0.05:  # If more than 5% difference
                    print(f"Applying carrier effect: ${final_prediction:.2f} → ${carrier_adjusted_prediction:.2f}")
                    final_prediction = carrier_adjusted_prediction
            
            # Convert all numpy types to Python types
            prediction_entry = {
                'date': pred_date,
                'predicted_fare': float(final_prediction),
                'rf_prediction': float(rf_predicted_fare) if rf_predicted_fare is not None else None,
                'ts_prediction': float(ts_predicted_fare) if ts_predicted_fare is not None else None,
                'is_holiday_period': bool(is_holiday),
                'year': int(pred_date.year),
                'month': int(pred_date.month),
                'month_name': pred_date.strftime('%B'),
                'quarter': int(quarter),
                'week': int(week_number),
                'day_of_week': pred_date.strftime('%A'),
                'carrier': carrier if carrier else None,
                'carrier_name': carrier_info['name'] if carrier_info else None
            }
            
            predictions.append(prediction_entry)
        
        # Convert to DataFrame for easier manipulation
        predictions_df = pd.DataFrame(predictions)
        
        # Filter predictions by travel period if specified
        filtered_predictions_df = predictions_df.copy()
        travel_period_filtered = False
        
        if start_month is not None and end_month is not None and granularity in ["month", "quarter"]:
            travel_period_filtered = True
            
            # Helper function to check if a month is within travel period
            def is_in_travel_period(month):
                if start_month <= end_month:
                    return start_month <= month <= end_month
                else:  # Wrap around case (e.g., November to February)
                    return month >= start_month or month <= end_month
            
            # Filter predictions by travel period
            if granularity == "month":
                filtered_predictions_df = predictions_df[predictions_df['month'].apply(is_in_travel_period)]
            elif granularity == "quarter":
                # Map months to quarters
                start_quarter = (start_month - 1) // 3 + 1
                end_quarter = (end_month - 1) // 3 + 1
                
                # Helper function to check if quarter is in travel period
                def is_quarter_in_travel_period(quarter):
                    if start_quarter <= end_quarter:
                        return start_quarter <= quarter <= end_quarter
                    else:  # Wrap around case
                        return quarter >= start_quarter or quarter <= end_quarter
                
                filtered_predictions_df = predictions_df[predictions_df['quarter'].apply(is_quarter_in_travel_period)]
        
        # If we filtered and have results, use the filtered data for best time determination
        if travel_period_filtered and not filtered_predictions_df.empty:
            active_df = filtered_predictions_df
        else:
            active_df = predictions_df
        
        # Find the best time at the specified granularity
        if granularity == "date":
            best_idx = active_df['predicted_fare'].idxmin()
            best_time_row = active_df.loc[best_idx]
            best_time = {k: convert_numpy_types(v) for k, v in best_time_row.to_dict().items()}
            
            # Check if date is string and convert if needed
            if isinstance(best_time['date'], str):
                date_obj = datetime.datetime.strptime(best_time['date'], '%Y-%m-%d').date()
                formatted_best = f"{date_obj.strftime('%A, %B %d, %Y')}"
            else:
                formatted_best = f"{best_time['date'].strftime('%A, %B %d, %Y')}"
            
        elif granularity == "week":
            # Group by week
            weekly_avg = active_df.groupby('week')['predicted_fare'].mean().reset_index()
            best_week = int(weekly_avg.loc[weekly_avg['predicted_fare'].idxmin()]['week'])
            best_week_data = active_df[active_df['week'] == best_week].iloc[0]
            best_time = {k: convert_numpy_types(v) for k, v in best_week_data.to_dict().items()}
            
            # Get date range for the best week
            # Convert date to datetime object if it's a string
            best_date = best_time['date']
            if isinstance(best_date, str):
                best_date = datetime.datetime.strptime(best_date, '%Y-%m-%d').date()
            start_of_week = best_date - datetime.timedelta(days=best_date.weekday())
            end_of_week = start_of_week + datetime.timedelta(days=6)
            formatted_best = f"Week {best_week} ({start_of_week.strftime('%b %d')} - {end_of_week.strftime('%b %d')})"
            
        elif granularity == "month":
            # Group by month
            monthly_avg = active_df.groupby(['month', 'month_name'])['predicted_fare'].mean().reset_index()
            best_month_idx = monthly_avg['predicted_fare'].idxmin()
            best_month = int(monthly_avg.loc[best_month_idx]['month'])
            best_month_name = monthly_avg.loc[best_month_idx]['month_name']
            best_month_fare = float(monthly_avg.loc[best_month_idx]['predicted_fare'])
            
            best_time = {
                'month': best_month,
                'month_name': best_month_name,
                'predicted_fare': best_month_fare,
                'carrier': carrier,
                'carrier_name': carrier_info['name'] if carrier_info else None
            }
            formatted_best = f"{best_month_name}"
            
        elif granularity == "quarter":
            # Group by quarter
            quarterly_avg = active_df.groupby('quarter')['predicted_fare'].mean().reset_index()
            best_quarter_idx = quarterly_avg['predicted_fare'].idxmin()
            best_quarter = int(quarterly_avg.loc[best_quarter_idx]['quarter'])
            best_quarter_fare = float(quarterly_avg.loc[best_quarter_idx]['predicted_fare'])
            
            best_time = {
                'quarter': best_quarter,
                'predicted_fare': best_quarter_fare,
                'carrier': carrier,
                'carrier_name': carrier_info['name'] if carrier_info else None
            }
            formatted_best = f"Q{best_quarter}"
        
        # For visualization, use the filtered data if available
        viz_df = filtered_predictions_df if travel_period_filtered and not filtered_predictions_df.empty else predictions_df
        
        # Instead of generating images, prepare chart data for React
        chart_data = {}

        if granularity == "date":
            # Prepare daily chart data
            chart_data = {
                'type': 'line',
                'data': [
                    {
                        'date': pred['date'].isoformat() if not isinstance(pred['date'], str) else pred['date'],
                        'fare': round(pred['predicted_fare'], 2),
                        'isHoliday': pred['is_holiday_period'],
                        'isBest': False  # Will mark the best date below
                    }
                    for pred in viz_df.to_dict('records')
                ],
                'xAxisKey': 'date',
                'yAxisKey': 'fare',
                'xAxisLabel': 'Date',
                'yAxisLabel': 'Predicted Fare ($)',
                'title': f'Predicted Fares for {route_name}'
            }
            
            # Add carrier info to title if available
            if carrier and carrier_info:
                chart_data['title'] += f' with {carrier} ({carrier_info["name"]})'
            
            # Add travel period info to title if filtered
            if travel_period_filtered:
                chart_data['title'] += f' (Travel Period: {months[start_month-1]} to {months[end_month-1]})'
            
            # Mark the best date
            best_idx = viz_df['predicted_fare'].idxmin()
            best_date = viz_df.loc[best_idx, 'date']
            best_date_str = best_date.isoformat() if not isinstance(best_date, str) else best_date
            
            # Find the best date in our chart data and mark it
            for point in chart_data['data']:
                if point['date'] == best_date_str:
                    point['isBest'] = True
                    break

        elif granularity in ["week", "month", "quarter"]:
            # Prepare bar chart data
            if granularity == "week":
                # Group by week
                grouped_data = viz_df.groupby('week')['predicted_fare'].mean().reset_index()
                label_key = 'week'
                label_formatter = lambda x: f"Week {int(x)}"
                
            elif granularity == "month":
                # Group by month
                grouped_data = viz_df.groupby(['month', 'month_name'])['predicted_fare'].mean().reset_index()
                grouped_data = grouped_data.sort_values('month')
                label_key = 'month_name'
                label_formatter = lambda x: x
                
            else:  # Quarter
                # Group by quarter
                grouped_data = viz_df.groupby('quarter')['predicted_fare'].mean().reset_index()
                label_key = 'quarter'
                label_formatter = lambda x: f"Q{int(x)}"
            
            # Find best time period
            best_idx = grouped_data['predicted_fare'].idxmin()
            best_value = grouped_data.loc[best_idx, label_key]
            
            # Create chart data
            chart_data = {
                'type': 'bar',
                'data': [
                    {
                        'label': label_formatter(row[label_key]),
                        'value': label_key,
                        'originalValue': row[label_key],
                        'fare': round(row['predicted_fare'], 2),
                        'isBest': row[label_key] == best_value
                    }
                    for _, row in grouped_data.iterrows()
                ],
                'xAxisKey': 'label',
                'yAxisKey': 'fare',
                'xAxisLabel': 'Time Period',
                'yAxisLabel': 'Predicted Fare ($)',
                'title': f'Predicted Fares for {route_name}'
            }
            
            # Add carrier info to title if available
            if carrier and carrier_info:
                chart_data['title'] += f' with {carrier} ({carrier_info["name"]})'
            
            # Add travel period info to title if filtered
            if travel_period_filtered:
                chart_data['title'] += f' (Travel Period: {months[start_month-1]} to {months[end_month-1]})'

        # Full analysis chart data
        full_analysis_chart_data = {}

        if travel_period_filtered and (granularity == "month" or granularity == "quarter"):
            if granularity == "month":
                # Group by month using ALL predictions
                full_grouped_data = predictions_df.groupby(['month', 'month_name'])['predicted_fare'].mean().reset_index()
                full_grouped_data = full_grouped_data.sort_values('month')
                label_key = 'month_name'
                label_formatter = lambda x: x
                
                # Function to check if a month is in travel period
                def is_in_travel_period(month):
                    if start_month <= end_month:
                        return start_month <= month <= end_month
                    else:  # Wrap around case (e.g., November to February)
                        return month >= start_month or month <= end_month
                
                # Find best month from filtered set
                best_month = int(monthly_avg.loc[monthly_avg['predicted_fare'].idxmin()]['month'])
                
            else:  # quarter
                # Group by quarter using ALL predictions
                full_grouped_data = predictions_df.groupby('quarter')['predicted_fare'].mean().reset_index()
                label_key = 'quarter'
                label_formatter = lambda x: f"Q{int(x)}"
                
                # Function to check if quarter is in travel period
                def is_quarter_in_travel_period(quarter):
                    start_quarter = (start_month - 1) // 3 + 1
                    end_quarter = (end_month - 1) // 3 + 1
                    if start_quarter <= end_quarter:
                        return start_quarter <= quarter <= end_quarter
                    else:  # Wrap around case
                        return quarter >= start_quarter or quarter <= end_quarter
                
                # Find best quarter from filtered set
                best_quarter = int(quarterly_avg.loc[quarterly_avg['predicted_fare'].idxmin()]['quarter'])
            
            # Create full analysis chart data
            full_analysis_chart_data = {
                'type': 'bar',
                'data': [],
                'xAxisKey': 'label',
                'yAxisKey': 'fare',
                'xAxisLabel': 'Time Period',
                'yAxisLabel': 'Predicted Fare ($)',
                'title': f'Full Year Price Analysis for {route_name}'
            }
            
            # Add carrier info to title if available
            if carrier and carrier_info:
                full_analysis_chart_data['title'] += f' with {carrier} ({carrier_info["name"]})'
            
            # Add travel period info to title
            full_analysis_chart_data['title'] += f' (Travel Period: {months[start_month-1]} to {months[end_month-1]})'
            
            # Build data points
            for _, row in full_grouped_data.iterrows():
                original_value = row[label_key]
                
                # Determine if in travel period and if best period
                if granularity == "month":
                    in_travel_period = is_in_travel_period(row['month'])
                    is_best = row['month'] == best_month
                else:  # quarter
                    in_travel_period = is_quarter_in_travel_period(row['quarter'])
                    is_best = row['quarter'] == best_quarter
                
                full_analysis_chart_data['data'].append({
                    'label': label_formatter(original_value),
                    'value': original_value,
                    'fare': round(row['predicted_fare'], 2),
                    'inTravelPeriod': in_travel_period,
                    'isBest': is_best
                })
        else:
            # If not filtered or not month/quarter, use the regular chart data
            full_analysis_chart_data = chart_data
        
        # Convert the predictions list to ensure all numpy types are converted
        filtered_predictions = []
        for pred in active_df.to_dict('records'):
            filtered_predictions.append({k: convert_numpy_types(v) for k, v in pred.items()})
        
        # All predictions (unfiltered) for comparison
        all_predictions = []
        for pred in predictions_df.to_dict('records'):
            all_predictions.append({k: convert_numpy_types(v) for k, v in pred.items()})
        
        # Create the result dictionary with all NumPy types converted
        result = {
            'route': route_name,
            'granularity': granularity,
            'carrier': carrier,
            'carrier_name': carrier_info['name'] if carrier_info else None,
            'best_time': best_time,
            'formatted_best_time': formatted_best,
            'filtered_predictions': filtered_predictions,
            'all_predictions': all_predictions,
            'chart_data': chart_data,
            'full_analysis_chart_data': full_analysis_chart_data,
            'travel_period': {
                'start_month': start_month,
                'end_month': end_month,
                'start_month_name': months[start_month-1] if start_month is not None else None,
                'end_month_name': months[end_month-1] if end_month is not None else None
            } if start_month and end_month else None,
            'travel_period_filtered': travel_period_filtered,
            'success': True
        }
        
        result = json.loads(json.dumps(result, default=lambda o: convert_numpy_types(o)))
        
        return result
        
    except Exception as e:
        import traceback
        print(f"Error in prediction: {e}")
        print(traceback.format_exc())
        return {
            'error': str(e),
            'success': False
        }

# Example usage
if __name__ == "__main__":
    # Show how to use each granularity option
    print("\n=== PREDICTING BY QUARTER ===")
    result_quarter = predict_best_time_to_buy_ticket('ABQ', 'PHX', granularity="quarter")
    
    print("\n=== PREDICTING BY MONTH ===")
    result_month = predict_best_time_to_buy_ticket('ABQ', 'PHX', granularity="month")
    
    print("\n=== PREDICTING BY MONTH WITH TRAVEL PERIOD ===")
    result_month_filtered = predict_best_time_to_buy_ticket('ABQ', 'PHX', granularity="month", start_month=4, end_month=8)
    
    print("\n=== PREDICTING BY WEEK ===")
    result_week = predict_best_time_to_buy_ticket('ABQ', 'PHX', granularity="week")
    
    print("\n=== PREDICTING BY DATE ===")
    result_date = predict_best_time_to_buy_ticket('ABQ', 'PHX', granularity="date", weeks_ahead=8)
    
    print("\n=== PREDICTING WITH SPECIFIC CARRIER ===")
    result_carrier = predict_best_time_to_buy_ticket('ABQ', 'PHX', granularity="month", carrier="WN")

    # Test predictions with various carriers
    carriers_to_test = ['AA', 'DL', 'WN', 'F9', 'NK', 'G4']
    for test_carrier in carriers_to_test:
        print(f"\n=== TESTING WITH CARRIER: {test_carrier} ===")
        result = predict_best_time_to_buy_ticket('ABQ', 'PHX', granularity="month", carrier=test_carrier)
        if result.get('success', False):
            print(f"Carrier: {test_carrier}")
            print(f"Predicted fare: ${result['best_time']['predicted_fare']:.2f}")