# File: verify_prediction.py
import pandas as pd
from datetime import datetime
from meteostat import Stations, Daily
import time

def verify_perth_prediction():
    """
    Fetches the actual historical weather for a specific date in Perth
    and compares it to a model's prediction.
    """
    # --- 1. Define the Prediction Details ---
    prediction_date = datetime(2017, 6, 26)
    model_prediction = "No, it is not likely to rain." # This is what our model predicted
    
    print("--- Verifying Weather Prediction for Perth ---")
    print(f"Prediction Date: {prediction_date.strftime('%B %d, %Y')}")
    print(f"Model's Forecast: '{model_prediction}'")
    print("-" * 45)

    # --- 2. Find the Weather Station for Perth ---
    print("Finding the official weather station for Perth...")
    stations = Stations()
    # Let's find stations near Perth's coordinates (approx. 31.95° S, 115.86° E)
    perth_stations = stations.nearby(-31.95, 115.86)
    perth_station = perth_stations.fetch(1) # Get the closest station
    
    # Meteostat uses station IDs. The primary one for Perth is '94610'.
    station_id = '94610' 
    print(f"Using station ID: {station_id} ({perth_station.index[0]})")
    
    # --- 3. Fetch the Historical Weather Data ---
    print(f"Fetching actual weather data for {prediction_date.strftime('%Y-%m-%d')}...")
    
    # Meteostat requires a start and end date for the fetch
    start_date = prediction_date
    end_date = prediction_date
    
    # Fetch daily data for the station and date
    data = Daily(station_id, start_date, end_date)
    data = data.fetch()
    
    # A small delay to be polite to the free API
    time.sleep(1) 
    
    if data.empty:
        print("\nCould not retrieve weather data for this date.")
        return

    # --- 4. Analyze and Compare ---
    actual_rainfall = data.iloc[0]['prcp'] # 'prcp' is the column for precipitation in mm
    
    # Determine the actual outcome
    if actual_rainfall > 0:
        actual_outcome = "Yes, it did rain."
    else:
        actual_outcome = "No, it did not rain."
        
    # Check if the model's prediction was correct
    prediction_was_correct = (model_prediction.startswith("No") and actual_rainfall == 0) or \
                             (model_prediction.startswith("Yes") and actual_rainfall > 0)

    # --- 5. Display the Results ---
    print("\n--- Verification Result ---")
    print(f"Actual Rainfall Recorded: {actual_rainfall} mm")
    print(f"Actual Outcome: '{actual_outcome}'")
    print("-" * 45)
    
    if prediction_was_correct:
        print("✅ SUCCESS: The model's prediction was CORRECT.")
    else:
        print("❌ INCORRECT: The model's prediction was incorrect.")
    print("=" * 45)


# Run the verification process
if __name__ == "__main__":
    verify_perth_prediction()