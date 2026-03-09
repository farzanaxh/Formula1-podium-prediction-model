import fastf1
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Enable cache
fastf1.Cache.enable_cache('../data')

def predict_current_race(year, event_name):
    """
    Predict podium for a current race weekend after qualifying.
    """
    print(f"Predicting podium for {year} {event_name}...")
    
    try:
        # Load the trained model
        with open('../models/podium_predictor_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Load current sessions
        print("Loading current race data...")
        try:
            fp2 = fastf1.get_session(year, event_name, 'FP2')
            fp2.load()
            print("✓ FP2 data loaded")
        except:
            fp2 = None
            print("✗ FP2 not available")
            
        try:
            fp3 = fastf1.get_session(year, event_name, 'FP3') 
            fp3.load()
            print("✓ FP3 data loaded")
        except:
            fp3 = None
            print("✗ FP3 not available")
            
        try:
            qualifying = fastf1.get_session(year, event_name, 'Q')
            qualifying.load()
            print("✓ Qualifying data loaded")
        except:
            print("✗ Qualifying data not available - need qualifying results to predict!")
            return
            
        # Calculate average practice pace
        def calculate_practice_pace(session):
            if session is None:
                return None
            try:
                laps = session.laps
                fast_lap = laps.pick_fastest()['LapTime']
                threshold = fast_lap * 1.07
                
                valid_laps = laps[
                    (laps['LapTime'] < threshold) &
                    (laps['PitOutTime'].isna()) & 
                    (laps['PitInTime'].isna())
                ]
                
                avg_pace = valid_laps.groupby('Driver')['LapTime'].mean()
                return avg_pace
            except:
                return None
        
        # Get practice paces
        pace_fp2 = calculate_practice_pace(fp2)
        pace_fp3 = calculate_practice_pace(fp3)
        
        # Combine practice paces (priority to FP3)
        avg_practice_pace = {}
        if pace_fp3 is not None:
            for driver, pace in pace_fp3.items():
                avg_practice_pace[driver] = pace.total_seconds()
        elif pace_fp2 is not None:
            for driver, pace in pace_fp2.items():
                avg_practice_pace[driver] = pace.total_seconds()
        else:
            print("✗ No valid practice data available")
            return
        
        # Get qualifying and grid positions
        qualifying_results = {}
        grid_positions = {}
        team_info = {}
        
        for _, result in qualifying.results.iterrows():
            driver = result['DriverNumber']
            driver_abbr = qualifying.get_driver(driver)['Abbreviation']
            qualifying_results[driver_abbr] = result['Position']
            grid_positions[driver_abbr] = result['Position']  # Assuming no penalties for prediction
            team_info[driver_abbr] = result['TeamName']
        
        # Prepare prediction data
        prediction_data = []
        for driver in qualifying_results.keys():
            if driver in avg_practice_pace:  # Only drivers with practice data
                pred_row = {
                    'Driver': driver,
                    'Team': team_info.get(driver, 'Unknown'),
                    'AvgPracticePace_seconds': avg_practice_pace[driver],
                    'QualifyingPos': qualifying_results[driver],
                    'GridPosition': grid_positions[driver]
                }
                prediction_data.append(pred_row)
        
        # Create DataFrame and make predictions
        pred_df = pd.DataFrame(prediction_data)
        
        # Features must be in same order as training
        features = ['AvgPracticePace_seconds', 'QualifyingPos', 'GridPosition']
        X_pred = pred_df[features]
        
        # Make predictions
        podium_probs = model.predict_proba(X_pred)[:, 1]  # Probability of podium
        pred_df['Podium_Probability'] = podium_probs
        
        # Sort by probability (most likely to podium)
        pred_df = pred_df.sort_values('Podium_Probability', ascending=False)
        
        # Display predictions
        print(f"\n🎯 PODIUM PREDICTIONS for {year} {event_name}")
        print("=" * 50)
        
        for i, (_, row) in enumerate(pred_df.head(10).iterrows(), 1):
            podium_emoji = " 🏆" if i <= 3 else ""
            print(f"{i:2d}. {row['Driver']:3} ({row['Team']:15}) - {row['Podium_Probability']:.1%}{podium_emoji}")
        
        print(f"\nTop 3 predicted podium:")
        podium_drivers = pred_df.head(3)['Driver'].tolist()
        print(f"🥇 {podium_drivers[0]}")
        print(f"🥈 {podium_drivers[1]}") 
        print(f"🥉 {podium_drivers[2]}")
        
        return pred_df
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to make predictions."""
    print("F1 Podium Predictor - Live Prediction")
    print("=" * 40)
    
    # Example: Predict for a specific race
    # Replace with current year and race name
    year = 2024
    race_name = "Monaco Grand Prix"  # Change this to current race
    
    predict_current_race(year, race_name)
    
    print(f"\nNote: Replace 'race_name' with the current Grand Prix name")
    print("Run this script after qualifying to get predictions!")

if __name__ == "__main__":
    main()