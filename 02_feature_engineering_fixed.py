import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Enable cache
fastf1.Cache.enable_cache('../data')

def calculate_practice_pace(session):
    """Calculate average practice pace for each driver in a session."""
    if session is None:
        return None
        
    try:
        laps = session.laps
        # Filter out in/out laps and slow laps (> 107% of fastest lap)
        fast_lap = laps.pick_fastest()['LapTime']
        threshold = fast_lap * 1.07
        
        valid_laps = laps[
            (laps['LapTime'] < threshold) &
            (laps['PitOutTime'].isna()) &
            (laps['PitInTime'].isna())
        ]
        
        # Group by driver abbreviation (VER, HAM, etc.) and calculate mean lap time
        avg_pace = valid_laps.groupby('Driver')['LapTime'].mean()
        return avg_pace
        
    except Exception as e:
        print(f"Error calculating practice pace: {e}")
        return None

def get_driver_abbreviation(session, driver_number):
    """Get 3-letter driver code from driver number."""
    try:
        return session.get_driver(driver_number)['Abbreviation']
    except:
        return driver_number

def process_season(year):
    """Process all races for a given season and extract features."""
    all_race_data = []
    
    # Get race schedule
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule['EventFormat'] == 'conventional']
    
    print(f"Processing {len(races)} races for {year}...")
    
    for round_num, event in tqdm(races.iterrows(), total=len(races), desc=f"Processing {year}"):
        try:
            event_name = event['EventName']
            print(f"\nProcessing {event_name}...")
            
            # Load race session first to get driver mappings
            try:
                race = fastf1.get_session(year, event_name, 'R')
                race.load()
            except:
                print(f"  Race not available for {event_name}")
                continue
            
            # Load other sessions
            try:
                fp2 = fastf1.get_session(year, event_name, 'FP2')
                fp2.load()
            except:
                fp2 = None
                print(f"  FP2 not available for {event_name}")
                
            try:
                fp3 = fastf1.get_session(year, event_name, 'FP3')
                fp3.load()
            except:
                fp3 = None
                print(f"  FP3 not available for {event_name}")
                
            try:
                qualifying = fastf1.get_session(year, event_name, 'Q')
                qualifying.load()
            except:
                qualifying = None
                print(f"  Qualifying not available for {event_name}")
            
            # Calculate average practice pace
            pace_fp2 = calculate_practice_pace(fp2) if fp2 else None
            pace_fp3 = calculate_practice_pace(fp3) if fp3 else None
            
            # Combine practice paces (priority to FP3)
            avg_practice_pace = {}
            if pace_fp3 is not None:
                for driver, pace in pace_fp3.items():
                    avg_practice_pace[driver] = pace
            elif pace_fp2 is not None:
                for driver, pace in pace_fp2.items():
                    avg_practice_pace[driver] = pace
            
            # Get qualifying results with driver abbreviations
            qualifying_results = {}
            if qualifying and hasattr(qualifying, 'results'):
                for _, result in qualifying.results.iterrows():
                    driver_num = result['DriverNumber']
                    driver_abbr = get_driver_abbreviation(qualifying, driver_num)
                    pos = result['Position']
                    qualifying_results[driver_abbr] = pos
            
            # Get grid positions with driver abbreviations
            grid_positions = {}
            team_info = {}
            race_results = {}
            
            if race and hasattr(race, 'results'):
                for _, result in race.results.iterrows():
                    driver_num = result['DriverNumber']
                    driver_abbr = get_driver_abbreviation(race, driver_num)
                    grid_pos = result['GridPosition']
                    finish_pos = result['Position']
                    team = result['TeamName']
                    
                    grid_positions[driver_abbr] = grid_pos
                    race_results[driver_abbr] = finish_pos
                    team_info[driver_abbr] = team
            
            # Compile data for this race
            for driver in race_results.keys():
                race_data = {
                    'Year': year,
                    'Round': event['RoundNumber'],
                    'Track': event_name,
                    'Driver': driver,
                    'Team': team_info.get(driver, 'Unknown'),
                    'AvgPracticePace': avg_practice_pace.get(driver, None),
                    'QualifyingPos': qualifying_results.get(driver, None),
                    'GridPosition': grid_positions.get(driver, None),
                    'FinishPosition': race_results.get(driver, None)
                }
                all_race_data.append(race_data)
                    
        except Exception as e:
            print(f"Error processing {event_name}: {e}")
            continue
    
    return pd.DataFrame(all_race_data)

def main():
    """Main function to process data and save features."""
    print("Starting feature engineering...")
    
    # Process 2023 data
    df_2023 = process_season(2023)
    
    # Save processed data
    output_path = '../data/processed_features_2023.csv'
    df_2023.to_csv(output_path, index=False)
    print(f"\nFeature engineering complete! Data saved to {output_path}")
    
    # Show preview
    print(f"\nGenerated {len(df_2023)} driver-race combinations")
    print("\nFirst few rows:")
    print(df_2023.head())

if __name__ == "__main__":
    main()