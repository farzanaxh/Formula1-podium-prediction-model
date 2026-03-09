import fastf1
import pandas as pd
import os
from tqdm import tqdm

def cache_historical_data(years, cache_dir='../data'):
    """
    Cache historical F1 data for specified years.
    
    Args:
        years (list): List of years to cache data for
        cache_dir (str): Directory to store cached data
    """
    
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created cache directory: {cache_dir}")
    
    # Enable the cache
    fastf1.Cache.enable_cache(cache_dir)
    
    # Get schedule for each year
    for year in years:
        print(f"Processing year {year}...")
        try:
            schedule = fastf1.get_event_schedule(year)
        except Exception as e:
            print(f"Error getting schedule for {year}: {e}")
            continue
        
        # Filter out testing events
        schedule = schedule[schedule['EventFormat'] == 'conventional']
        
        # Iterate through each race weekend
        for _, event in tqdm(schedule.iterrows(), total=len(schedule), desc=f"Year {year}"):
            event_name = event['EventName']
            try:
                # Cache session data - get all important sessions
                sessions_to_cache = ['FP1', 'FP2', 'FP3', 'Qualifying', 'Race']
                
                for session in sessions_to_cache:
                    try:
                        # This will download and cache the data if not already cached
                        session_data = fastf1.get_session(year, event_name, session)
                        session_data.load()  # This triggers the caching
                    except Exception as e:
                        print(f"Error caching {session} for {event_name} {year}: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing {event_name} {year}: {e}")
                continue
    
    print("Data caching completed!")

if __name__ == "__main__":
    # Start with just 2023 for testing
    years_to_cache = [2023]
    cache_historical_data(years_to_cache)
    print("Historical data caching complete!")