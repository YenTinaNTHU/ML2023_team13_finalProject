import numpy as np
import pandas as pd
from time_preprocess import *
from config import settings

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def distance(lat1, lon1, lat2, lon2):
    
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...

# add geographical information
def add_location_info(df:pd.DataFrame()):
    THRESHOLD = 4
    JFK_COORD = (40.641766, -73.780968)
    LGA_COORD = (40.776927, -73.873966)
    EWR_COORD = (40.689531, -74.174462)
    MANHATTAN = (40.776676, -73.971321)

    df['from_JKF']        = distance(df.pickup_latitude, df.pickup_longitude, \
                                        *JFK_COORD) <= THRESHOLD
    df['to_JKF']          = distance(df.dropoff_latitude, df.dropoff_longitude, \
                                        *JFK_COORD) <= THRESHOLD        
    df['from_LGA']        = distance(df.pickup_latitude, df.pickup_longitude, \
                                        *LGA_COORD) <= THRESHOLD
    df['to_LGA']          = distance(df.dropoff_latitude, df.dropoff_longitude, \
                                        *LGA_COORD) <= THRESHOLD  
    df['to_EWR']          = distance(df.dropoff_latitude, df.dropoff_longitude, \
                                        *EWR_COORD) <= THRESHOLD  
    df['from_Manhattan']  = distance(df.pickup_latitude, df.pickup_longitude, \
                                        *MANHATTAN) <= THRESHOLD
    df['to_Manhattan']    = distance(df.dropoff_latitude, df.dropoff_longitude, \
                                        *MANHATTAN) <= THRESHOLD
    df['distance'] = distance(df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude)
    return df

geolocator = Nominatim(user_agent="geoapiExercises")

def check_location_in_areas(lat, long):
    try:
        location = geolocator.reverse([lat, long], exactly_one=True)
        address = location.raw['address']
        county = address.get('county', '')
        city = address.get('city', '')
        areas = ['New York County', 'Nassau', 'Suffolk', 'Westchester', 'Rockland', 'Dutchess', 'Orange', 'Putnam']
        if city == 'City of New York' or county in areas:
            return True
        else:
            return False
    except GeocoderTimedOut:
        return check_location_in_areas(lat, long)
    
def calculate_total_fixed_fee(df, accelerate = True, train = False):
    # Assume df has 'pickup_datetime' as a pandas datetime column
    df['pickup_hour'] = df['pickup_datetime'].dt.hour

    # Assume fixed base charge
    base_charge = 2.50

    # Calculate night and peak-hour surcharges
    df['night_surcharge'] = ((df['pickup_hour'] >= 20) | (df['pickup_hour'] < 6)) * 0.50
    df['peak_hour_surcharge'] = ((df['pickup_hour'] >= 16) & (df['pickup_hour'] < 20)) * 1.00

    # Newark surcharge for trips from Manhattan to Newark
    df['newark_surcharge'] = (df['from_Manhattan'] & df['to_EWR']) * 17.50

    # Improvement surcharge for 2015 and later
    df['pickup_year'] = df['pickup_datetime'].dt.year
    df['improvement_surcharge'] = (df['pickup_year'] >= 2015) * 0.30

    # Estimate MTA state surcharge based on dropoff location
    if accelerate:
        df['mta_state_surcharge'] = 0.5
    else:
        df['mta_state_surcharge'] = df.apply(lambda row: 0.50 if check_location_in_areas(row['dropoff_latitude'], row['dropoff_longitude']) else 0, axis=1)

    # Calculate all the fixed fees
    df['total_fixed_fees'] = base_charge + df['mta_state_surcharge'] + df['night_surcharge'] + df['peak_hour_surcharge'] + df['newark_surcharge'] + df['improvement_surcharge']

    if not train:
        # drop unnecessary column
        df = df.drop([
            'pickup_hour',
            'mta_state_surcharge',
            'night_surcharge',
            'peak_hour_surcharge',
            'newark_surcharge',
            'pickup_year',
            'improvement_surcharge',
            ], axis=1)

    return df

# call this only for training
def calculate_net_fare(df:pd.DataFrame(), accelerate = True):

    # Calculate all the fixed fees
    df = calculate_total_fixed_fee(df, accelerate, train=True)

    # Subtract the fixed fees from the original fare to get the net fare
    df['net_fare'] = df['fare_amount'] - df['total_fixed_fees']

    if accelerate:
        # If the net fare is negative, check if the trip ends in a specified area
        df.loc[df['net_fare'] < 0, 'in_areas'] = df[df['net_fare'] < 0].apply(lambda row: check_location_in_areas(row['dropoff_latitude'], row['dropoff_longitude']), axis=1)
        # Assume fixed base charge
        base_charge = 2.50
        # If the trip does not end in the specified area, refund the MTA state surcharge and recalculate the net fare
        df.loc[df['in_areas'] == False, 'mta_state_surcharge'] = 0
        df['total_fixed_fees'] = base_charge + df['mta_state_surcharge'] + df['night_surcharge'] + df['peak_hour_surcharge'] + df['newark_surcharge'] + df['improvement_surcharge']
        df['net_fare'] = df['fare_amount'] - df['total_fixed_fees']

    # drop unnecessary column
    df = df.drop([
        'pickup_hour',
        'mta_state_surcharge',
        'night_surcharge',
        'peak_hour_surcharge',
        'newark_surcharge',
        'pickup_year',
        'improvement_surcharge',
        ], axis=1)
    df = df.drop([
        'in_areas'
        ], axis=1)

    return df
