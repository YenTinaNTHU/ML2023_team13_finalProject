import numpy as np
import pandas as pd
from time_preprocess import *
from config import settings
# pip install geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

def distance(lat1, lon1, lat2, lon2):
    
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...

#Function aiming at calculating the direction
def direction(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371 #km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

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
    df['direction'] = direction(df.pickup_latitude, df.pickup_longitude, df.dropoff_latitude, df.dropoff_longitude)
    return df

geolocator = Nominatim(user_agent="geoapiExercises")

def check_location_in_areas(lat, long):
    # try:
    #     location = geolocator.reverse([lat, long], exactly_one=True)
    #     address = location.raw['address']
    #     county = address.get('county', '')
    #     city = address.get('city', '')
    #     areas = ['New York County', 'Nassau', 'Suffolk', 'Westchester', 'Rockland', 'Dutchess', 'Orange', 'Putnam']
    #     if city == 'City of New York' or county in areas:
    #         return True
    #     else:
    #         return False
    # except GeocoderTimedOut:
    #     return check_location_in_areas(lat, long)
    counties = {
        'New York County': {'coords': (40.7831, -73.9712), 'area': 59.13},
        'Nassau': {'coords': (40.6546, -73.5594), 'area': 1173},
        'Suffolk': {'coords': (40.9849, -72.6151), 'area': 6146},
        'Westchester': {'coords': (41.1220, -73.7949), 'area': 1295},
        'Rockland': {'coords': (41.1489, -73.9830), 'area': 516},
        'Dutchess': {'coords': (41.7784, -73.7478), 'area': 2135},
        'Orange': {'coords': (41.3912, -74.3118), 'area': 2173},
        'Putnam': {'coords': (41.4351, -73.7949), 'area': 637},
    }
    def is_nearby(lat, lon):
        for county, details in counties.items():
            clat, clon = details['coords']
            # Convert area in km^2 to miles^2, then take square root
            limit = np.sqrt(details['area'] * 0.386102)  # 1 km^2 = 0.386102 miles^2
            if distance(lat, lon, clat, clon) <= limit:
                return True
        return False
    nearby = is_nearby(lat, long)
    if nearby:
        return True
    else:
        return False
        
        
def calculate_total_fixed_fee(df, train = False, simple=True):
    # Assume fixed base charge
    base_charge = 0   
    
    if simple:
        df['total_fixed_fees'] = base_charge
        return df
        
    # Assume df has 'pickup_datetime' as a pandas datetime column
    df['pickup_hour'] = df['pickup_datetime'].dt.hour


    # Calculate night and peak-hour surcharges
    df['night_surcharge'] = ((df['pickup_hour'] >= 20) | (df['pickup_hour'] < 6)) * 0.50
    df['peak_hour_surcharge'] = ((df['pickup_hour'] >= 16) & (df['pickup_hour'] < 20)) * 1.00

    # Newark surcharge for trips from Manhattan to Newark
    df['newark_surcharge'] = (df['from_Manhattan'] & df['to_EWR']) * 17.50

    # Improvement surcharge for 2015 and later
    df['pickup_year'] = df['pickup_datetime'].dt.year
    df['improvement_surcharge'] = (df['pickup_year'] >= 2015) * 0.30


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
def calculate_net_fare(df:pd.DataFrame(), simple=True):

    # Calculate all the fixed fees
    df = calculate_total_fixed_fee(df, train=True, simple=simple)

    # Subtract the fixed fees from the original fare to get the net fare
    df['net_fare'] = df['fare_amount'] - df['total_fixed_fees']

    # drop unnecessary column
    if not simple:
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

def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)

def add_coordinate_features(df):
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    # Add new features
    df['latdiff'] = (lat1 - lat2)
    df['londiff'] = (lon1 - lon2)

    return df


def add_distances_features(df):
    # Add distances from airpot and downtown
    ny = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)
    
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    df['euclidean'] = (df['latdiff'] ** 2 + df['londiff'] ** 2) ** 0.5
    df['manhattan'] = manhattan(lat1, lon1, lat2, lon2)
    
    df['downtown_pickup_distance'] = manhattan(ny[1], ny[0], lat1, lon1)
    df['downtown_dropoff_distance'] = manhattan(ny[1], ny[0], lat2, lon2)
    df['jfk_pickup_distance'] = manhattan(jfk[1], jfk[0], lat1, lon1)
    df['jfk_dropoff_distance'] = manhattan(jfk[1], jfk[0], lat2, lon2)
    df['ewr_pickup_distance'] = manhattan(ewr[1], ewr[0], lat1, lon1)
    df['ewr_dropoff_distance'] = manhattan(ewr[1], ewr[0], lat2, lon2)
    df['lgr_pickup_distance'] = manhattan(lgr[1], lgr[0], lat1, lon1)
    df['lgr_dropoff_distance'] = manhattan(lgr[1], lgr[0], lat2, lon2)
    
    return df