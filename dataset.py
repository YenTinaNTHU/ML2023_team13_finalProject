# coding: utf-8
import os
import numpy as np
import pandas as pd
from time_preprocess import *
from location_preprocess import *
from torch.utils import data
from sklearn.model_selection import train_test_split
from config import settings


DATA_PATH = './data'
'reference: https://www.kaggle.com/code/debjeetdas/nyc-taxi-fare-eda-prediction-using-linear-reg/notebook'            

def distance(lat1, lon1, lat2, lon2):
    
    p = 0.017453292519943295 # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p)/2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    return 0.6213712 * 12742 * np.arcsin(np.sqrt(a)) # 2*R*asin...

def load_data(filename='train', total_sample=None, random_sample=None, scaling_transformers=None):
    if filename=='train':
        df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'), nrows=total_sample)
        print(f'loaded csv file shape: {df.shape}')
        if random_sample:
            rn_sample_df = df.sample(random_sample, random_state=settings.RANDOM_SEED)
        else:
            rn_sample_df = df
    else:
        df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
        print(f'loaded test csv file shape: {df.shape}')
        rn_sample_df = df
    rn_sample_df['pickup_datetime'] = pd.to_datetime(rn_sample_df['pickup_datetime'], format= "%Y-%m-%d %H:%M:%S UTC")

    # remove illegal data    
    if filename == 'train':
        rn_sample_df = rn_sample_df.dropna()


    year_scaler = None
    weekday_scaler = None
    time_scaler = None
    weather_scaler = None
    if scaling_transformers:
        year_scaler = scaling_transformers['year']
        weekday_scaler = scaling_transformers['weekday']
        time_scaler = scaling_transformers['time']
        weather_scaler = scaling_transformers['weather']

    # 2009-06-15 17:26:21 UTC
    # add time information
    print('setting time info...')
    rn_sample_df['year'], year_scaler = get_year(rn_sample_df.pickup_datetime, transformer = year_scaler, train=(filename=='train'))
    rn_sample_df[['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']], weekday_scaler = get_weekday(rn_sample_df.pickup_datetime, mode='onehot', transformer = weekday_scaler,train=(filename=='train'))
    rn_sample_df['hour'], time_scaler = get_time(rn_sample_df.pickup_datetime, transformer = time_scaler, train=(filename=='train'))
    rn_sample_df['is_holiday'] = get_is_holiday(rn_sample_df.pickup_datetime)
    rn_sample_df[['temperature','weathercode']], weather_scaler = get_weather(rn_sample_df.pickup_datetime, transformer = weather_scaler, train=(filename=='train'))
    rn_sample_df['distance'] = distance(rn_sample_df.pickup_latitude, rn_sample_df.pickup_longitude, 
                                    rn_sample_df.dropoff_latitude, rn_sample_df.dropoff_longitude)
    
    if scaling_transformers:
        scaling_transformers['year'] = year_scaler
        scaling_transformers['week_day'] = weekday_scaler
        scaling_transformers['time'] = time_scaler
        scaling_transformers['weather'] = weather_scaler
    
    # add geographical information
    print('setting geo info...')
    # THRESHOLD = 4
    # JFK_COORD = (40.641766, -73.780968)
    # LGA_COORD = (40.776927, -73.873966)
    # EWR_COORD = (40.689531, -74.174462)
    # MANHATTAN = (40.776676, -73.971321)
    
    # rn_sample_df['from_JKF']        = distance(rn_sample_df.pickup_latitude, rn_sample_df.pickup_longitude, \
    #                                     *JFK_COORD) <= THRESHOLD
    # rn_sample_df['to_JKF']          = distance(rn_sample_df.dropoff_latitude, rn_sample_df.dropoff_longitude, \
    #                                     *JFK_COORD) <= THRESHOLD        
    # rn_sample_df['from_LGA']        = distance(rn_sample_df.pickup_latitude, rn_sample_df.pickup_longitude, \
    #                                     *LGA_COORD) <= THRESHOLD
    # rn_sample_df['to_LGA']          = distance(rn_sample_df.dropoff_latitude, rn_sample_df.dropoff_longitude, \
    #                                     *LGA_COORD) <= THRESHOLD  
    # rn_sample_df['to_EWR']          = distance(rn_sample_df.dropoff_latitude, rn_sample_df.dropoff_longitude, \
    #                                     *EWR_COORD) <= THRESHOLD  
    # rn_sample_df['from_Manhattan']  = distance(rn_sample_df.pickup_latitude, rn_sample_df.pickup_longitude, \
    #                                     *MANHATTAN) <= THRESHOLD
    # rn_sample_df['to_Manhattan']    = distance(rn_sample_df.dropoff_latitude, rn_sample_df.dropoff_longitude, \
    #                                     *MANHATTAN) <= THRESHOLD

    rn_sample_df = add_location_info(rn_sample_df)
    
    if filename == 'train':
        print('counting net fare...')
        rn_sample_df = calculate_net_fare(rn_sample_df)
    else:
        # we cannot count the net fare for test set
        print('counting fixed fee...')
        rn_sample_df = calculate_total_fixed_fee(rn_sample_df)

    # remove illegal data    
    if filename == 'train':
        rn_sample_df = rn_sample_df.dropna()
        rn_sample_df = rn_sample_df[rn_sample_df.net_fare > 0]
        rn_sample_df = rn_sample_df[rn_sample_df.distance > 0]
        rn_sample_df = rn_sample_df[(0 <rn_sample_df.passenger_count) & (rn_sample_df.passenger_count < 7)]
        rn_sample_df = rn_sample_df.drop(['key'], axis=1)
    '''without location'''
    rn_sample_df = rn_sample_df.drop(['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], axis=1)  
    
    '''with location'''
    # rn_sample_df = rn_sample_df.drop(['pickup_datetime'], axis=1)
    
    return rn_sample_df, scaling_transformers

class DataFolder(data.Dataset):
    def __init__(self, split='train', df=None, transformers=None):
        assert(split == 'train' or split == 'test' or split == 'valid')
        self.split = split
        if split == 'train' or split == 'valid':
            train_df, valid_df = train_test_split(df, test_size=settings.valid_rate, random_state=settings.RANDOM_SEED)
            self.df = valid_df if split == 'valid' else train_df
            self.features = self.df.drop(['fare_amount','net_fare'], axis=1).values
        else:
            self.df, self.transformers = load_data(filename='test', random_sample=settings.testN, scaling_transformers=transformers)
            self.key_list = self.df.key.values
            self.features = self.df.drop('key', axis=1).values


    def __getitem__(self, index):
        if self.split == 'test':
            return self.features[index].astype(np.float32)
        else:
            # return self.features[index].astype(np.float32), self.df['fare_amount'].values[index].astype(np.float32)
            return self.features[index].astype(np.float32), self.df['net_fare'].values[index].astype(np.float32)


    def __len__(self):
        if self.split == 'test':
            return np.size(self.df.values, 0)
        else:
            # return len(self.df.fare_amount)
            return len(self.df.net_fare)

