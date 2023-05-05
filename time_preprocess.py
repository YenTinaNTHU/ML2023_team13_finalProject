import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import settings
import requests

def get_year(date_time:pd.Series, mode='normal'):
    year = date_time.apply(lambda t: t.year)
    if mode == 'normalized':
        year = year.values.reshape(-1, 1)
        minmax_scaler = MinMaxScaler()
        year_normalized = minmax_scaler.fit_transform(year)
        return year_normalized
    elif mode == 'standardized':
        year = year.values.reshape(-1, 1)
        std_scaler = StandardScaler()
        year_standardized = std_scaler.fit_transform(year)
        return year_standardized
    else:
        return year
    

def get_weekday(date_time:pd.Series, mode='normal'):
    weekday = date_time.apply(lambda t: t.weekday())
    if mode == 'normalized':
        weekday = weekday.values.reshape(-1, 1)
        minmax_scaler = MinMaxScaler()
        weekday_normalized = minmax_scaler.fit_transform(weekday)
        return weekday_normalized
    elif mode == 'standardized':
        weekday = weekday.values.reshape(-1, 1)
        std_scaler = StandardScaler()
        weekday_standardized = std_scaler.fit_transform(weekday)
        return weekday_standardized
    elif mode == 'onehot':
        weekday_onehot = pd.get_dummies(weekday.replace({0: 7}))
        return weekday_onehot.values
    else:
        return weekday

def get_time(date_time:pd.Series, mode='normal', precision='hour'):
    time = date_time.apply(lambda t: t.hour)
    if precision == 'minute':
        hour = date_time.apply(lambda t: t.hour)
        minute = date_time.apply(lambda t: t.minute)
        time = hour*60+minute
        
    if mode == 'normalized':
        time = time.values.reshape(-1, 1)
        minmax_scaler = MinMaxScaler()
        time_normalized = minmax_scaler.fit_transform(time)
        return time_normalized
    elif mode == 'standardized':
        time = time.values.reshape(-1, 1)
        std_scaler = StandardScaler()
        time_standardized = std_scaler.fit_transform(time)
        return time_standardized
    else:
        return time

def get_is_holiday(date_time:pd.Series):
    def is_holiday(date):
        # 判斷是否為週六或週日
        if date.weekday() >= 5:
            return 1
        # 判斷是否為聯邦假日
        if (date.month == 1 and date.day == 1) or \
        (date.month == 7 and date.day == 4) or \
        (date.month == 12 and date.day == 25) or \
        (date.month == 12 and date.day == 31):
            return 1
        # 判斷馬丁路德金紀念日、總統日、好萊塢日、勞動節、哥倫布日和退伍軍人節是否為週一
        if (date.month == 1 and date.weekday() == 1 and 15 <= date.day <= 21) or \
        (date.month == 2 and date.weekday() == 1 and 15 <= date.day <= 21) or \
        (date.month == 5 and date.weekday() == 1 and 25 <= date.day <= 31) or \
        (date.month == 9 and date.weekday() == 1 and date.day <= 7) or \
        (date.month == 10 and date.weekday() == 1 and 8 <= date.day <= 14) or \
        (date.month == 11 and date.day == 11):
            return 1
        # 其他情況均視為工作日
        return 0
    return date_time.apply(is_holiday)

def get_weather(date_time:pd.Series, mode='normal'):
    # The longitude and latitude of New York City.
    lon = '-74.00'
    lat = '40.71'
    # get all data
    start_dt = min(date_time).strftime('%Y-%m-%d')
    end_dt = max(date_time).strftime('%Y-%m-%d')
    # call api
    url = f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_dt}&end_date={end_dt}&hourly=temperature_2m,weathercode&timezone=America%2FNew_York&timeformat=unixtime'
    response = requests.get(url)
    if response.status_code == 200:
        response = response.json()
        # search the weather by datetime
        weather_data_df = pd.DataFrame(response["hourly"])
        weather_data_df['time'] = pd.to_datetime(weather_data_df['time'],unit='s')
        weather_data_df['time'] = weather_data_df['time'].dt.floor('H')
        
        # 將兩個DataFrame的頻率都轉換成小時
        weather_df = pd.DataFrame({'timestamp': date_time})
        weather_df['hour'] = weather_df['timestamp'].dt.floor('H')
        weather_data_df['hour'] =  weather_data_df['time'].dt.floor('H')

        # 合併兩個DataFrame
        weather_df = pd.merge(weather_df, weather_data_df, on='hour', how='inner').drop('hour', axis=1)
        weather_df = weather_df.drop(columns=['timestamp', 'time'])

        if mode == 'normalized':
            scaler = MinMaxScaler()
            weather_df = pd.DataFrame(scaler.fit_transform(weather_df), columns=weather_df.columns)
        elif mode == 'standardized':
            scaler = StandardScaler()
            weather_df = pd.DataFrame(scaler.fit_transform(weather_df), columns=weather_df.columns)
        else:
            ...
        return weather_df.values
    else:
        print("Failed to get weather information.")