import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import settings

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

def get_weather(x, normalized=False, normalize_params=None):
    return "TODO"
