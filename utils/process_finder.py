import numpy as np
import pandas as pd
from common_utils import *
    
"""Find Excavation States"""
def find_exc_start_time(dt_thre,df):
    date_time = df['datetime']
    t_prev = date_time.iloc[0]
    start_time_list = [t_prev]
    for i,t in enumerate(date_time):
        if i >0:
            exc_start_condition = get_start_condition(t,t_prev,dt_thre,df)
            if exc_start_condition:
                start_time_list.append(t)
                t_prev = t
    return start_time_list[1:]

def get_start_condition(t,t_prev,dt_thre,df):
    dt = abs(t-t_prev)
    return ((dt>dt_thre) and (verify_exc(t,df)))

def verify_exc(detect_time,df):
    start,end = get_start_end_sample_time(detect_time)
    sample_df = select_short_period_df(df,start,end)
    score = get_excavation_score(sample_df)
    exc_score_thre = 1.5*len(sample_df)
    return (score> exc_score_thre)
        
def get_start_end_sample_time(detect_time):
    before_dt = pd.to_timedelta(0,'min')
    after_dt = pd.to_timedelta(25,'min')
    start = detect_time-before_dt
    end = detect_time+after_dt
    return start,end

def get_excavation_score(sample_df):
    excavator_factor = 0.5
    loader_factor = 0.5
    negative_factor = 1
    score = 0
    score+=1.5*sum(list(sample_df['cls_pred']==1))
    score+=excavator_factor*sum(list(sample_df['Excavator']))
    score+=loader_factor*sum(list(sample_df['Loader']))
    score+= negative_factor*get_negative_score(sample_df)
    return score


def get_negative_score(sample_df):
    bad_equips = ['Jumbo','Erector_shotcrete_equip']
    score = 0
    for equip in bad_equips:
        score -= sum(list(sample_df[equip]))
    return score
        
"""Tools to evaluate finding accuracy"""
def get_matched_date(start_time_est,start_time_real,dt_thre):
    matched_date = []
    for est_st in start_time_est:
        for real_st in start_time_real:
            match = date_match(est_st,real_st,dt_thre)
            if match:
                matched_date.append(real_st)
    return matched_date
            
def date_match(t1,t2,dt_thre):
    match = abs((t1-t2))<dt_thre
    return match

def get_unmatched_date(matched,original):
    unmatched_date = list(set(original)-set(matched))
    return np.sort(unmatched_date)

def find_closest_time(time_list,searching_time):
    searching_time = pd.to_datetime(searching_time)
    min_dt = pd.to_timedelta(10,'hour')
    for t in time_list:
        dt = abs(t-searching_time)
        if dt <min_dt:
            min_dt = dt
            close_t = t
    return close_t
            
"""Tools to find processes"""
def update_process_time(start_time_list_real,df):
    df = df.copy()
    start_end_time_pairs = get_start_end_time_pairs(start_time_list_real)
    for start, end in start_end_time_pairs:
        process_mask = get_process_mask(df,start,end)
        process_df = df[process_mask]
        process_time = get_process_time(process_df)
        df.loc[process_mask, 'in_process_time'] = process_time
        norm_process_time = 5*(process_time/max(process_time))
        df.loc[process_mask, 'process'] = norm_process_time
    return df

def get_start_end_time_pairs(start_time_list_real):
    pairs = []
    start = start_time_list_real[0]
    for t in start_time_list_real[1:]:
        pair = (start,t)
        pairs.append(pair)
        start = t
    return pairs

def get_process_mask(df,start_time,end_time):
    return ((df['datetime']>=start_time) & 
                 (df['datetime']<=end_time))


def get_process_time(process_df,thre = 120):
    time_list = list(process_df['datetime'])
    start_time = time_list[0]
    process_time = np.zeros(len(time_list))
    
    for i,time in enumerate(time_list[1:]):
        dt = time-start_time
        dt_minutes = dt.seconds/60
        process_time[i+1] = dt_minutes
        if i > thre:
            break   
    process_time[0] = 1
    return process_time

def get_df_with_process(short_period_df):
    short_period_df['datetime'] = pd.DatetimeIndex(short_period_df['datetime'])
    short_period_df['in_process_time'] = 0
    short_period_df['process'] = 0
    
    dt_thre_real = pd.to_timedelta(3,'hour')
    start_time_list = find_exc_start_time(dt_thre_real,short_period_df)  
    start_end_time_pairs = get_start_end_time_pairs(start_time_list)
    df_time = update_process_time(start_time_list,short_period_df)
    
    return (df_time,start_end_time_pairs)
    
