import pandas as pd
import numpy as np

def select_short_period_df(df,start_date,end_date):
    time_constrain = (df['time']>start_date) & (df['time']<end_date)
    return df[time_constrain]
    