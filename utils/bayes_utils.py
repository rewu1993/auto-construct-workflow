import pandas as pd 
import numpy as np
from common_utils import *
from data_prepare_utils import filt_res

"""Functions for combine bayes with cls results"""
def update_combined_res(df_time,start_end_time_pairs,prior_knowledge):
    df_time['combined_pred'] = df_time['cls_pred']
    acc,count = 0,0
    anom_scores = []
    for i in range (len(start_end_time_pairs)):
        s,e = start_end_time_pairs[i]
        cond = (df_time['datetime']>s) & (df_time['datetime']<e)

        df_process = df_time[cond]
        res_df,anom_score = get_bayes_pred(df_process,prior_knowledge)

        cls_pred = np.array(df_process['cls_pred'])
        bayes_pred = np.array(res_df['bayes_pred'])
        combined_res = get_combined_result(cls_pred,bayes_pred)
        
        
        df_time.loc[cond,'combined_pred'] = combined_res
        anom_scores.append(anom_score)
        
                
        prob_combine = sum(combined_res==res_df['label'])/len(res_df)
        prob_cls = sum(cls_pred==res_df['label'])/len(res_df)
        print (prob_combine,prob_cls,anom_score)
        acc+= prob_combine
        count +=1
    print ("final prob", acc/count)

    return df_time,anom_scores
    
def get_bayes_pred(df_process,prior_knowledge):
    time_prob_df,veh_prob_df,veh_names = prior_knowledge
    res_df = create_empty_df()
    final_score = np.zeros(6)
    penalty = 0.25
    anomaly_score = 0
    for index, row in df_process.iterrows():
        time_prob = get_time_condition_prob(row,time_prob_df)
        veh_probs = get_veh_condition_probs(row,veh_prob_df,veh_names)
        final_score = final_score*(1-penalty)
        for veh_prob in veh_probs:
            time_state_prob = veh_prob*time_prob
            final_score =  update_final_score(final_score,time_state_prob)
        cur_state = get_res_df(row,final_score)      
        if uncertain(final_score): 
            anomaly_score += 1
        res_df = res_df.append(cur_state)
        anom_norm = anomaly_score/len(df_process)
    return res_df,anom_norm


def get_combined_result(cls_pred,bayes_pred):
    combined_pred = []
    for i in range(len(cls_pred)):
        cls = cls_pred[i]
        bayes = bayes_pred[i]
        if (cls != bayes) and odd(i,cls_pred):
            combined_pred.append(bayes)
        else:
            combined_pred.append(cls)
    return filt_res(np.array(combined_pred))      

def odd(i,l):
    prev = max(0,i-1)
    nxt = min(len(l)-1,i+1)
    if (l[i] != l[prev]) & (l[i] != l[nxt]):
        return True
    return False


def calculate_condition_prob(condition_counts):
    return condition_counts/sum(condition_counts)

def get_time_condition_prob(row,time_prob_df):
    dt = float(row['in_process_time'])
    prob_states = time_prob_df[time_prob_df['t']==dt]
    if prob_states.empty:
        return np.zeros(6)
    else:
        a = [float(prob_states['rest']),float(prob_states['exc']),float(prob_states['muc']),float(prob_states['prf']),
           float(prob_states['sht']),float(prob_states['blt'])]
    return np.array(a)

def get_veh_condition_probs(row,veh_prob_df,veh_names):
    detected_veh = get_detected_veh(row,veh_names)
    veh_list = list(veh_prob_df.columns)[1:]
    veh_condition_probs = []
    for veh_idx in detected_veh:
        veh = veh_list[veh_idx]
        veh_condition_probs.append(np.array(veh_prob_df[veh]))
    return veh_condition_probs

def get_detected_veh(row,veh_names):
    detected_thre = 0.2 
    detected_veh = []
    for i, vehs in enumerate(veh_names):
        for veh in vehs:
            if row[veh]>detected_thre:
                detected_veh.append(i)
    if not detected_veh:
        detected_veh.append(8)
    return set(detected_veh)
        
"""Results data frame"""   
def create_empty_df(column_names = ['t','label','bayes_pred','path']):
    df = pd.DataFrame(columns=column_names)
    return df

def get_res_df(row,final_score):
    time = [row['in_process_time']]
    label = [row['label']]
    img_path = [row['path']]
    preds = interpret_final_score(final_score)
    df = create_empty_df()
    for i,pred in enumerate(preds):
        pred = [label_trans(int(pred[0]))]
        df.loc[i] = time+label+pred+img_path
    return df
def interpret_final_score(final_score):
    max_prob = 0
    for i,state_score in enumerate(final_score):
        if state_score>max_prob:
            max_prob = state_score
    states = find_possible_state (max_prob,final_score)
    return states

def find_possible_state(max_prob,final_score):
    thre = 0.2
    possible_state = []    
    for i,state_score in enumerate(final_score):
        if state_score == max_prob:   
#             score_diff = abs(max_prob-state_score)
#             if thre/10>score_diff:
            possible_state.append((i,state_score))
            break
    if not possible_state:
        possible_state.append((0,0.1))
    return possible_state
  
'''Visualize Utils '''
def update_final_score(final_score,time_state_prob):
    for i in range(6):
        final_score[i] = max(final_score[i],time_state_prob[i])
    return final_score

def uncertain(final_score):
    return (max(final_score)<0.2)

def plot_res(df):
    plot_df = pd.melt(df, id_vars=['datetime'], value_vars=['label', 'pred'])
    LINE_PLOT_CONFIG['x_name'] = 'datetime'
    LINE_PLOT_CONFIG['title'] = 'work process'
    plot_lines(plot_df,LINE_PLOT_CONFIG)
            
def plot_process(df):
    plot_df = pd.melt(df, id_vars=['datetime'], value_vars=['label', 'process'])
    LINE_PLOT_CONFIG['x_name'] = 'datetime'
    LINE_PLOT_CONFIG['title'] = 'work process'
    plot_lines(plot_df,LINE_PLOT_CONFIG)

        