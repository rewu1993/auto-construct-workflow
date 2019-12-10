import pandas as pd 
import numpy as np

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
def create_empty_df(column_names = ['datetime','label','pred']):
    df = pd.DataFrame(columns=column_names)
    return df

def get_res_df(row,final_score):
    time = [row['datetime']]
    label = [row['label']]
    preds = interpret_final_score(final_score)
    df = create_empty_df()
    for i,pred in enumerate(preds):
        pred = [float(pred[0])]
        df.loc[i] = time+label+pred
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
        if state_score > thre:   
            score_diff = abs(max_prob-state_score)
            if thre/5>score_diff:
                possible_state.append((i,state_score))
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

        