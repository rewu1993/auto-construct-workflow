import pandas as pd
import numpy as np
from site_info import *

"""Function to get datetime and site info"""
def convert_name_date_site(name):
    date_site = remove_jpg(name)
    time,site = get_time_site(date_site)
    return time,site 

def remove_jpg(name):
    name = name[:-4]
    return name

def get_time_site(date_site):
    time,site = date_site.split('_')
    return time[1:],site

"""Function to conver one-hot label to label"""
def sum_label(df):
    label = 0*round(df[OP_LIST[0]])
    for i in range(1, len(OP_LIST)):
        label += i* round(df[OP_LIST[i]])
    return list(label)

def get_path(label,base_path):
    label_idx = int(label)
    op_name = OP_LIST[label_idx]
    path = base_path+op_name+'/'
    return path
    
        
"""Functions to get the pre and post state for current state"""
def chunck_size(cycle):
    duration_list = []
    for c in cycle:
        start,end = c
        duration_list.append(end-start)
    return np.array(duration_list)

def filt_noise(i, label_list, min_duration = 2, prev = False):
    if not prev:
        nxt_label = label_list[i]
        stab_label = label_list[i+min_duration-1]
        if nxt_label==stab_label:
            return nxt_label
        else: 
#             print ("return stable label",label_list[i-1],
#                    nxt_label,stab_label,label_list[i+min_duration])
            return stab_label
    if prev:
        if i-1<0: return 0
        prev_label = label_list[i-1]
        stab_label = label_list[i-min_duration]
        if prev_label==stab_label:
            return prev_label
        else: 
#             print ("return stable label")
            return stab_label
    
def prev_nxt_state(cycle,label_list,prev_state_list,post_state_list):
    for c in cycle:
        start,end = c
        if (end+1)<len(label_list):
            nxt_state = filt_noise(end,label_list)
            prev_state = filt_noise(start,label_list,prev=True)
            prev_state_list[start:end] = prev_state
            post_state_list[start:end] = nxt_state
                   
def get_prev_post_state_list(label_list):
    prev_state_list = np.zeros(len(label_list))
    post_state_list = np.zeros(len(label_list))
    for v in range(len(OP_LIST)):
        cycle = find_repeat_data(label_list,v,thre=1)
        prev_nxt_state(cycle,label_list,prev_state_list,post_state_list)
    return prev_state_list,post_state_list
        
def find_repeat_data(data_list,val,thre=0):
    """A function that find consecutive chuncks from the list"""
    start , end = 0,0
    len_list = []
    for elem in data_list:
        dif = end-start
        end+=1
        if elem!=val:
            if dif>thre:
                len_list.append((start,end-1))
            start = end
    dif = end-start
    if dif>thre:len_list.append((start,end-1)) 
    return len_list
            
            
def filt_res(pred_list,thre_w=15,thre_r = 3):
    """A function that remove small chunks of rest/work from prediction"""
    pred_res_cycles = find_repeat_data(pred_list,0)
    
    filter_pred = pred_list.copy()
    # remove the rest cycles first for small rest segment
    for prc in pred_res_cycles:
        start,end = prc
        duration = end-start
        if duration <thre_r:
            print ("remove wrong rest:: ",start,end)
            filter_pred[start:end]=1
    # remove work cycle
    pred_work_cycles = find_repeat_data(filter_pred,1)        
    for pwc in pred_work_cycles:
        start,end = pwc
        duration = end-start
        if duration <thre_w:
            print ("remove wrong works: ",start,end)
            filter_pred[start:end]=0
    return filter_pred

      
        