import pandas as pd
import numpy as np
from site_info import *
    
    
'''Functions to extract vehicle information from raw data (dataframe) to dictionary with label, veh_list, site_name'''

def get_image_detected_vehicle_dict(df):
    detected_vehicle_dict = {}
    for index,row in df.iterrows():
        image_path = row['path']
        image_info = get_image_info(row)
        detected_vehicle_dict[image_path] = image_info
    return detected_vehicle_dict
        
def get_image_info(row):
    label = get_state_label(row)
    veh_list = get_detected_vehicles(row)
    info = {'label': label,
            'veh_list':veh_list,}
    return info
    
def get_state_label(row):
    thre = 0.5
    for op in OP_LIST:
        img_op = row[op]
        if img_op >0.5:
            return op

def get_detected_vehicles(row,thre = 0.1):
    vehi_dict = {}
    for veh in VEH_LIST[:-1]:
        img_veh_prob = row[veh]
        if img_veh_prob > thre:
            vehi_dict[veh] = img_veh_prob
    return vehi_dict
        

        
'''Functions to extract info from the img_detected_veh_info dictionary '''       
def select_no_veh_paths(img_detected_veh_info):
    no_veh_img_paths = []
    for image_path,info in img_detected_veh_info.items():
        veh_list = info['veh_list']
        if not len(veh_list):
            no_veh_img_paths.append(image_path)
    return no_veh_img_paths



'''Functions that summarize the vehicle info from img_detected_veh_info dictionary, return in df for easy usage ''' 
def sum_veh_info(img_detected_veh_info):
    vehs = set()
    image_paths, veh_patterns = [], []
    for image_path,info in img_detected_veh_info.items():
        img_vehs = get_img_vehs(info)
        vehs = add_vehs(vehs,img_vehs)
        pattern = get_pattern(img_vehs)
        veh_patterns.append(pattern)
        image_paths.append(image_path)
    pattern_df = create_pattern_df (image_paths, veh_patterns)
    return (vehs,pattern_df)
    
def get_img_vehs(info):
    vehs = []
    veh_list = info['veh_list']
    if not len(veh_list):
        vehs.append('No_Veh')
    else:
        for veh in veh_list:
            vehs.append(veh)
    return vehs

def add_vehs(vehs,img_vehs):
    for img_veh in img_vehs:
        vehs.add(img_veh)
    return vehs

def get_pattern(img_vehs):
    veh_pattern = img_vehs.pop()
    for veh in img_vehs:
        veh_pattern += ('&'+veh)
    return veh_pattern
        
def create_pattern_df (image_paths, veh_patterns):
    names = get_names_from_paths(image_paths)
    df_dict = {'name' : names,
        'path':image_paths,
              'veh_pattern':veh_patterns}
    df_pattern = pd.DataFrame.from_dict(df_dict)
    return df_pattern

def get_name_from_path(image_path):
    return image_path.split('/')[-1]
    

def get_names_from_paths(image_paths):
    names = []
    for path in image_paths:
        names.append(get_name_from_path(path))
    return names
    
def get_veh_code(veh_name):
    for i,veh in enumerate(VEH_LIST):
        if veh==veh_name:
            return str(i)
        
def get_veh_name(veh_code):
    return VEH_LIST[int(veh_code)]



'''Plot functions'''
def visualize_veh_info(veh_pattern, state_name):
    fig = plt.figure(figsize=(18, 6))
    ax = sns.countplot(data=veh_pattern, x = 'veh_pattern',
                       order=veh_pattern['veh_pattern'].value_counts().iloc[:8].index)
    fig.suptitle('Vehcle Info Summary for State:  '+state_name)
            
def plot_selected_imgs(tot_img_paths,selected_img_idx_list):
    for idx in selected_img_idx_list:
        file_path = tot_img_paths[idx]
        print (file_path)
        img = read_jpg(file_path)
        show_image(img)
        
def plot_imgs(img_paths):
    for img_path in img_paths:
        print (img_path)
        img = read_jpg(img_path)
        show_image(img)   