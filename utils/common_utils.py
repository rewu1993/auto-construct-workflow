from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np

def select_short_period_df(df,start_date,end_date):
    time_constrain = (df['datetime']>start_date) & (df['datetime']<end_date)
    return df[time_constrain]

"""Label Transfer Function"""
def label_trans(r):
    l1 = [2, 3, 4, 5, 6]
    l2 = [7]
    if r in l1:
        r+=1
    elif r in l2:
        r+=2
    return r

def cls_label2_ori(res):
    pred = []
    for r in res:
        pred.append(label_trans(r))
    return pred

"""Images"""
def read_jpg(filename,show_shape = False):
    """A function that read jpg file
        Args: 
            filename, a string for image path
        Returns:
            image in numpy format, 
    """
    image = imread(filename)
    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]
    if show_shape:
        print ("Image shape is %s"%(image.shape,))
    return image

def show_image(img,figsize = (8,8)):
    """A function that visualize image
        Args:
            img: numpy array with dimension X*X*3 (or 1)
            figsize: tuple with dimension 1*2 that defines the image visualization size
        Return:
            None, image will be visualized
    """
    fig = plt.figure(figsize=figsize)
    if len(img.shape) == 2:
        img = img[:,:,np.newaxis]
    shape = img.shape
    if len(shape)!=3 or shape[2] not in [1,3]:
        raise TypeError("Input image has dimension %s, which can not be visualized!"%(shape,))
    if shape[2]==1:
        plt.imshow(img[:,:,0],cmap="bone")
    else:
        plt.imshow(img)
    plt.show()
    

# find the overlapping ratio
def overlap_ratio(df,label_name='label_binary',pred_name = 'predictions'): 
    pred_list = np.array(df[pred_name])
    label_list = np.array(df[label_name])

    bool_mask = pred_list==label_list
    overlap_ratio = bool_mask.sum()/len(bool_mask)
    return overlap_ratio