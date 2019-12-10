from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np

''''Functions to deal with images'''
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
    