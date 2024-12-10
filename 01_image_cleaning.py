
#libraries
import os 
import random 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
from PIL import Image
from skimage import img_as_float 
from IPython.display import display

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = ROOT_PATH + '/data/'

img_data = {} 

for folder in os.listdir(DATA_DIR): 
    img_path = DATA_DIR + folder + '/'
    img_data[folder] = [img_path + img for img in os.listdir(img_path)]
    

#Sample some images 
for i, folder in enumerate(img_data.keys()): 
    print(folder)
    sample_size = 10 
    ims = [Image.open(x) for x in random.sample(img_data[folder], sample_size)]
    im_size = 128 
    new_im = Image.new('RGB', (im_size*sample_size, im_size))
    x_offset = 0 
    for i in ims: 
        i.thumbnail((im_size, im_size))
        new_im.paste(i, (x_offset, 0))
        x_offset += i.size[0]
   
    display(new_im) 
    

