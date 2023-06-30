# -*- coding: utf-8 -*-

from __future__ import print_function

from collections import namedtuple
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import random
import os


#############################
    # global variables #
#############################
root_dir  = "CityScapes/"

label_dir = os.path.join(root_dir, "gtFine")
train_dir = os.path.join(label_dir, "train")
val_dir   = os.path.join(label_dir, "val")
test_dir  = os.path.join(label_dir, "test")

# create dir for label index
label_idx_dir = os.path.join(root_dir, "Labeled_idx")
train_idx_dir = os.path.join(label_idx_dir, "train")
val_idx_dir   = os.path.join(label_idx_dir, "val")
test_idx_dir  = os.path.join(label_idx_dir, "test")
for dir in [train_idx_dir, val_idx_dir, test_idx_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

train_file = os.path.join(root_dir, "train.csv")
val_file   = os.path.join(root_dir, "val.csv")
test_file  = os.path.join(root_dir, "test.csv")

color2index = {}

Label = namedtuple('Label', [
                   'name', 
                   'id', 
                   'trainId', 
                   'category', 
                   'categoryId', 
                   'hasInstances', 
                   'ignoreInEval', 
                   'color'])

labels = [
    #       name                        id    trainId   category               catId     hasInstances   ignoreInEval   color
    Label(  'Black Background'        ,  0 ,      0 , 'Black Background'        , 1       , True        , False    , (  127, 127, 127) ),
    Label(  'Liver'                   ,  1 ,      1 , 'Liver'                   , 2       , True        , False    , (  255, 114, 114) ),
    Label(  'Gastrointestinal Tract'  ,  2 ,      2 , 'Gastrointestinal Tract'  , 3       , True        , False    , (  231,  70, 156) ),
    Label(  'Fat'                     ,  3 ,      3 , 'Fat'                     , 4       , True        , False    , (  186, 183,  75) ),
    Label(  'Grasper'                 ,  4 ,      4 , 'Instrument'              , 5       , True        , False    , (  170, 255,   0) ),
    Label(  'Connective Tissue'       ,  5 ,      5 , 'Connective Tissue'       , 6       , True        , False    , (  255,  85,   0) ),
    Label(  'Abdominal Wall'          ,  6 ,      6 , 'Abdominal Wall'          , 0       , True        , False    , (  210, 140, 140) ),
    Label(  'Blood'                   ,  7 ,      7 , 'Blood'                   , 7       , True        , False    , (  255,   0,   0) ),
    Label(  'Cystic Duct'             ,  8 ,      8 , 'Cystic Duct'             , 8       , True        , False    , (  255, 255,   0) ),
    Label(  'L-hook Electrocautery'   ,  9 ,      9 , 'Instrument'              , 5       , True        , False    , (  169, 255, 184) ),
    Label(  'Hepatic Vein'            , 10 ,     10 , 'Hepatic Vein'            , 11      , True        , False    , (    0,  50, 128) ),
    Label(  'Gallbladder'             , 11 ,     11 , 'Gallbladder'             , 10      , True        , False    , (  255, 160, 165) ),
    Label(  'Liver Ligament'          , 12 ,     12 , 'Liver Ligament'          , 12      , True        , False    , (  111,  74,   0) ),
]


def parse_label():
    # change label to class index
    color2index[(0,0,0)] = 0  # add an void class 
    for obj in labels:
        if obj.ignoreInEval:
            continue
        idx   = obj.trainId
        label = obj.name
        color = obj.color
        color2index[color] = idx

    # parse train, val, test data    
    for label_dir, index_dir, csv_file in zip([train_dir, val_dir, test_dir], [train_idx_dir, val_idx_dir, test_idx_dir], [train_file, val_file, test_file]):
        f = open(csv_file, "w")
        f.write("img,label\n")
        for city in os.listdir(label_dir):
            city_dir = os.path.join(label_dir, city)
            city_idx_dir = os.path.join(index_dir, city)
            data_dir = city_dir.replace("gtFine", "leftImg8bit")
            if not os.path.exists(city_idx_dir):
                os.makedirs(city_idx_dir)
            for filename in os.listdir(city_dir):
                if 'color' not in filename:
                    continue
                lab_name = os.path.join(city_idx_dir, filename)
                img_name = filename.split("gtFine")[0] + "leftImg8bit.png"
                img_name = os.path.join(data_dir, img_name)
                f.write("{},{}.npy\n".format(img_name, lab_name))

                if os.path.exists(lab_name + '.npy'):
                    print("Skip %s" % (filename))
                    continue
                print("Parse %s" % (filename))
                img = os.path.join(city_dir, filename)
                img = scipy.misc.imread(img, mode='RGB')
                height, weight, _ = img.shape
        
                idx_mat = np.zeros((height, weight))
                for h in range(height):
                    for w in range(weight):
                        color = tuple(img[h, w])
                        try:
                            index = color2index[color]
                            idx_mat[h, w] = index
                        except:
                            # no index, assign to void
                            idx_mat[h, w] = 13
                idx_mat = idx_mat.astype(np.uint8)
                np.save(lab_name, idx_mat)
                print("Finish %s" % (filename))


'''debug function'''
def imshow(img, title=None):
    try:
        img = mpimg.imread(img)
        imgplot = plt.imshow(img)
    except:
        plt.imshow(img, interpolation='nearest')

    if title is not None:
        plt.title(title)
    
    plt.show()


if __name__ == '__main__':
    parse_label()
