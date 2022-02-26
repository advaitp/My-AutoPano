#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
from datetime import datetime
import tensorflow as tf
import tensorflow.keras as keras 
from Network.Network import *

from keras.callbacks import ModelCheckpoint

#from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import os
import tensorflow as tf
import cv2
import sys
import os
import glob
#import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
#from Network.Network import HomographyModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from io import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
import tensorflow as tf
import cv2
import os
import sys
import glob
#import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
#from Network.Network import HomographyModel
from Misc.MiscUtils import *
import numpy as np
import time
import argparse
import shutil
from io import StringIO
import string
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
from PIL import Image

# Don't generate pyc codes
sys.dont_write_bytecode = True

def SetupAll(BasePath):
    """
    Inputs: 
    BasePath - Path to images
    Outputs:
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    """   
    # Image Input Shape
    ImageSize = [32, 32, 3]
    DataPath = []
    NumImages = len(glob.glob(BasePath+'*.jpg'))
    SkipFactor = 1
    for count in range(1,NumImages+1,SkipFactor):
        DataPath.append(BasePath + str(count) + '.jpg')

    return ImageSize, DataPath

def unnormalize(img):
    img = img.permute(1,2,0).cpu().numpy()
    img = img - img.min()
    img = img / img.max()
    return (img*255).astype(np.uint8)

def save_visualizations(Images, H_gt, H_pred, ptsA, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    for j in range(Images.shape[0]):
        
        img = unnormalize(Images[j])
        
        base_pts= (ptsA[j]).cpu().numpy().reshape(-1,1,2).astype(np.int32)
        
        gt_pts= (H_gt[j]*32).cpu().numpy().reshape(-1,1,2).astype(np.int32)
        pred_pts= (H_pred[j]*32).cpu().numpy().reshape(-1,1,2).astype(np.int32)

        gt_pts = gt_pts + base_pts
        pred_pts = pred_pts + base_pts

        img = cv2.polylines(img.copy(), [gt_pts], True, (0,0,255), 2)
        img= cv2.polylines(img, [pred_pts], True, (255,0,0), 2)

        cv2.imwrite(f"{save_path}/{j}.jpg", img) 
def ReadImages(ImageSize, DataPath):
    """
    Inputs: 
    ImageSize - Size of the Image
    DataPath - Paths of all images where testing will be run on
    Outputs:
    I1Combined - I1 image after any standardization and/or cropping/resizing to ImageSize
    I1 - Original I1 image for visualization purposes only
    """
    
    ImageName = DataPath
    
    I1 = cv2.imread(ImageName)
    
    if(I1 is None):
        # OpenCV returns empty list if image is not read! 
        print('ERROR: Image I1 cannot be read')
        sys.exit()
        
    ##########################################################################
    # Add any standardization or cropping/resizing if used in Training here!
    ##########################################################################

    I1S = iu.StandardizeInputs(np.float32(I1))

    I1Combined = np.expand_dims(I1S, axis=0)

    return I1Combined, I1

class DataGenerator(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, og_img_paths,homo_img_paths ,target,modeltype='sup'):
        self.batch_size = batch_size
        #self.img_size = img_size
        self.og_paths = og_img_paths
        self.homo_paths=homo_img_paths
        self.target = target
        self.modeltype=modeltype

    def __len__(self):
        return len(self.og_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_og_img_paths = self.og_paths[i : i + self.batch_size]
        batch_homo_img_paths=self.homo_paths[i : i + self.batch_size]
        batch_target = self.target[i : i + self.batch_size]
        x1 =np.zeros((self.batch_size,) + (128,128) + (2,), dtype="float32")
        x2 =np.zeros((self.batch_size,) + (128,128) + (1,), dtype="float32")
        x3 =np.zeros((self.batch_size,) + (128,128) + (1,), dtype="float32")
        for j, path in enumerate(zip(batch_og_img_paths,batch_homo_img_paths)):
            og_img = load_img(path[0], target_size=(128,128),grayscale=True)
            x2[j]=img_to_array(og_img)
            x2[j]=x2[j]/255
            homo_img = load_img(path[1], target_size=(128,128),grayscale=True)
            x3[j]=img_to_array(homo_img)
            x3[j]=x3[j]/255

            x1[j]=np.concatenate((x2[j],x3[j]),axis=-1)

        
        
        y = np.zeros((self.batch_size,8), dtype="float32")
        for j, path in enumerate(batch_target):
            img = np.load(path)

            y_=np.load(path).reshape(8)
            y[j] =y_
            y[j]=np.asarray(y[j]/32)

        if self.modeltype=='sup':
            return x1, y
        else:
            return (x1,x2),y 
 
             
def mse_loss(y_pred,y_true):
    
    return tf.math.reduce_mean(tf.math.square(tf.math.subtract(y_true,y_pred)))
def TestOperation(ModelPath, BasePath, ModelType,LatestFile):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    ImageSize is the size of the image
    ModelPath - Path to load trained model from
    DataPath - Paths of all images where testing will be run on
    LabelsPathPred - Path to save predictions
    Outputs:
    Predictions written to ./TxtFiles/PredOut.txt
    """
    # ModelPath
    MiniBatchSize=1
    batch_size=MiniBatchSize
    dir_folder_train= BasePath+'Test/'

    dir_original_patch_img_train=dir_folder_train+'original_patch/'

    dir_homo_patch_img_train=dir_folder_train+'homography_patch/'

    dir_output_train=dir_folder_train+'output/'

    image_name_train=os.listdir(dir_original_patch_img_train)



    og_img_paths_t=[dir_original_patch_img_train+i for i in image_name_train]


    homo_img_paths_t=[dir_homo_patch_img_train+i for i in image_name_train]


    target_paths_t=[dir_output_train+i +'.npy' for i in image_name_train]

    modely=myModel()
    error=0
    loss_value=[]
    if ModelType.lower()=='sup':
        print('supervised model')
        model=modely.model_('sup')

        cp_path=ModelPath+'/supervised/checkpoints'+ datetime.now().strftime("try%Y%m%d-%H%M%S")
        model=tf.keras.models.load_model(LatestFile,custom_objects={'customModel':model,'photometric_loss':modely.photometric_loss,'epe_loss':modely.epe_loss})
        #logdir = LogsPath+"/supervised/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        loss=modely.epe_loss
        test_data_generated= DataGenerator(
        batch_size, og_img_paths_t, homo_img_paths_t, target_paths_t,modeltype='sup'
        )
    else:
        print('unsupervised model')        
        model=modely.model_('unsup')
        optimizer=keras.optimizers.Adam(lr=0.0001,epsilon=1e-08)
        loss=modely.photometric_loss
        model=tf.keras.models.load_model(LatestFile,custom_objects={'customModel':model,'photometric_loss':modely.photometric_loss,'epe_loss':modely.epe_loss})
        #cp_path=CheckPointPath+'/supervised/checkpoints'+ datetime.now().strftime("try%Y%m%d-%H%M%S")
        newmodel = keras.Model(inputs=model.input, outputs=model.get_layer('dense_47').output)
        model=newmodel
        #logdir = LogsPath+"/unsupervised/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        test_data_generated= DataGenerator(
        batch_size, og_img_paths_t, homo_img_paths_t, target_paths_t,modeltype='unsup'
        )
        
        #print(test_data_generated[0])
    for i,j in test_data_generated:
        x=model.predict(i)
        #i=i[0,:,:,:3]*255
        #cv2.imshow('img',i)
        #print(i.shape)
        
        #print(1)
        #print(tf.shape(x))
        #print(tf.shape(j))
        #print(x,j)
        error=mse_loss(x,j)
        #print(error.numpy())
        loss_value.append(error.numpy())
        break
    print('epe_test:',sum(loss_value)*32/len(loss_value))
        #loss=
    img_file="9.jpg"
    img=cv2.imread('../Data/Test/'+img_file)
    img=cv2.resize(img,(320,240))

    img1=img.copy()
    #img1_t=tf.cast(img1,tf.float32)
    #img1_t=tf.expand_dims(img1_t,0)
    img_collage=np.zeros([128,128,2])
    img_homo=np.array(Image.open('../Data/train_homo/Test/homography_patch/0_'+img_file).convert('L'))
    img_og=np.array(Image.open('../Data/train_homo/Test/original_patch/0_'+img_file).convert('L'))
    output=np.load('../Data/train_homo/Test/output/0_'+img_file+'.npy')
    output=np.reshape(output,[-1])
    img_collage[:,:,0]=img_og/255
    img_collage[:,:,1]=img_homo/255
    img1_t1=tf.cast(img_og,tf.float32)
    img1_t1=tf.expand_dims(img1_t1,0)
    img1_t=tf.cast(img_collage,tf.float32)
    img1_t=tf.expand_dims(img1_t,0)
    points=[[42,42],[42,170],[170,170],[170,42]]
    points=np.reshape(points,[-1])
    print(img1_t.shape)
    if ModelType=='sup':
        x=model.predict(img1_t)
    else:
        x=model.predict((img1_t,img1_t1))
    #x=model.predict(img1_t)
    final_points_t=points+output
    print(x*32)
    print(output)
    final_points_p=points+x[0,:]*32
    final_points_t=np.reshape(final_points_t,[4,2])
    
    final_points_p=np.reshape(final_points_p,[4,2])
    final_points_p=final_points_p.astype(np.int32)
    final_points_t=final_points_t.astype(np.int32)
    img = cv2.polylines(img1, [final_points_p], True, (0,0,255), 2)
    img= cv2.polylines(img, [final_points_t], True, (255,0,0), 2)
    cv2.imwrite('output2.jpg',img)

def ReadLabels(LabelsPathTest, LabelsPathPred):
    if(not (os.path.isfile(LabelsPathTest))):
        print('ERROR: Test Labels do not exist in '+LabelsPathTest)
        sys.exit()
    else:
        LabelTest = open(LabelsPathTest, 'r')
        LabelTest = LabelTest.read()
        LabelTest = map(float, LabelTest.split())

    if(not (os.path.isfile(LabelsPathPred))):
        print('ERROR: Pred Labels do not exist in '+LabelsPathPred)
        sys.exit()
    else:
        LabelPred = open(LabelsPathPred, 'r')
        LabelPred = LabelPred.read()
        LabelPred = map(float, LabelPred.split())
        
    return LabelTest, LabelPred

        
def main():
    """
    Inputs: 
    None
    Outputs:
    Prints out the confusion matrix with accuracy
    """

    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='./tmp', help='Path to save Checkpoints, it is different for superviused and unsupervised  if it is supervised then checkpoints can be found in ./tmp/supervised/checkpoints Default: ./tmp ')
    #Parser.add_argument('--ModelPath', dest='ModelPath', default='/home/chahatdeep/Downloads/Checkpoints/144model.ckpt', help='Path to load latest model from, Default:ModelPath')
    Parser.add_argument('--BasePath', default='../Data/train_homo/', help='Base path of code, Default:../Data/train_homo/')
    #Parser.add_argument('--BasePath', dest='BasePath', default='/home/chahatdeep/Downloads/aa/CMSC733HW0/CIFAR10/Test/', help='Path to load images from, Default:BasePath')
    #Parser.add_argument('--LabelsPath', dest='LabelsPath', default='./TxtFiles/LabelsTest.txt', help='Path of labels file, Default:./TxtFiles/LabelsTest.txt')
    Parser.add_argument('--ModelType', default='unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Args = Parser.parse_args()
    CheckPointPath = Args.CheckPointPath
    BasePath = Args.BasePath
    #LabelsPath = Args.LabelsPath
    ModelType=Args.ModelType
    # Setup all needed parameters including file reading
    #ImageSize, DataPath = SetupAll(BasePath)
    if ModelType.lower()=='sup':
        CheckPointPath=CheckPointPath+'/supervised/'
    else:
        CheckPointPath=CheckPointPath+'/unsupervised/'
    # Define PlaceHolder variables for Input and Predicted output
    #ImgPH = tf.placeholder('float', shape=(1, ImageSize[0], ImageSize[1], 3))
    #LabelsPathPred = './TxtFiles/PredOut.txt' # Path to save predicted labels

    LatestFile = FindLatestModel(CheckPointPath)
    TestOperation( CheckPointPath, BasePath,ModelType,LatestFile)

    # Plot Confusion Matrix
    #LabelsTrue, LabelsPred = ReadLabels(LabelsPath, LabelsPathPred)
    #ConfusionMatrix(LabelsTrue, LabelsPred)
     
if __name__ == '__main__':
    main()
 
