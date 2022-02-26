#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)
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
import pickle
# Don't generate pyc codes
sys.dont_write_bytecode = True
def read_resize(imname, shape):
    img = cv2.imread(imname)
    img = cv2.resize(img, shape)
    return img

def crop_patch(img, pts):
    pts = np.rint(pts).astype(np.int32)
    x1,y1 = pts[0]
    x2,y2 = pts[2]
    return img[y1:y2, x1:x2, :]


def get_rect_pts(center, size):
    centerx, centery = center
    pts = list()
    pts.append([centerx - size//2, centery - size//2])
    pts.append([centerx + size//2, centery - size//2])
    pts.append([centerx + size//2, centery + size//2])
    pts.append([centerx - size//2, centery + size//2])

    return np.array(pts).astype(np.float32)


def getPatchPoint(img):
    #img size is assumed to be 320, 240
    centerx = random.randint(100,220)
    centery = random.randint(100, 140)
    return np.array([centerx, centery])

def getHomo(pts):
    if len(pts.shape) == 1:
        pts = np.expand_dims(pts, 0)
    ones = np.ones((len(pts), 1))
    return np.concatenate([pts, ones], axis=-1)
def warp_points(pts, H):
    pts = getHomo(pts)
    ptsB = H@pts.T
    ptsB = ptsB[0:2] / ptsB[-1]
    return ptsB.T
class HomographyDataset(keras.utils.Sequence):

    def __init__(self, dirpath, generate=True, transform=None, name="train"):
        
        
        '''imnames = glob.glob(dirpath + "/*.jpg")
        if generate:
            self.info = list()
            num = 500000 if name=="train" else 5000
            for i in range(num):
                if i % 1000 == 0:
                    print("Completed ", i)
                IAimname = random.choice(imnames)
                IA = read_resize(IAimname, (320,240))
                centerA = getPatchPoint(IA)
                ptsA = get_rect_pts(centerA, 128)
                error = np.random.randint(-32, 32, size=(4,2)).astype(np.float32)
            
                H_AB = cv2.getPerspectiveTransform(ptsA, ptsA + error)
                H_BA = np.linalg.inv(H_AB)
                IB = cv2.warpPerspective(IA, H_BA, (320, 240))
                centerB = warp_points(centerA, H_BA)
                centerB = centerB[0]
                ptsB = get_rect_pts(centerB, 128)
                    
                # im1 = cv2.circle(IA, (int(centerA[0]), int(centerA[1])), 2, (0,255,0), -1)
                # im2 = cv2.circle(IB, (int(centerB[0]), int(centerB[1])), 2, (0,255,0), -1)
                # p1 = crop_patch(IA, ptsA)
                # p2 = crop_patch(IB, ptsB)
                # cv2.imwrite("im1.jpg", im1)
                # cv2.imwrite("im2.jpg", im2)
                # cv2.imwrite("p1.jpg", p1)
                # cv2.imwrite("p2.jpg", p2)

                self.info.append([IAimname, ptsA, error, H_AB, H_BA, ptsB, centerA, centerB])
            pickle.dump(self.info, open(f"/vulcanscratch/sonaalk/Stitching/Phase2/Data/homography_data_{name}.pkl", "wb"))
        
        else:
            self.info = pickle.load(open(f"/vulcanscratch/sonaalk/Stitching/Phase2/Data/homography_data_{name}.pkl", "rb"))

        if name == "train":
           self.info = self.info[:10000]'''
        self.info = pickle.load(open(f"/vulcanscratch/sonaalk/Stitching/Phase2/Data/homography_data_{name}.pkl", "rb"))
    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, idx):
        IAimname, ptsA, error, H_AB, H_BA, ptsB, centerA, centerB = self.info[idx]
        IA = read_resize(IAimname, (320,240))
        IB = cv2.warpPerspective(IA, H_BA, (320,240))
        pA = crop_patch(IA, ptsA)
        pB = crop_patch(IB, ptsB)
        # try:
        gt = error / 32.
        x1 =np.zeros((self.batch_size,) + (128,128) + (6,), dtype="float32")
        x2 =np.zeros((self.batch_size,) + (128,128) + (3,), dtype="float32")
        x3 =np.zeros((self.batch_size,) + (128,128) + (3,), dtype="float32")
        #pA = self.transform(pA)
        #pB = self.transform(pB)
        #X = torch.cat([pA,pB], axis=0)
        # except:
        #     import pdb;pdb.set_trace()
            
        return X, gt, ptsA, self.transform2(IA)

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
            #print(x3[j].shape)
            #og_img = img
            #og_img=img/255
            x1[j]=np.concatenate((x2[j],x3[j]),axis=-1)
            #x1[j]=x1[j]/255
            #print(x1[j].shape)
        
        
        y = np.zeros((self.batch_size,8), dtype="float32")
        for j, path in enumerate(batch_target):
            img = np.load(path)
            #print(img)
            y_=np.load(path).reshape(8)
            y[j] =y_
            y[j]=np.asarray(y[j])/32
            #print(y[j].shape)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            #y[j] -= 1
        if self.modeltype=='sup':
            return x1, y
        else:
            return (x1 ,x2),x3
        #return x, y    



def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    
def TrainOperation(  NumEpochs, MiniBatchSize, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """      
    batch_size=MiniBatchSize
    dir_folder_train= BasePath+'Train/'
    dir_folder_val= BasePath+'Val/'
    dir_original_patch_img_train=dir_folder_train+'original_patch/'
    dir_original_patch_img_val=dir_folder_val+'original_patch/'
    dir_homo_patch_img_train=dir_folder_train+'homography_patch/'
    dir_homo_patch_img_val=dir_folder_val+'homography_patch/'
    dir_output_train=dir_folder_train+'output/'
    dir_output_val=dir_folder_val+'output/'
    image_name_train=os.listdir(dir_original_patch_img_train)
    image_name_val=os.listdir(dir_original_patch_img_val)


    og_img_paths_t=[dir_original_patch_img_train+i for i in image_name_train]
    og_img_paths_v=[dir_original_patch_img_val+i for i in image_name_val]

    homo_img_paths_t=[dir_homo_patch_img_train+i for i in image_name_train]
    homo_img_paths_v=[dir_homo_patch_img_val+i for i in image_name_val]

    target_paths_t=[dir_output_train+i +'.npy' for i in image_name_train]
    target_paths_v=[dir_output_val+i +'.npy' for i in image_name_val]

   
    # Predict output with forward pass
    modely=myModel()
    #prLogits, prSoftMax = HomographyModel(ImgPH, ImageSize, MiniBatchSize)
    
    if ModelType.lower()=='sup':
        print('supervised model')
        model=modely.model_('sup')

        cp_path=CheckPointPath+'/supervised/checkpoints'+ datetime.now().strftime("try%Y%m%d-%H%M%S")

        logdir = LogsPath+"/supervised/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        loss=modely.epe_loss
        train_data_generated= DataGenerator(
        batch_size, og_img_paths_t, homo_img_paths_t, target_paths_t,modeltype='sup'
        )
        val_data_generated= DataGenerator(
            batch_size, og_img_paths_v, homo_img_paths_v, target_paths_v,modeltype='sup'
        )
        optimizer=keras.optimizers.SGD(lr=0.005,momentum=0.9)
        #optimizer=keras.optimizers.Adam(lr=0.005,epsilon=1e-08)
    else:
        print('unsupervised model')        
        model=modely.model_('unsup')
        optimizer=keras.optimizers.Adam(lr=0.0001,epsilon=1e-08)
        #loss=modely.photometric_loss
        loss=modely.epe_loss
        cp_path=CheckPointPath+'/unsupervised/checkpoints'+ datetime.now().strftime("try%Y%m%d-%H%M%S")

        logdir = LogsPath+"/unsupervised/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        train_data_generated= DataGenerator(
        batch_size, og_img_paths_t, homo_img_paths_t, target_paths_t,modeltype='unsup'
        )
        val_data_generated= DataGenerator(
            batch_size, og_img_paths_v, homo_img_paths_v, target_paths_v,modeltype='unsup'
        )
    model.compile(optimizer=optimizer,loss=loss,run_eagerly=True)
    if LatestFile is not None:
        model=tf.keras.models.load_model(LatestFile,custom_objects={'customModel':model,'photometric_loss':modely.photometric_loss,'epe_loss':modely.epe_loss})
        #mod.fit(train_data_generated,validation_data=val_data_generated,epochs=100)

    mcp_save = ModelCheckpoint(filepath=cp_path, save_best_only=True, monitor='val_loss', mode='min')
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,update_freq=1)
    # Tensorboard
    # Create a summary to monitor loss tensor
    #tf.summary.scalar('LossEveryIter', loss)
    # tf.summary.image('Anything you want', AnyImg)
    # Merge all summaries into a single operation
    #MergedSummaryOP = tf.summary.merge_all()

    # Setup Saver
    model.fit(train_data_generated,validation_data=val_data_generated,epochs=NumEpochs,callbacks=[mcp_save,tensorboard_callback])

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='../Data/train_homo/', help='Base path of code, Default:../Data/train_homo/')
    Parser.add_argument('--CheckPointPath', default='./tmp', help='Path to save Checkpoints, it is different for superviused and unsupervised  if it is supervised then checkpoints can be found in ./tmp/supervised/checkpoints Default: ./tmp ')
    Parser.add_argument('--ModelType', default='unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=32, help='Size of the MiniBatch to use, Default:32')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='./logs', help='Path to save Logs for Tensorboard, it is different for supervised and unsupervised, if it is supervised then logs can be found in ./logs/supervised/scalar Default=./logs')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    #DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)



    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    #PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    #ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
    #LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, NumClasses)) # OneHOT labels
    
    TrainOperation(  NumEpochs, MiniBatchSize, CheckPointPath,
                   DivTrain, LatestFile, BasePath, LogsPath, ModelType)
        
    
if __name__ == '__main__':
    main()
 