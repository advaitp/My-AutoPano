#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 18:19:33 2022

@author: abhi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 15:18:26 2022

@author: abhi
"""
import os
#dir_parent=os.path.dirname(os.getcwd())
#dir_image_folder= '../Data/Test/'

import cv2
import matplotlib.pyplot as plt
import argparse
import numpy as np
#total_diff_tosave=np.zeros((len(image_name)*100,4,2))
def main(im_name,i,dir_image_folder,ip_type):
    img=cv2.imread(dir_image_folder+im_name)
    img=cv2.resize(img,(320,240))
    m,n,_=img.shape
    
    #m=128*4
    #n=128*4
    for j in range(1):
        patch_size=(128,128)
        m_offset=np.random.randint(patch_size[0]/4,m-patch_size[0]-patch_size[0]/4)
        n_offset=np.random.randint(patch_size[1]/4,n-patch_size[1]-patch_size[1]/4)
        
        orignal_path_points=[[n_offset,m_offset],[n_offset,m_offset+patch_size[0]],[n_offset+patch_size[0],m_offset+patch_size[0]],[n_offset+patch_size[0],m_offset]]
        orignal_path_points=np.array(orignal_path_points)
        '''for i in orignal_path_points:
            cv2.circle(img,(i),2,color=(255, 0, 0),thickness=5)
        '''
        
        pertub_m=np.random.randint(-32,32,size=(4,))
        pertub_n=np.random.randint(-32,32,size=(4,))
        pertub=[pertub_n,pertub_m]
        pertub=np.array(pertub).T
        
        
        projected_points=orignal_path_points+pertub
        
        #np.random.randint(20)
        
        '''for i in projected_points:
            cv2.circle(img,(i),2,color=(0, 0, 255),thickness=5)'''
        
        #plt.imshow(img)
        #plt.show()
        H=cv2.getPerspectiveTransform(np.float32(orignal_path_points),np.float32(projected_points))
        H_=np.linalg.inv(H)
        
        diff_=pertub
        
        
        img_=cv2.warpPerspective(img,H_,(n,m))
        #plt.imshow(img[orignal_path_points[0,1]:orignal_path_points[2,1],orignal_path_points[0,0]:orignal_path_points[2,0]])
        if not os.path.isdir('../Data/train_homo/'+ip_type+'/original_patch'):
            os.makedirs('../Data/train_homo'+ip_type+'/original_patch')
        if not os.path.isdir('../Data/train_homo/'+ip_type+'/homography_patch'):
            os.makedirs('../Data/train_homo'+ip_type+'/homoraphy_patch')
        if not os.path.isdir('../Data/train_homo/'+ip_type+'/output'):
            os.makedirs('../Data/train_homo'+ip_type+'/output')
        cv2.imwrite('../Data/train_homo/'+ip_type+'/original_patch/'+str(j)+'_'+im_name,img[orignal_path_points[0,1]:orignal_path_points[2,1],orignal_path_points[0,0]:orignal_path_points[2,0]])
        #cv2.waitKey(0)
        #plt.show()
        #plt.imshow(img_)
        #plt.show()
        #plt.imshow(img_[orignal_path_points[0,1]:orignal_path_points[2,1],orignal_path_points[0,0]:orignal_path_points[2,0]])
        cv2.imwrite('../Data/train_homo/'+ip_type+'/homography_patch/'+str(j)+'_'+im_name,img_[orignal_path_points[0,1]:orignal_path_points[2,1],orignal_path_points[0,0]:orignal_path_points[2,0]])
        #cv2.waitKey(0)
        #plt.show()
        im_no=im_name.split(',')
        np.save('../Data/train_homo'+ip_type+'/output/'+str(j)+'_'+im_no[0],diff_)
        #total_diff_tosave[i,:,:]=diff_
    
#import torch
#torch.cuda.is_available()
    
    
if __name__ == '__main__':
    i=0
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath_train', default='../Data/Train', help='Path of images that needs to be augmented, Default:../Data/train_homo/')
    Parser.add_argument('--DataPath_test', default='../Data/Test', help='Path of images that needs to be augmented, Default:../Data/train_homo/')
    Parser.add_argument('--DataPath_val', default='../Data/Val', help='Path of images that needs to be augmented, Default:../Data/train_homo/')
    Args = Parser.parse_args()
    dir_image_folder= Args.DataPath_train
    
    image_name=os.listdir(dir_image_folder)
    image_name.sort()
    for im_name in image_name:
        
        main(im_name,i,'train')
        #print(i)
        i=i+1
        #break
    dir_image_folder= Args.DataPath_test
    image_name=os.listdir(dir_image_folder,dir_image_folder)
    image_name.sort()
    for im_name in image_name:
        
        main(im_name,i,dir_image_folder,'test')
        #print(i)
        i=i+1
        #break
    dir_image_folder= Args.DataPath_val
    image_name=os.listdir(dir_image_folder,dir_image_folder)
    image_name.sort()
    for im_name in image_name:
        
        main(im_name,i,"val")
        #print(i)
        i=i+1
        #break