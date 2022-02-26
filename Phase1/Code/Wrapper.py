#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:

import numpy as np
import cv2
from glob import glob
import os
import matplotlib.pyplot as plt 
from numpy.linalg import inv, svd
from skimage.feature import peak_local_max
# Add any python libraries here

def showFeatures(features, num) :
	numfeatures = len(features)
	fig, ax = plt.subplots(10, 10)
	feature_num = 0
	for r in range(10) : 
		for c in range(10) :
			ax[r, c].axis('off')
			feat = features[feature_num].reshape((8,8))
			ax[r, c].imshow(feat, cmap='gray')
			feature_num += 1

	fig.tight_layout()
	plt.axis('off')
	plt.savefig(f"fd{num}.png")
	plt.close('all')

def featureDesc(img, ipoints, num) : 
	im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	h, w = im.shape
	
	features = []
	nipoints = []
	for point in ipoints :
		if point[0]+20 < w-1 and point[0]-20 >= 0 and point[1]+20 < h-1 and point[1]-20 >= 0 :
			crop = im[point[1]-20:point[1]+20, point[0]-20:point[0]+20]
			g_blur = cv2.GaussianBlur(crop, ksize=(5,5), sigmaX=0, sigmaY=0)
			res = cv2.resize(g_blur, None, fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
			res = res.reshape((64,1))
			mean = np.mean(res)
			sd = np.std(res)
			res = (res-mean)/(sd+10e-7)
			features.append(res)
			nipoints.append(point)

	showFeatures(features, num)
	return features, nipoints

def ANMS(im, option, num) :
	if option == 1 :
		flag = True
		im1 = im.copy()
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		Nbest = 500
		Ncorners = 2*Nbest
		corners = cv2.goodFeaturesToTrack(gray,Ncorners,0.01,5)
		if type(corners) == type(None) : 
			flag = False
			return im, im, [], flag

		Ns = len(corners)
		ipoints = []

		for i in range(Ns) :
			xi, yi= int(corners[i][0][0]), int(corners[i][0][1])
			ipoints.append([xi, yi])
			cv2.circle(im1, (xi, yi), 2, (0, 0, 255), -1)

		cv2.imwrite(f"anms{num}.png", im1)
		print(f'Length of points Shi Tomasi : {len(ipoints)}')

	else :
		flag = True
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		im1 = im.copy()
		im2 = im.copy()
		cscore = cv2.cornerHarris(gray,2,3,0.001)
		lmax = peak_local_max(cscore, min_distance=8)
		if type(lmax) == type(None) : 
			flag = False
			return im, im, [], flag

		Ns = len(lmax)
		Nbest = 500
		Ncorners = 2*Nbest

		for i in range(Ns) :
			xi, yi = lmax[i][0], lmax[i][1]
			cv2.circle(im1, (int(yi), int(xi)), 2, (0, 0, 255), -1)

		cv2.imwrite(f"corner{num}.png", im1)
		
		r = [float('inf') for i in range(Ns)]
		r = np.array(r)
		x = np.zeros((Ns,))
		y = np.zeros((Ns,))
		ED = 0
		for i in range(Ns) :
			for j in range(Ns) :
				xi, yi, xj, yj = lmax[i][0], lmax[i][1], lmax[j][0], lmax[j][1]
				if cscore[xj][yj] > cscore[xi][yi] :
					ED = (xi - xj)**2 + (yi - yj)**2
				if ED < r[i] :
					r[i] = ED
					x[i] = lmax[j][0]
					y[i] = lmax[j][1]

		r = np.argsort(r)
		r = np.flip(r)
		idx = r[:Nbest]

		if Ns < Nbest : Nbest = Ns
		ipoints = []

		for i in range(Nbest):
			ipoints.append([int(y[idx[i]]), int(x[idx[i]])])
			cv2.circle(im2, (int(y[idx[i]]), int(x[idx[i]])), 2, (0, 0, 255), -1)

		cv2.imwrite(f"anms{num}.png", im2)

		print(f'Length of points Corner Harris : {len(ipoints)}')

	return im, im1, ipoints, flag

def findSSD(feature1, feature2) :
	ssd = np.sum((feature1-feature2)**2, axis=0)
	return ssd

def equateSize(im1, im2) :
	img1 = im1.copy()
	img2 = im2.copy()

	img1 = cropImage(img1)
	img2 = cropImage(img2)
	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]

	h = max(h1, h2)
	w = max(w1, w2) 

	img1 = cv2.resize(img1, (h,w))
	img2 = cv2.resize(img2, (h,w))

	return img1, img2

def drawMatches(im1, keypoints1, im2, keypoints2) :
	img1 = im1.copy()
	img2 = im2.copy()

	img1, img2 = equateSize(img1, img2)
	new_img = np.concatenate((img1, img2), axis=1)
	numkeypoints = len(keypoints1)

	r = 4
	thickness = 1
	
	for i in range(numkeypoints) :

		end1 = keypoints1[i]
		end2 = (keypoints2[i][0]+img1.shape[1], keypoints2[i][1])

		cv2.line(new_img, end1, end2, (0,255,255), thickness)
		cv2.circle(new_img, end1, r, (0,0,255), thickness)
		cv2.circle(new_img, end2, r, (0,255,0), thickness)

	return new_img

def featureMatch(im1, im2, num) :

	keypoints1 = []
	keypoints2 = []

	newfeatures1 = []
	newfeatures2 = []

	matchRatio = 0.8
	anmsc1, corner1, ipoints1, flag1 = ANMS(im1, 1, num)
	features1, nipoints1 = featureDesc(im1, ipoints1, num)

	anmsc2, corner2, ipoints2, flag2 = ANMS(im2, 1, num+1)
	features2, nipoints2 = featureDesc(im2, ipoints2, num+1)

	keypoints1 = []
	keypoints2 = []
	if len(features2) == 0 or len(features1) == 0 :
		return keypoints1, keypoints2, False

	for i, feat1 in enumerate(features1) :
		difference = np.sum(abs(features2-feat1), axis=1)
		dsort = np.sort(difference, axis=0)
		indexes = np.argsort(difference, axis=0)

		# print(dsort.shape)
		mratio = int(dsort[0][0])/(int(dsort[1][0])+1e-3)
		if mratio < matchRatio :
			keypoints1.append(nipoints1[i])
			keypoints2.append(nipoints2[indexes[0][0]])

	
	# imMatches = drawMatches(im1, keypoints1, im2, keypoints2)
	# cv2.imwrite(f"matching{num+1}.png", imMatches)

	return keypoints1, keypoints2, (flag1 or flag2)


def rejectOutliers(keypoints1, keypoints2, im1, im2, num) :
	sample_size = len(keypoints1)
	all_indices = np.arange(sample_size)
	maxinliers = 0
	numiterations = 2000
	thresh = 5000
	n = 4
	flag = True
	
	HomoG = np.zeros((3,3))
	kpts1, kpts2 = [], []

	finalInliers = []
	for iteration in range(numiterations) :
		np.random.shuffle(all_indices)
		indices = all_indices[:n]
		pi, pii = [], []

		for idx in indices :
			pi.append(keypoints1[idx])
			pii.append(keypoints2[idx])

		pi = np.array(pi)
		pii = np.array(pii)
		H, status = cv2.findHomography(pi, pii)
		inliers = 0
		inliersP = []

		if H is not None :
			for kp1, kp2 in list(zip(keypoints1, keypoints2)) :
				kpm2 = np.array([kp2[0], kp2[1], 1]).T
				kpm1 = np.array([kp1[0], kp1[1], 1]).T
				ssd = findSSD(kpm2, np.dot(H, kpm1))
				# print(ssd)
				
				if ssd < thresh :
					inliers += 1
					inliersP.append([kp1, kp2])

			if maxinliers < inliers :
				maxinliers = inliers
				HomoG = H
				finalInliers = inliersP

	kpts1 = [item[0] for item in finalInliers]
	kpts2 = [item[1] for item in finalInliers]

	print(f'Length of inliers {len(kpts1)} and {len(kpts2)}')
	if len(kpts1) < 4 or len(kpts2) < 4 :
		return [], HomoG, False
	HomoG, mask = cv2.findHomography(np.array(kpts1), np.array(kpts2))

	# imMatches = drawMatches(im1, kpts1, im2, kpts2)
	# cv2.imwrite(f"RANSAC{num}.png", imMatches)

	return finalInliers, HomoG, flag 

def stitchImage(wimg1, wimg2) :
	for i in range(wimg2.shape[0]) :
		for j in range(wimg2.shape[1]) :
			if wimg1[i, j].any() > 0 :
				wimg2[i, j] = wimg1[i,j]
	
	return wimg2
	
def transform(img_list, counter) :
	img1 = img_list[0]

	for i in range(1,len(img_list)) :
		img2 = img_list[i]
		keypoints1, keypoints2, flag = featureMatch(img1, img2, i+counter)
		print(f'Keypoints after matching : {len(keypoints1)}')
		if not flag : 
			print(f'Found less features or not a good match')
			continue
			
		if len(keypoints1) < 4 or len(keypoints2) < 4 : 
			print(f'Found less than 4 features')
			continue

		finalInliers, Hbest, flag = rejectOutliers(keypoints1, keypoints2, img1, img2, i+counter)
		if not flag or len(finalInliers) < 4 :
			continue
		# Hbest, mask = cv2.findHomography(np.array(keypoints1), np.array(keypoints2), cv2.RANSAC)
		# keypoints1, keypoints2, Hbest = ransac(np.array(keypoints1), np.array(keypoints2))
		
		h1, w1 = img1.shape[0], img1.shape[1]
		h2, w2 = img2.shape[0], img2.shape[1]

		points1 = np.array([[0,0], [w1,0], [0,h1],[w1,h1]])
		points2 = np.array([[0,0], [w2,0], [0,h2],[w2,h2]])

		points1 = points1.reshape(-1,1,2).astype(np.float32)
		if type(Hbest) == type(None) : 
			print('None')
			continue

		wpoints1 = cv2.perspectiveTransform(points1,  Hbest).reshape(-1,2)
		
		rpoints = np.concatenate((wpoints1, points2), axis=0)

		xmin, ymin = int(np.min(rpoints, axis=0)[0]), int(np.min(rpoints, axis=0)[1])
		xmax, ymax = int(np.max(rpoints, axis=0)[0]), int(np.max(rpoints, axis=0)[1])
	
		Htrans = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]]).astype(float)

		Hbest = Htrans@Hbest

		height = ymax-ymin
		width = xmax-xmin
		size = (height,width)
		
		wimg1 = cv2.warpPerspective(img1, Hbest, (2000,2000)) 
		wimg2 = cv2.warpPerspective(img2, Htrans, (2000,2000)) 

		mask1 = cv2.threshold(wimg1, 0, 255, cv2.THRESH_BINARY)[1]
		kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
		mask1 = cv2.morphologyEx(mask1, cv2.MORPH_ERODE, kernel1)
		wimg1[mask1==0] = 0

		mask2 = cv2.threshold(wimg2, 0, 255, cv2.THRESH_BINARY)[1]
		kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
		mask2 = cv2.morphologyEx(mask2, cv2.MORPH_ERODE, kernel2)
		wimg2[mask2==0] = 0
		
		img1 = stitchImage(wimg1, wimg2)

		cv2.imwrite(f"pano{i+counter}.jpg", img1)

	return img1

def cropImage(image) :
	h, w = image.shape[:2]
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	first_passc = True
	first_passr = True
	pixelscol = np.sum(gray, axis=0).tolist()
	nresult = image
	for index, value in enumerate(pixelscol):
		if value == 0:
			continue
		else:
			ROI = image[0:h, index:index+1]
			if first_passc:
				result = image[0:h, index+1:index+2]
				first_passc = False
				continue
			result = np.concatenate((result, ROI), axis=1)

	
	gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
	pixelsrow = np.sum(gray, axis=1).tolist()
	h, w = result.shape[:2]
	for index, value in enumerate(pixelsrow):
		if value == 0:
			continue
		else:
			ROI = result[index:index+1, 0:w]
			if first_passr:
				nresult = result[index+1:index+2, 0:w]
				first_passr = False
				continue
			nresult = np.concatenate((nresult, ROI), axis=0)

	return nresult


def main():
	# Add any Command Line arguments here
	# Parser = argparse.ArgumentParser()
	# Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
	
	# Args = Parser.parse_args()
	# NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""
	img_dir = "C://Users//Advait//Desktop//Course//CMSC733//YourDirectoryID_p1//YourDirectoryID_p1//Phase1//Data//Test//TestSet4"
	inOrder = False
	warped = True
	img_list = []
	img_name=[]
	for imgp in glob(img_dir+'/*.jpg') :
		img = cv2.imread(imgp)
		img_list.append(img)

	print(len(img_list))

	if warped :
		if not inOrder :
			img_list = img_list[:]
			n = len(img_list)
			
			anchor = n//2
			flist = []
			left_imgs = img_list[:anchor+1]
			
			limg = transform(left_imgs, 0)
			flist.append(limg)

			right_imgs = img_list[anchor:]
			right_imgs.reverse()
			
			rimg = transform(right_imgs, anchor)
			flist.append(rimg)

			fimg = transform(flist, anchor+len(right_imgs))
			fimg = cropImage(fimg)
			cv2.imwrite('mypano.png', fimg)	 

		else :
			img_list = img_list[:]
			n = len(img_list)
			all_imgs = img_list
		
			fimg = transform(all_imgs, 0)
			fimg = cropImage(fimg)
			cv2.imwrite('mypano.png', fimg)	 

	else :
		n = len(img_list)
		
		for i in range(0,len(img_list)-1) :
			img1, img2 = img_list[i], img_list[i+1]
			keypoints1, keypoints2, flag = featureMatch(img1, img2, i)
			print(f'Keypoints after matching : {len(keypoints1)}')
			if not flag : 
				print(f'Found less features or not a good match')
				continue
				
			if len(keypoints1) < 4 or len(keypoints2) < 4 : 
				continue

			finalInliers, Hbest, flag = rejectOutliers(keypoints1, keypoints2, img1, img2, i)
	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""
	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""


	"""
	Refine: RANSAC, Estimate Homography
	"""


	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

	
if __name__ == '__main__':
	main()
 
