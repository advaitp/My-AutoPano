Phase1
To run the code to get panorama 
```
python Wrapper.py
```
To stitch images in order put inorder = True otherwise the stitching will be done by dividing image set in 2 parts.
The outputs are saved in the same directory of the code

Input : img_dir : Image Directory form the panorama

Output of panorama : mypano.png
Output of matches : matching.png
Output of RANSAC : RANSAC.png
Output of features : fd.png

Phase2
First run the main_dataset_generation.py file which is in Dataset folder it will save the data in folder '../Data/train_homo/'
The data will be saved in train_homo under train ,validation and test. Each of the folder has the following structure.
original_patch homography_patch and output folders
output consists of difference point.

Then run the Train.py giving the appropriate paths and considering all the arguments. Then run test.py it outputs epe for each model and gives an example output of image with real and predicted homography points 