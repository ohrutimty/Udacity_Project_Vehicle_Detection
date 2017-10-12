
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/slide-windows1.png
[image4]: ./examples/slide-windows2.png
[image5]: ./examples/slide-windows3.png
[image6]: ./examples/slide-windows4.png
[image7]: ./examples/six_out_one_scale.png
[image8]: ./examples/six_out_multi_scale.png
[image9]: ./examples/out_heatmap.png
[image10]: ./examples/out_heatmap_threshold.png
[image11]: ./examples/out_heatmap_threshold_label.png
[image12]: ./examples/out_label_bbox.png
[image13]: ./examples/six_example_detect.png

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters including:

Parameters:
* Color space
* Orientations
* Pixels per cell
* cells per block
* HOG channel

| Label | Colorspace | Orientations | Pixels Per Cell | Cells Per Block | HOG Channel | Accuracy | Extract Time | Train Time |
| :---: | :--------: | :----------: | :-------------: | :-------------: | :---------: | :------: | :----------: | :--------: |
| 1     | RGB        | 9            | 8               | 2               | 0           | 89.95    | 30.36        | 17.89      |
| 2     | RGB        | 9            | 8               | 2               | 1           | 90.74    | 28.12        | 14.63      |
| 3     | RGB        | 9            | 8               | 2               | 2           | 91.81    | 30.54        | 14.33      |
| 4     | RGB        | 9            | 8               | 2               | ALL         | 92.68    | 84.73        | 29.86      |
| 5     | HSV        | 9            | 8               | 2               | ALL         | 95.47    | 94.53        | 21.85      |
| 6     | LUV        | 9            | 8               | 2               | ALL         | 95.97    | 96.68        | 21.25      |
| 7     | HLS        | 9            | 8               | 2               | ALL         | 95.69    | 102.61       | 23.09      |
| 8     | YUV        | 9            | 8               | 2               | ALL         | 96.79    | 91.48        | 19.42      |
| 9     | YCrCb      | 9            | 8               | 2               | ALL         | 96.62    | 88.79        | 18.98      |
| 10    | YCrCb      | 12           | 10              | 2               | ALL         | 96.42    | 112.13       | 15.36      |
| 11    | YCrCb      | 11           | 12              | 2               | ALL         | 96.48    | 58.04        | 9.18       |
| 12    | YCrCb      | 8            | 6               | 2               | ALL         | 95.21    | 145.07       | 32.30      |
| 13    | YCrCb      | 9            | 14              | 2               | ALL         | 96.62    | 40.91        | 4.92       |
| 14    | YCrCb      | 10           | 14              | 2               | ALL         | 97.02    | 38.69        | 4.05       |
| 15    | YCrCb      | 12           | 16              | 2               | ALL         | 97.61    | 42.69        | 3.34       |
| 16    | YUV        | 10           | 8               | 2               | ALL         | 98.11    | 60.8         | 18.43      |
| 17    | YUV        | 10           | 15              | 2               | ALL         | 96.06    | 44.75        | 1.97       |
| 18    | YUV        | 11           | 16              | 2               | ALL         | 97.33    | 36.11        | 1.91       |
| 19    | YUV        | 9            | 16              | 2               | ALL         | 97.55    | 42.31        | 3.02       |
| 20    | YUV        | 9            | 12              | 2               | ALL         | 96.79    | 42.45        | 6.78       |


First, I test hog channel, 'ALL' has better accuracy. Then the next testing parameter is color space, both of 'YCrCb' and 'YUV' have better accuracy than other color space. The last one is testing 'orientations' and 'pixels per cell', when 'pixels per cell' is bigger, the extraction  time will decrease. The 'orientations' seems like having little affection in accuracy. After testing, with `colorspace='YUV'`, `orientations=10`, `pixels_per_cell=8`, `cells_per_block=2`, `hog_channel='ALL'` have better accuracy.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LinearSVC() which I learned from the class and tried various combinations of parameters above to get this result.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Here are some examples of using different scale boxes in restricting area to do sliding window search.

This one is y start from 400 to 496, and the box size is 64*64.
![alt text][image3]

This one is y start from 400 to 544, and the box size is 96*96.
![alt text][image4]

This one is y start from 400 to 592, and the box size is 128*128.
![alt text][image5]

This one is y start from 400 to 640, and the box size is 160*160.
![alt text][image6]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

With one size of the window to search the vehicles, only this size of cars will be recognized.
This image is shown using one scale and YUV 3-channnel HOG features to predict vehicles result.
![alt text][image7]

Therefore, using multi scale of windows to search and this image is shown using multi scale and YUV 3-channnel HOG features to predict vehicles result.
![alt text][image8]

Now we are able to find different size of cars, but there are including some of error prediction.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections, I created a heatmap and then threshold that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### This image show the heatmap from previous output image:

![alt text][image9]

Add threshold to solve false positive problem. I use threshold=2 in order to obtain better output in video.

![alt text][image10]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from previous frame:
![alt text][image11]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image12]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As you can see some test image may recognize error between vehicle and non vehicle. In the output video, it only shows in a few frames, but it still has some method to improve the result, such as classifier, integrate more function in extract feature not only HOG, and so on.

![alt text][image13]
