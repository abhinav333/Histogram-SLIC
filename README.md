# Improving Accuracy of Superpixel Segmentation using Colour Histogram based Seed Initialization
- Superpixels are perceptually meaningful image segments formed by grouping pixels based on common properties, such as color, and are useful for accelerating computer vision applications.
- In this work, the Histogram Super Linear Iterative Clustering (SLIC) algorithm is prototyped to enhance the accuracy of the SLIC algorithm by using a color histogram during seed initialization. The quantized color histogram captures localized color profiles within uniform grids in the raw image.
- The quality of initial seeding can be controlled through user-defined parameters, such as histogram suppression window and maximum seeds per grid box.
- Label enforcement of erroneous pixels is implemented on the superpixel segmented image by assigning a pixel with a particular cluster label based on the majority of surrounding labels.

The code for the work can be found in file 'colch.cpp'.
