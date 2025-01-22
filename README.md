# Feature based 3d Multi-modal Multi-object Tracking for Autonomous Navigation
Part of my master thesis: **Probabilistic 3D Multi-Modal Multi-Object Tracking via Machine Learning and Analytic Collision Risk Calculation for Autonomous Vehicles Navigation.**
## Overview
This is the main module. You can check my extensive report for all modules in the ```Report``` folder.
A review paper in english is under construction.

## Instructions
### 1. Extract Features
If not already extract the Geometric features from CenterPoint detector and the corresponding Appearance features from each camera from Mask R-CNN. Follow our [Point Cloud Feature Extractor](https://github.com/DimSpathoulas/Point_Cloud_Feature_Extractor.git) and our [2d Feature Extractor](https://github.com/DimSpathoulas/2D_FEATURE_EXTRACTOR.git).

### 2. Clone our repo and setup the environment
```bash
cd path_to_your_projects/GEOMAPP_Feature_Tracking
conda create --name feature_tracking python=3.8.19
git clone https://github.com/DimSpathoulas/GeomApp_3MOT.git
conda env create -f environment.yml
conda activate feature_tracking
```

### 3. Calculate covariance matrix
I have hardcoded the results for the variances based on centerpoint detector in ```cov_centerpoint.py ```, but you can also run the results with ```covariance_matrix/cov_calc_centerpoint.py``` with:
```bash
python covariance_matrix/cov_calc_centerpoint.py > covariance_matrix/centerpoint_cov.txt
```
### 4. Run main code
You can run the main code for all 3 learning multi-stages both in train and val mode. Note train mode in stages 1 and 2 will give as outputs both the resulting net .pth file and the .json tracking results ready for evaluation.
To run our results:

NOTE: CURRENTLY ONLY G1,G2 FUNCTION PROPERLY

### 5. Dimensionaly Reduction
You can vizualise the manifolds individually using ```manifolds.py ``` for multiple instances. This will create a directory named features storing the results in .png format.
You can also run SVD on both geometric with ```svd_per_point.py ``` for per point SVD in point cloud features or ```svd.py ``` for first flattening and then applying SVD on them and ```svd_cam.py``` for appearance features . They will output a .pkl file.
You can use these transformation matrices for training the modules with whatever features dimensions desired.

### 6. Get Results
To get the tracking metric results run:
```bash
python evaluate_nuscenes.py --output_dir results tracking_output.json > results/tracking_output.txt
```


## The Final Project directory
```bash
# Path_to_your_projects        
└── GEOMAPP_Feature_Tracking
       ├── main.py           <-- main code
       ├── main_2.py         <-- main code for running only G1,G4
       ├── mmot_mot_3d.py    <-- backbone of main.py
       ├── g1_g4.py          <-- backbone code of main_2.py
       ├── manifolds.py      <-- vizualise geometric features
       ├── svd.py            <-- run SVD on geometric features flattened
       ├── svd_per_point.py  <-- run SVD on geometric features per point
       ├── svd_cam.py        <-- run SVD on camera features
       ├── svd_umap.py       <-- run svd and then vizualize features with umap
       ├── functions         <-- complementary modules
              ├── inner_funcs.py     <-- functions used for tracking
              ├── outer_funcs.py     <-- groundtruth retrieval and output customization
              ├── Kalman_Filter.py   <-- Linear Kalman Filter
       ├── results           <-- directory containing results
```

## References and Acknowledgments

### References
- Module heavily based on the principles outlined in this paper: [Probabilistic 3D Multi-Modal, Multi-Object Tracking for Autonomous Driving](https://arxiv.org/pdf/2012.13755)
### Acknowledgments
- Built on top of this implementation:
https://github.com/eddyhkchiu/mahalanobis_3d_multi_object_tracking


## Notes
