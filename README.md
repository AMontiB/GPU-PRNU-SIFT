# GPU-accelerated SIFT-aided source identification of stabilized videos

This is the official code implementation of the ICIP 2022 paper ["GPU-accelerated SIFT-aided source identification of stabilized videos"]()

## Requirements

- Download the python libraries of [Camera-fingerprint](https://dde.binghamton.edu/download/camera_fingerprint/) ;
 - if [Camera-fingerprint](https://dde.binghamton.edu/download/camera_fingerprint/) is not already, reorganize the folders such that ```PRNU/CameraFingerprint``` ;
 - Download the Reference Camera Fingerprints [here](https://drive.google.com/drive/folders/1q6FpTvP5FYsgaQf5kbC3vjuT6s8jbmxs?usp=sharing);
 - at least 9G GPU.
## Set up Virtual-Env
```
conda env create -f environment.yml
```
## VISION DATASET

Download Vision dataset [here](https://lesc.dinfo.unifi.it/en/datasets).

# Test

## Test a match (H1) hypothesis case
```
nohup python -u main_H1.py --videos PATH_TO_VIDEOS --fingerprint PATH_TO_FINGERPRINTS --output PATH_TO_OUTPUT_FOLDER --gpu_dev /gpu:N >| output_H1.log & 
```

## Test a mis-match (H0) hypothesis case
```
nohup python -u main_H0.py --videos PATH_TO_VIDEOS --fingerprint PATH_TO_FINGERPRINTS --output PATH_TO_OUTPUT_FOLDER --gpu_dev /gpu:N >| output_H0.log & 
```

## Run both
Edit and Run ```bash runner.sh```

## NOTE:
You need to edit:
- ```PATH_TO_VIDEOS``` changing it with the path to your dataset
- ```PATH_TO_FINGERPRINTS``` changing it with the path to your reference camera fingerprints
- ```PATH_TO_OUTPUT_FOLDER``` changing it with the path to your output folder
- ```N``` chaging it with your GPU ID

# Results of the Paper and its Tech Repo

Check ["GPU-accelerated SIFT-aided source identification of stabilized videos"]()

<p align="center">
  <img src="https://github.com/AMontiB/GPU-PRNU-SIFT/blob/main/figures/ROC.png">
</p>

![tables](https://github.com/AMontiB/GPU-PRNU-SIFT/blob/main/figures/table.png?raw=true)

# Cite Us
If you use this material please cite: 

@misc{https://doi.org/10.48550/arxiv.2207.14507,\
  doi = {10.48550/ARXIV.2207.14507},\
  url = {https://arxiv.org/abs/2207.14507},\
  author = {Montibeller, Andrea and Pasquini, Cecilia and Boato, Giulia and Dell'Anna, Stefano and Pérez-González, Fernando},\
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Multimedia (cs.MM), Image and Video Processing (eess.IV), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},\
  title = {GPU-accelerated SIFT-aided source identification of stabilized videos},\
  publisher = {arXiv},\
  year = {2022},\
  copyright = {Creative Commons Attribution 4.0 International}\
}

