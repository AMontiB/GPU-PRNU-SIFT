# GPU-accelerated SIFT-aided source identification of stabilized videos

This is the official code implementation of the ICIP 2022 paper ["GPU-accelerated SIFT-aided source identification of stabilized videos"]()

## Requirements

- Download the python libraries of [Camera-fingerprint](https://dde.binghamton.edu/download/camera_fingerprint/) ;
 - if [Camera-fingerprint](https://dde.binghamton.edu/download/camera_fingerprint/) is not already, reorganize the folders such that ```PRNU/CameraFingerprint``` ;
 - Download the Reference Camera Fingerprints [here](https://drive.google.com/drive/folders/1q6FpTvP5FYsgaQf5kbC3vjuT6s8jbmxs?usp=sharing);
 - at least 9G of GPU.
## Set up Virtual-Env
```
conda env create -f environment.yml
```
## VISION DATASET

Download Vision dataset [here](https://lesc.dinfo.unifi.it/en/datasets).

# Test

## Test a match (H1) hypothesis case
```
nohup python -u main_H1_ds.py --videos PATH_TO_VIDEOS --fingerprint PATH_TO_FINGERPRINTS --output PATH_TO_OUTPUT_FOLDER --gpu_dev /gpu:N >| output_H1.log & 
```

## Test a mis-match (H0) hypothesis case
```
nohup python -u main_H0_ds.py --videos PATH_TO_VIDEOS --fingerprint PATH_TO_FINGERPRINTS --output PATH_TO_OUTPUT_FOLDER --gpu_dev /gpu:N >| output_H0.log & 
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

![ROC](https://github.com/AMontiB/GPU-PRNU-SIFT/blob/main/figures/ROC.png?raw=true)

![tables](https://github.com/AMontiB/GPU-PRNU-SIFT/blob/main/figures/table.png?raw=true)

# Cite Us
If you use this material please cite: 

@inproceedings{Montibeller22GPU, \
  title={GPU-accelerated SIFT-aided source identification of stabilized videos}, \
  author={Montibeller, Andrea and Pasquini, Cecilia and Boato, Giulia and Dell'Anna, Stefano and P\'erez-Gonz\'alez, Fernando}, \
  year={2022, ICIP}, \
}

