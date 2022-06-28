#FOR H1
echo "running H1 hypothesis"
nohup python -u main_H1_ds.py --videos PATH_TO_VIDEOS --fingerprint PATH_TO_FINGERPRINTS --gpu_dev /gpu:N >| output_H1.log & #N is the GPU_ID, if you have just 1 GPU N=0
echo "running H0 hypothesis"
nohup python -u main_H0_ds.py --videos PATH_TO_VIDEOS --fingerprint PATH_TO_FINGERPRINTS --gpu_dev /gpu:N >| output_H0.log & #N is the GPU_ID, if you have just 1 GPU N=0
