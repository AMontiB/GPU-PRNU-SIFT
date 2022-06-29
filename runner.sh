echo "running H1 hypothesis"
nohup python -u main_H1.py --videos PATH_TO_VIDEOS --fingerprint PATH_TO_FINGERPRINTS --output PATH_TO_OUTPUT_FOLDER --gpu_dev /gpu:N >| output_H1.log & 

echo "running H0 hypothesis"
nohup python -u main_H0.py --videos PATH_TO_VIDEOS --fingerprint PATH_TO_FINGERPRINTS --output PATH_TO_OUTPUT_FOLDER --gpu_dev /gpu:N >| output_H0.log & 
