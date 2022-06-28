import sys
sys.path.insert(1, 'PRNU/CameraFingerprint/')
import src.Functions as Fu
import src.Filter as Ft
import src.maindir as md
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import statistics
import math


def frame_selector(path_video, index_frame, flag_init=0):
    cap = cv2.VideoCapture(path_video)
    mov_array = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, index_frame[0])
    ret, oframe = cap.read()
    print(np.shape(oframe))
    for idx in index_frame[1:len(index_frame)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret is True:
                orb = cv2.SIFT_create()
                queryKeypoints, queryDescriptors = orb.detectAndCompute(frame, None)
                if 'trainKeypoints' not in locals():
                    trainKeypoints, trainDescriptors = orb.detectAndCompute(oframe, None)
                if queryDescriptors is not None and trainDescriptors is not None:
                    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
                    matches = matcher.match(queryDescriptors, trainDescriptors)
                    matches = sorted(matches, key=lambda x: x.distance)
                    matches = matches[:int(len(matches) * 0.9)]
                    if len(matches):
                        med = statistics.median([math.sqrt(
                            (queryKeypoints[match.queryIdx].pt[0] - trainKeypoints[match.trainIdx].pt[0]) ** 2 + (
                                queryKeypoints[match.queryIdx].pt[1] - trainKeypoints[match.trainIdx].pt[1]) ** 2) for
                                         match in matches])
                        matches = [match for match in matches if math.sqrt(
                            (queryKeypoints[match.queryIdx].pt[0] - trainKeypoints[match.trainIdx].pt[0]) ** 2 + (
                                queryKeypoints[match.queryIdx].pt[1] - trainKeypoints[match.trainIdx].pt[
                                1]) ** 2) < med * 10]
                        no_of_matches = len(matches)
                        mov_array.append(statistics.mean([math.sqrt(
                                (queryKeypoints[match.queryIdx].pt[0] - trainKeypoints[match.trainIdx].pt[0]) ** 2 + (
                                    queryKeypoints[match.queryIdx].pt[1] - trainKeypoints[match.trainIdx].pt[1]) ** 2) for
                                                  match in matches]))
                        trainKeypoints = queryKeypoints
                        trainDescriptors = queryDescriptors
                    else:
                        mov_array.append(1000)
                        trainKeypoints = queryKeypoints
                        trainDescriptors = queryDescriptors
                else:
                    mov_array.append(1000)
                    del trainKeypoints, trainDescriptors
                    oframe = frame
        else:
                cap.release()
                break
    min_mov = np.where(mov_array == np.min(mov_array))[0][0]
    if mov_array[min_mov] > 10:
        min_mov = 0
    return min_mov

# crosscorrelation function with selected fingerprint
def crosscorr_Fingerprint(array1, TA, norm2):
    array1 = array1.astype(np.double)
    array1 = array1 - array1.mean()
    normalizator = np.sqrt(np.sum(np.power(array1, 2)) * norm2)
    FA = np.fft.fft2(array1)

    del array1
    AC = np.multiply(FA, TA)
    del FA

    if normalizator == 0:
        ret = None
    else:
        ret = np.real(np.fft.ifft2(AC)) / normalizator
    return ret

# Compute PCE from noise
def compute_PCE(noise, TA, norm2, Fingerprint):
    Noisex1 = np.zeros_like(Fingerprint)
    Noisex1[:noise.shape[0], :noise.shape[1]] = noise
    shift_range = [Fingerprint.shape[0] - noise.shape[0], Fingerprint.shape[1] - noise.shape[1]]
    #shift_range = [160, 240]
    C = crosscorr_Fingerprint(Noisex1, TA, norm2)
    det, det0 = md.PCE(C, shift_range=shift_range)
    return det['PCE']


# compute PCE from image
def compute_PCE_Check(image, TA, norm2, Fingerprint):
    Noisex = Ft.NoiseExtractFromImage(image, sigma=2.)
    Noisex = Fu.WienerInDFT(Noisex, np.std(Noisex))
    Noisex1 = np.zeros_like(Fingerprint)
    Noisex1[:Noisex.shape[0], :Noisex.shape[1]] = Noisex
    shift_range = [Fingerprint.shape[0] - Noisex.shape[0], Fingerprint.shape[1] - Noisex.shape[1]]
    C = crosscorr_Fingerprint(Noisex1, TA, norm2)
    det, det0 = md.PCE(C, shift_range=shift_range)
    return det['PCE']

def calibration(homography, noise, centerrot, centerres, step, TA, norm2, Fingerprint):
    bestpce=0
    rotation=0
    scaling=0
    print(step)
    count=0
    modified=False
    modifiedcheck = False
    for i in sorted(np.arange(-((5 if step==0 else 0.5)*(10**-step)), ((5 if step==0 else 0.5)*(10**-step)), 0.1*(10**-step)), key=abs):
        matrix = homography + np.r_[cv2.getRotationMatrix2D((noise.shape[0] / 2, noise.shape[1] / 2), 2 * (centerrot+i), 1.0), [[0, 0, 1]]]
        matrix[2,2] = matrix[2,2] / (1+centerres)
        rotated = cv2.warpPerspective(noise, matrix, (np.shape(noise)[1], np.shape(noise)[0]))
        pcerot = compute_PCE(rotated, TA, norm2, Fingerprint)
        if pcerot > bestpce:
            bestpce = pcerot
            rotation = centerrot+i
            count = 0
            modified = True
            modifiedcheck = True
        else:
            count+=1
        if count>2 and modified:
            break

    count = 0
    modified = False
    for i in sorted(np.arange(-(0.05*(10**-step)), (0.05*(10**-step)), 0.01*(10**-step)), key=abs):
        scale = (1 + centerres + i)
        matrix = homography + np.r_[cv2.getRotationMatrix2D((noise.shape[0] / 2, noise.shape[1] / 2), 2 * rotation, 1.0), [[0, 0, 1]]]
        matrix[2, 2] = matrix[2, 2] / scale

        resized = cv2.warpPerspective(noise, matrix, (np.uint32(np.rint(np.shape(noise)[1] * scale)), np.uint32(np.rint(np.shape(noise)[0] * scale))))
        pceres = compute_PCE(resized, TA, norm2, Fingerprint)

        if pceres > bestpce:
            bestpce = pceres
            scaling = centerres+i

            count=0
            modified = True
            modifiedcheck = True

        else:
            count+=1

        if count>2 and modified:
            break

    if(step<3) and modifiedcheck:
        matrix, bestpce, rotation, scaling = calibration(homography, noise, rotation, scaling, step+1, TA, norm2, Fingerprint)

    matrix = homography + np.r_[cv2.getRotationMatrix2D((noise.shape[0] / 2, noise.shape[1] / 2), 2 * rotation, 1.0), [[0, 0, 1]]]
    matrix[2, 2] = matrix[2, 2] / (scaling + 1)

    if step==0:
        best = cv2.warpPerspective(noise, matrix, (np.uint32(np.rint(np.shape(noise)[1] * (scaling+1))), np.uint32(np.rint(np.shape(noise)[0] * (scaling+1)))))
        bestpce = compute_PCE(best, TA, norm2, Fingerprint)

    return matrix, bestpce, rotation, scaling

def calibration_GPU(homography, noise, centerrot, centerres, step, TA, norm2, Fingerprint):
    #pre calcolo i parametri dei cicli for, prima rotation e poi scaling,
    #parallelizzo su tfa.transform. RICORDA CHE L'ANGOLO IN ALTO A SINISTRA PER QUESTO METODO DEVE RIMANERE
    #IN ALTO A SINISTRA.

    #Usa tf function per stima in parallelo di cross corr e PCE
    bestpce=0
    rotation=0
    scaling=0

    count=0
    modified=False
    modifiedcheck = False
    # rotation estimation
    matrix = rotation_matrix_estimator(homography, noise.shape, centerrot, centerres, step)

    list_Wrs = tf.repeat(tf.expand_dims(tf.convert_to_tensor(noise, dtype=tf.float32), axis=0), repeats=samp, axis=0)
    rotated = tfa.image.transform(list_Wrs, matrix)

    #
    count = 0
    modified = False
    for i in sorted(np.arange(-(0.05*(10**-step)), (0.05*(10**-step)), 0.01*(10**-step)), key=abs):
        scale = (1 + centerres + i)
        matrix = homography + np.r_[cv2.getRotationMatrix2D((noise.shape[0] / 2, noise.shape[1] / 2), 2 * rotation, 1.0), [[0, 0, 1]]]
        matrix[2, 2] = matrix[2, 2] / scale

        resized = cv2.warpPerspective(noise, matrix, (np.uint32(np.rint(np.shape(noise)[1] * scale)), np.uint32(np.rint(np.shape(noise)[0] * scale))))
        pceres = compute_PCE(resized, TA, norm2, Fingerprint)

        #print("PCE: %f" %pceres)

        if pceres > bestpce:
            bestpce = pceres
            scaling = centerres+i

            count=0
            modified = True
            modifiedcheck = True

        else:
            count+=1

        if count>2 and modified:
            break

    if(step<3) and modifiedcheck:
        matrix, bestpce, rotation, scaling = calibration(homography, noise, rotation, scaling, step+1, TA, norm2, Fingerprint)

    matrix = homography + np.r_[cv2.getRotationMatrix2D((noise.shape[0] / 2, noise.shape[1] / 2), 2 * rotation, 1.0), [[0, 0, 1]]]
    matrix[2, 2] = matrix[2, 2] / (scaling + 1)

    if step==0:
        best = cv2.warpPerspective(noise, matrix, (np.uint32(np.rint(np.shape(noise)[1] * (scaling+1))), np.uint32(np.rint(np.shape(noise)[0] * (scaling+1)))))
        bestpce = compute_PCE(best, TA, norm2, Fingerprint)

    return matrix, bestpce, rotation, scaling


def rotation_matrix_estimator(hom, noise_shape, centerrot, centerres, step):
    idx_hom = 0
    rotation_arr = sorted([i for i in np.arange(-((5 if step==0 else 0.5)*(10**-step)), ((5 if step==0 else 0.5)*(10**-step)), 0.1*(10**-step))], key=abs)
    matrix = np.zeros([100, 3, 3])
    for i in rotation_arr:
        matrix[idx_hom] = hom[idx_hom] + np.r_[cv2.getRotationMatrix2D((noise_shape[0] / 2, noise_shape[1] / 2), 2 * (centerrot+i), 1.0), [[0, 0, 1]]]
        matrix[idx_hom, 2, 2] = matrix[idx_hom, 2, 2] / (1+centerres)
        idx_hom += 1
    mat_reshape = matrix.reshape([100, 9])
    return mat_reshape[:, 0:8]


