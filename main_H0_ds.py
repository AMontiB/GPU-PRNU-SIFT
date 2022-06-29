import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import sys
sys.path.insert(1, 'PRNU/CameraFingerprint/')
import src.Functions as Fu
import src.Filter as Ft
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_addons as tfa
import time
import support_function as sf
import statistics
import math
import exiftool
import glob
from scipy.io import loadmat
from scipy.io import savemat
import random


def findbasescaling(noise, TA_tf):
    step = 0
    pcethres = 0
    bestscaling = 0.85
    size_fing = np.shape(TA_tf.numpy())
    interval = 0.003
    while step < 4:
        scale_arr = sorted(np.arange(-0.15, 0.15,interval) if step == 0 else np.arange(-(0.05 * (10 ** -step)), (0.05 * (10 ** -step)), 0.01 * 10 ** -step), key=abs)
        size = (size_fing[0]*size_fing[1])*len(scale_arr)
        memlimit = 20000
        if size>memlimit:
            div = np.int32(np.ceil(size/memlimit))
            off = np.int32(np.floor(len(scale_arr)/div))
            for i in range(div):
                matrix, ranges, arr_scale = basescale_matrix_estimator(scale_arr[off*i:off*(i+1)], bestscaling, np.shape(noise), size_fing)
                list_Wrs = tf.expand_dims(tf.repeat(tf.expand_dims(tf.convert_to_tensor(noise, dtype=tf.float32), axis=0), repeats=len(matrix), axis=0), axis=-1)
                batch_Wrs_transformed = tfa.image.transform(tf.convert_to_tensor(list_Wrs, dtype=tf.float32), matrix, 'BILINEAR', [size_fing[1], size_fing[2]])
                XC = (crosscorr_Fingeprint_GPU(batch_Wrs_transformed, TA_tf, norm2, np.shape(TA_tf.numpy())))
                pce_values = parallel_PCE(XC.numpy(), len(XC), ranges)
                index_max = np.argmax(pce_values)
                if pce_values[index_max] > pcethres:
                    pcethres = pce_values[index_max]
                    bestscaling_temp = arr_scale[index_max]
                else:
                    if np.max(pce_values) < pcethres/3:
                        break
        else:
            matrix, ranges, arr_scale = basescale_matrix_estimator(scale_arr, bestscaling, np.shape(noise), np.shape(TA))
            list_Wrs = tf.expand_dims(tf.repeat(tf.expand_dims(tf.convert_to_tensor(noise, dtype=tf.float32), axis=0), repeats=len(matrix), axis=0), axis=-1)
            batch_Wrs_transformed = tfa.image.transform(tf.convert_to_tensor(list_Wrs, dtype=tf.float32), matrix, 'BILINEAR', [size_fing[1], size_fing[2]])
            XC = (crosscorr_Fingeprint_GPU(batch_Wrs_transformed, TA_tf, norm2, np.shape(TA_tf.numpy())))
            pce_values = parallel_PCE(XC.numpy(), len(XC), ranges)
            index_max = np.argmax(pce_values)
            if pce_values[index_max] > pcethres:
                pcethres = pce_values[index_max]
                bestscaling_temp = arr_scale[index_max]
        if pcethres < 50 and step == 0:
            if interval == 0.001:
                fail = True
                break
            interval = 0.001
        else:
            bestscaling = bestscaling_temp
            step += 1
    if fail is True:
        #estimate empirically
        bestscaling = 0.63
    return bestscaling


def basescale_matrix_estimator(scale_arr, bestscaling, noise_shape, K_shape):
    idx_hom = 0
    matrix = np.zeros([len(scale_arr), 3, 3])
    ranges = []
    arr_scale = []
    for i in scale_arr:
        scale = bestscaling + i
        arr_scale.append(scale)
        ranges.append([(K_shape[0] - np.round(noise_shape[0]*scale)).astype(int), (K_shape[1] - np.round(noise_shape[1]*scale)).astype(int)])
        matrix[idx_hom, 0, 0] = 1/scale
        matrix[idx_hom, 1, 1] = 1/scale
        matrix[idx_hom, 2, 2] = 1
        idx_hom += 1
    mat_reshape = matrix.reshape([len(scale_arr), 9])
    return mat_reshape[:, 0:8], ranges, arr_scale


def crosscorr_Fingeprint_GPU(batchW, TA, norm2, sizebatch_K):
    meanW_batch = (tf.repeat(tf.repeat((tf.expand_dims(tf.expand_dims(tf.reduce_mean(batchW,
                                                       axis=[1, 2]), axis=2), axis=3)),
                              repeats=[sizebatch_K[1]], axis=1), repeats=[sizebatch_K[2]], axis=2))
    batchW = batchW - meanW_batch
    normalizator = tf.math.sqrt(tf.reduce_sum(tf.math.pow(batchW, 2)) * norm2)
    FA = tf.signal.fft2d(tf.cast(tf.squeeze(batchW, axis= 3), tf.complex64))
    AC = tf.multiply(FA, tf.repeat(tf.cast(TA, dtype=tf.complex64), axis=0, repeats=len(batchW.numpy())))
    return tf.math.real(tf.signal.ifft2d(AC)) / normalizator


def parallel_PCE(CXC, idx, ranges, squaresize=11):
    out = np.zeros(idx)
    for i in range(0, idx):
        shift_range = ranges[i]
        Out = dict(PCE=[], pvalue=[], PeakLocation=[], peakheight=[], P_FA=[], log10P_FA=[])
        C = CXC[i]
        Cinrange = C[-1-shift_range[0]:,-1-shift_range[1]:]
        [max_cc, imax] = np.max(Cinrange.flatten()), np.argmax(Cinrange.flatten())
        [ypeak, xpeak] = np.unravel_index(imax,Cinrange.shape)[0], np.unravel_index(imax,Cinrange.shape)[1]
        Out['peakheight'] = Cinrange[ypeak,xpeak]
        del Cinrange
        Out['PeakLocation'] = [shift_range[0]-ypeak, shift_range[1]-xpeak]
        C_without_peak = _RemoveNeighborhood(C,
                                         np.array(C.shape)-Out['PeakLocation'],
                                         squaresize)
        PCE_energy = np.mean(C_without_peak*C_without_peak)
        out[i] = (Out['peakheight']**2)/PCE_energy * np.sign(Out['peakheight'])
    return out


def _RemoveNeighborhood(X,x,ssize):
    # Remove a 2-D neighborhood around x=[x1,x2] from matrix X and output a 1-D vector Y
    # ssize     square neighborhood has size (ssize x ssize) square
    [M,N] = X.shape
    radius = (ssize-1)/2
    X = np.roll(X,[np.int(radius-x[0]),np.int(radius-x[1])], axis=[0,1])
    Y = X[ssize:,:ssize];   Y = Y.flatten()
    Y = np.concatenate([Y, X.flatten()[int(M*ssize):]], axis=0)
    return Y


def parallel_PCE2(CXC, idx, ranges, neigh_radius: int = 2):
    out = np.zeros(idx)
    for i in range(0, idx):
        ranges_pre = ranges[i]
        cc = CXC[i]
        shape_cc = np.shape(cc)
        assert (cc.ndim == 2)
        assert (isinstance(neigh_radius, int))
        cc_inrange = cc[-1-ranges_pre[0]:, -1-ranges_pre[1]:]
        max_idx = np.argmax(cc_inrange.flatten())
        max_y, max_x = np.unravel_index(max_idx, cc_inrange.shape)
        max_y = shape_cc[0] - max_y - 1
        max_x = shape_cc[1] - max_x - 1
        peak_height = cc[max_y, max_x]
        cc_nopeaks = cc.copy()
        cc_nopeaks[max_y - neigh_radius:max_y + neigh_radius, max_x - neigh_radius:max_x + neigh_radius] = 0
        pce_energy = np.mean(cc_nopeaks.flatten() ** 2)
        out[i] = (peak_height ** 2) / pce_energy * np.sign(peak_height)
    return out


def calibration_GPU(homography, noise, centerrot, centerres, step, TA, norm2, size_Fingeprint, matrix_off, bestpce=0):
    #Usa tf function per stima in parallelo di cross corr e PCE
    rotation=0
    scaling=0
    modifiedcheck = False
    # rotation estimation
    matrix, thetas = rotation_matrix_estimator(homography, noise.shape, centerrot, centerres, step) 
    list_Wrs = tf.expand_dims(tf.repeat(tf.expand_dims(tf.convert_to_tensor(noise, dtype=tf.float32), axis=0), repeats=len(matrix), axis=0), axis=-1)
    batchW = tfa.image.transform(list_Wrs, matrix, 'BILINEAR', [size_Fingeprint[1], size_Fingeprint[2]])
    ranges = np.repeat([[size_Fingeprint[1] - noise.shape[0], size_Fingeprint[2] - noise.shape[1]]], repeats=100, axis=0)
    print(len(matrix))
    if len(matrix) > 49:
        XC = (crosscorr_Fingeprint_GPU(batchW[0:25], TA, norm2, size_Fingeprint))
        PCE_arr = parallel_PCE(XC.numpy(), len(batchW[0:25]), ranges[0:25])
        XC = (crosscorr_Fingeprint_GPU(batchW[25:50], TA, norm2, size_Fingeprint))
        PCE_arr1 = parallel_PCE(XC.numpy(), len(batchW[25:50]), ranges[25:50])
        XC = (crosscorr_Fingeprint_GPU(batchW[50:75], TA, norm2, size_Fingeprint))
        PCE_arr2 = parallel_PCE(XC.numpy(), len(batchW[50:75]), ranges[50:75])
        XC = (crosscorr_Fingeprint_GPU(batchW[75:100], TA, norm2, size_Fingeprint))
        PCE_arr3 = parallel_PCE(XC.numpy(), len(batchW[75:100]), ranges[75:100])
        PCE_MAX_ARR = [np.max(PCE_arr), np.max(PCE_arr1), np.max(PCE_arr2), np.max(PCE_arr3)]
        idx_max = np.where(PCE_MAX_ARR==np.max(PCE_MAX_ARR))
        if idx_max[0][0] == 0:
            idx = np.where(PCE_arr==np.max(PCE_arr))
            if np.max(PCE_arr) > bestpce:
                bestpce = np.max(PCE_arr)
                rotation = thetas[idx[0][0]]
                modifiedcheck = True
                matrix_off = matrix[idx[0][0]]

        elif idx_max[0][0] == 1:
            idx = np.where(PCE_arr1==np.max(PCE_arr1))
            if np.max(PCE_arr1) > bestpce:
                bestpce = np.max(PCE_arr1)
                rotation = thetas[25+idx[0][0]]
                modifiedcheck = True
                matrix_off = matrix[25+idx[0][0]]

        elif idx_max[0][0] == 2:
            idx = np.where(PCE_arr2==np.max(PCE_arr2))
            if np.max(PCE_arr2) > bestpce:
                bestpce = np.max(PCE_arr2)
                rotation = thetas[50+idx[0][0]]
                modifiedcheck = True
                matrix_off = matrix[50+idx[0][0]]

        elif idx_max[0][0] == 3:
            idx = np.where(PCE_arr3==np.max(PCE_arr3))
            if np.max(PCE_arr3) > bestpce:
                bestpce = np.max(PCE_arr1)
                rotation = thetas[75+idx[0][0]]
                matrix_off = matrix[75+idx[0][0]]
                modifiedcheck = True
        del PCE_arr1, PCE_arr, batchW

    else:
        XC = (crosscorr_Fingeprint_GPU(batchW[0:len(matrix)], TA, norm2, size_Fingeprint))
        PCE_arr = parallel_PCE(XC.numpy(), len(batchW[0:len(matrix)]), ranges[0:len(matrix)])
        del XC
        idx = np.where(PCE_arr==np.max(PCE_arr))
        if np.max(PCE_arr) > bestpce:
            bestpce = np.max(PCE_arr)
            rotation = thetas[idx[0][0]]
            modifiedcheck = True
            matrix_off = matrix[idx[0][0]]
        del batchW, PCE_arr
    #scaling estimation
    matrix, ranges, arr_scale = scale_matrix_estimator(homography, noise.shape, centerres, step, rotation, size_Fingeprint)
    if len(matrix) != len(list_Wrs):
        del list_Wrs
        list_Wrs = tf.expand_dims(tf.repeat(tf.expand_dims(tf.convert_to_tensor(noise, dtype=tf.float32), axis=0), repeats=len(matrix), axis=0), axis=-1)
        batchW = tfa.image.transform(list_Wrs, matrix, 'BILINEAR', [size_Fingeprint[1], size_Fingeprint[2]])
    else:
        batchW = tfa.image.transform(list_Wrs, matrix, 'BILINEAR', [size_Fingeprint[1], size_Fingeprint[2]])
    XC = (crosscorr_Fingeprint_GPU(batchW, TA, norm2, size_Fingeprint))
    PCE_arr = parallel_PCE(XC.numpy(), len(batchW), ranges)
    del XC, batchW
    pceres = np.max(PCE_arr)
    idx = np.where(PCE_arr==pceres)
    del PCE_arr
    #
    if pceres > bestpce:
        bestpce = pceres
        scaling = arr_scale[idx[0][0]] - 1
        modifiedcheck = True
        matrix_off = matrix[idx[0][0]]
    if step < 3 and modifiedcheck:
        matrix_off, bestpce, rotation, scaling = calibration_GPU(homography, noise, rotation, scaling, step+1, TA, norm2, size_Fingeprint, matrix_off, bestpce)
    if step == 0:
        W_T = tf.expand_dims(tfa.image.transform(list_Wrs[0], matrix_off, 'BILINEAR', [size_Fingeprint[1], size_Fingeprint[2]]), axis=0)
        XC = (crosscorr_Fingeprint_GPU(W_T, TA, norm2, size_Fingeprint))
        ranges = [[(size_Fingeprint[1] - np.round(noise.shape[0]/(scaling+1))).astype(int), (size_Fingeprint[2] - np.round(noise.shape[1]/(scaling+1))).astype(int)]]
        bestpce = parallel_PCE(XC.numpy(), len(W_T), ranges)
    del list_Wrs
    return matrix_off, bestpce, rotation, scaling


def rotation_matrix_estimator(hom, noise_shape, centerrot, centerres, step):
    idx_hom = 0
    rotation_arr = sorted([i for i in np.arange(-((5 if step==0 else 0.5)*(10**-step)), ((5 if step==0 else 0.5)*(10**-step)), 0.1*(10**-step))], key=abs)
    if not np.any(hom):
        hom = np.zeros([len(rotation_arr), 3, 3])
    else:
        hom = np.matmul(hom, np.ones([len(rotation_arr), 3, 3]))
    matrix = np.zeros([len(rotation_arr), 3, 3])
    arr_rotation = []
    for i in rotation_arr:
        matrix[idx_hom] = hom[idx_hom] + np.r_[cv2.getRotationMatrix2D((noise_shape[0] / 2, noise_shape[1] / 2), 2 * (centerrot-i), 1.0), [[0, 0, 1]]]
        arr_rotation.append(centerrot-i)
        matrix[idx_hom] = matrix[idx_hom] / (1+centerres)
        matrix[idx_hom, 2, 2] = matrix[idx_hom, 2, 2] * (1+centerres)
        idx_hom += 1
    mat_reshape = matrix.reshape([len(rotation_arr), 9])
    return mat_reshape[:, 0:8], arr_rotation


def scale_matrix_estimator(hom, noise_shape, centerres, step, rotation, K_shape):
    scale_arr = sorted(np.arange(-(0.05*(10**-step)), (0.05*(10**-step)), 0.01*(10**-step)), key=abs)
    idx_hom = 0
    if not np.any(hom):
        hom = np.zeros([len(scale_arr), 3, 3])
    else:
        hom = np.repeat(hom, repeats=len(scale_arr), axis=0)
    matrix = np.zeros([len(scale_arr), 3, 3])
    ranges = []
    arr_scale = []
    for i in scale_arr:
        scale = (1 + centerres + i)
        arr_scale.append(scale)
        ranges.append([(K_shape[0] - np.round(noise_shape[0]/scale)).astype(int), (K_shape[1] - np.round(noise_shape[1]/scale)).astype(int)])
        matrix[idx_hom] = hom[idx_hom] + np.r_[cv2.getRotationMatrix2D((noise_shape[0] / 2, noise_shape[1] / 2), 2 * rotation, 1.0), [[0, 0, 1]]]
        matrix[idx_hom] = matrix[idx_hom] / scale
        matrix[idx_hom, 2, 2] = matrix[idx_hom, 2, 2] * scale
        idx_hom += 1
    mat_reshape = matrix.reshape([len(scale_arr), 9])
    return mat_reshape[:, 0:8], ranges, arr_scale


FLAGS = tf.compat.v1.flags.FLAGS
# dataset
tf.compat.v1.flags.DEFINE_string('videos', '../../../home/testImages/VISION/video_stabilization/', 'path to videos')
tf.compat.v1.flags.DEFINE_string('fingerprint', 'fingerprints/*', 'path to fingerprint')
tf.compat.v1.flags.DEFINE_string('output', 'OUTPUT/', 'path to output')
tf.compat.v1.flags.DEFINE_string('gpu_dev', '/gpu:0', 'gpu device')
physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)
with tf.device(FLAGS.gpu_dev):
    fingerprint_paths = sorted(glob.glob((FLAGS.fingerprint+'*')))
    test_set = FLAGS.videos
    for fingerprint_path in fingerprint_paths:
        #import camera fingerprint
        print(fingerprint_path)
        K = loadmat(fingerprint_path)
        flag_choice = True
        while flag_choice:
            choice = fingerprint_paths[random.randint(0, len(fingerprint_paths)-1)]
            if choice != fingerprint_path:
                flag_choice = False
        device = choice[len(fingerprint_path)-7:-4]
        device2 = fingerprint_path[len(fingerprint_path)-7:-4]
        if device2 == 'D02':
                    basescaling = 1.333174
                    crop_array = [345, 1491, 206, 2242] #[off_x, size_x, off_y, size_y]
        elif device2 == 'D20':
                    basescaling = 1.2270420000000004
                    crop_array = [216, 1362, 38, 2074] #[off_x, size_x, off_y, size_y]

        elif device2 == 'D29':
                    basescaling = 1.454894
                    crop_array = [269, 1414, 103, 2140] #[off_x, size_x, off_y, size_y]

        elif device2 == 'D34':
                    basescaling = 1.454877
                    crop_array = [269, 1414, 103, 2140] #[off_x, size_x, off_y, size_y]

        elif device2 == 'D05':
                    basescaling = 1.454894
                    crop_array = [269, 1414, 103, 2140] #[off_x, size_x, off_y, size_y]

        elif device2 == 'D14':
                    basescaling = 1.45486
                    crop_array = [269, 1414, 103, 2140] #[off_x, size_x, off_y, size_y]

        elif device2 == 'D06':
                    basescaling = 1.416848
                    crop_array = [291, 1437, 134, 2170] #[off_x, size_x, off_y, size_y]

        elif device2 == 'D19':
                    basescaling = 1.417239
                    crop_array = [291, 1436, 133, 2170] #[off_x, size_x, off_y, size_y]

        elif device2 == 'D25':
                    basescaling = 1.93311583333333
                    crop_array = [327, 1473, 182, 2218] #[off_x, size_x, off_y, size_y]

        elif device2 == 'D10':
                    basescaling = 1.333123
                    crop_array = [345, 1491, 206, 2242] #[off_x, size_x, off_y, size_y]

        elif device2 == 'D18':
                    basescaling = 1.454826
                    crop_array = [269, 1414, 104, 2140] #[off_x, size_x, off_y, size_y]

        elif device2 == 'D15':
                    basescaling = 1.416525
                    crop_array = [291, 1437, 134, 2170] #[off_x, size_x, off_y, size_y]

        elif device2 == 'D12':
                    basescaling = 2.63852
                    crop_array = [460, 1988, 275, 2989] #[off_x, size_x, off_y, size_y]

        Fingerprint = K['fing']
        Fingerprint = cv2.resize(Fingerprint, (0, 0), fx=(1/basescaling), fy=(1/basescaling))
        Fingerprint = Fingerprint[crop_array[0]:crop_array[1],crop_array[2]:crop_array[3]]
        size_fing = np.shape(Fingerprint)
        array2 = Fingerprint.astype(np.double)
        array2 = array2 - array2.mean()
        tilted_array2 = np.fliplr(array2)
        tilted_array2 = np.flipud(tilted_array2)
        norm2 = np.sum(np.power(array2, 2))
        TA = np.fft.fft2(tilted_array2)
        TA_tf = tf.expand_dims(tf.convert_to_tensor(TA, dtype=tf.complex64), axis=0)
        test_set_device = test_set + device + '*'
        videos_path = glob.glob(test_set_device)
        for video_path in videos_path:
            print(video_path)
            out_file = FLAGS.output + '/' + device2 + '_' + video_path[len(test_set):-4] + '_PCE.mat'
            if not os.path.exists(out_file):
                start_run = time.time()
                with exiftool.ExifTool() as et:
                    orientation = et.get_metadata(video_path)["Composite:Rotation"]
                    print("Rotation: %d" % orientation)
                if orientation == 90 or orientation == 270:
                    print("vertical video, skipping")
                else:
                    id_job = os.getpid()
                    path_to_file = 'frames' + str(id_job) + '.txt'
                    if os.path.exists(path_to_file):
                        cmd =  ("rm -r %s" % path_to_file)
                        os.system(cmd)
                    cmd = ("ffprobe %s -show_frames | grep -E pict_type > %s" % (video_path, path_to_file))
                    os.system(cmd)
                    f = open(path_to_file, "r")
                    lines = f.readlines()
                    index=[]
                    count = 0
                    for line in lines:
                        if line[-2] == 'I':
                            index.append(count)
                        count += 1
                    # selection frames
                    start = time.time()
                    idx_start_frame = sf.frame_selector(video_path, index)
                    print('TIME CONSUMED frame selector: ', time.time()-start)
                    index_first_anchor = index[idx_start_frame]
                    index_second_anchor = index[idx_start_frame+1]
                    cap = cv2.VideoCapture(video_path)
                    print('real index 1: ', index[idx_start_frame])
                    print('real index 2: ', index[idx_start_frame+1])
                    pce_anchors = []
                    #pce achor 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index[idx_start_frame])
                    _, frame = cap.read()
                    inversion = False
                    if orientation == 180:
                        inversion = True
                        print("Inverted video")
                    if inversion:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    resized = frame
                    noise = Ft.NoiseExtractFromImage(resized, sigma=2.)
                    noise = Fu.WienerInDFT(noise, np.std(noise))
                    W_T1 = tfa.image.transform(tf.convert_to_tensor(noise, dtype=tf.float32), [1,0,0,0,1,0,0,0], 'BILINEAR', [size_fing[0], size_fing[1]])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index[idx_start_frame+1])
                    _, frame = cap.read()
                    if inversion:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    resized = frame
                    noise = Ft.NoiseExtractFromImage(resized, sigma=2.)
                    noise = Fu.WienerInDFT(noise, np.std(noise))
                    W_T2 = tfa.image.transform(tf.convert_to_tensor(noise, dtype=tf.float32), [1,0,0,0,1,0,0,0], 'BILINEAR', [size_fing[0], size_fing[1]])
                    time_array = []
                    XC = (crosscorr_Fingeprint_GPU((tf.expand_dims([W_T1, W_T2], axis=3)), TA_tf, norm2,
                                               np.shape(TA_tf.numpy())))
                    ranges = [[(size_fing[0] - noise.shape[0]), (size_fing[1] - noise.shape[1])], [(size_fing[0] - noise.shape[0]), (size_fing[1] - noise.shape[1])]]
                    pce_anchors = parallel_PCE(XC.numpy(), len(XC), ranges)
                    #find maximum
                    idx_anchor_start = np.where(pce_anchors == np.max(pce_anchors))[0][0]
                    if idx_anchor_start == 0:
                        start_idx_frame = index[idx_start_frame]
                        end_idx_frame = index[idx_start_frame+1]
                        set_idx_frame = [i for i in range(start_idx_frame, end_idx_frame+1)]
                    elif idx_anchor_start == 1:
                        start_idx_frame = index[idx_start_frame]
                        end_idx_frame = index[idx_start_frame+1]
                        set_idx_frame = sorted([i for i in range(start_idx_frame, end_idx_frame+1)], reverse=True)
                    cap.release()
                    cap = cv2.VideoCapture(video_path)
                    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
                    pce_array = []
                    start_loop = time.time()
                    for idx_frame in set_idx_frame:
                        print('IDX frame: ', idx_frame)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
                        ret, frame = cap.read()
                        if inversion:
                                frame = cv2.rotate(frame, cv2.ROTATE_180)
                        start_time = time.time()
                        if ret:
                            resized = frame
                            noise = Ft.NoiseExtractFromImage(resized, sigma=2.)
                            noise = Fu.WienerInDFT(noise, np.std(noise))

                            #
                            W_T = tfa.image.transform(tf.convert_to_tensor(noise, dtype=tf.float32), [1,0,0,0,1,0,0,0], 'BILINEAR', [size_fing[0], size_fing[1]])
                            XC = (crosscorr_Fingeprint_GPU((tf.expand_dims(tf.expand_dims(W_T, axis=0), axis=3)), TA_tf, norm2,
                                                    np.shape(TA_tf.numpy())))
                            ranges = [[(size_fing[0] - noise.shape[0]), (size_fing[1] - noise.shape[1])]]
                            pceres = parallel_PCE(XC.numpy(), len(XC), ranges)
                            #
                            print("PCE after resizing: %f" % pceres)
                            if ((int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1) not in index) and ('oframe' in locals()):
                                orb = cv2.SIFT_create()
                                start = time.time()
                                queryKeypoints, queryDescriptors = orb.detectAndCompute(resized, None)
                                trainKeypoints, trainDescriptors = orb.detectAndCompute(oframe, None)
                                if queryDescriptors is not None and trainDescriptors is not None:
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
                                    if no_of_matches > 4:
                                        p1 = np.zeros((no_of_matches, 2))
                                        p2 = np.zeros((no_of_matches, 2))
                                        for i in range(len(matches)):
                                            p1[i, :] = queryKeypoints[matches[i].queryIdx].pt
                                            p2[i, :] = trainKeypoints[matches[i].trainIdx].pt
                                        ## FAST
                                        hom, mask = cv2.findHomography(p1, p2, cv2.USAC_DEFAULT)
                                        if hom is not None:
                                            #
                                            hom[0, 2] = 0
                                            hom[1, 2] = 0
                                            #
                                            noisecorr = cv2.warpPerspective(noise, hom, (np.shape(noise)[1], np.shape(noise)[0]))
                                            W_corr = tfa.image.transform(tf.convert_to_tensor(noisecorr, dtype=tf.float32),
                                                         [1, 0, 0, 0, 1, 0, 0, 0], 'BILINEAR', [size_fing[0], size_fing[1]])
                                            XC = (crosscorr_Fingeprint_GPU((tf.expand_dims(tf.expand_dims(W_corr, axis=0), axis=3)), TA_tf,
                                                           norm2, np.shape(TA_tf.numpy())))
                                            ranges = [[(size_fing[0] - noisecorr.shape[0]), (size_fing[1] - noisecorr.shape[1])]]
                                            pcecorr = parallel_PCE(XC.numpy(), len(XC), ranges)
                                            print("PCE after correction: %f" % pcecorr)
                                            if pcecorr > pceres:
                                                start = time.time()
                                                homography, pce, rotation, scaling = calibration_GPU(hom, noise, 0, 0, 0, TA_tf, norm2, np.shape(TA_tf), [1,0,0,0,1,0,0,0], pcecorr)
                                                pce_array.append(pce)
                                                time_array.append(time.time() - start_run)
                                                print('time calibration: ', time.time()-start)
                                                print(homography[0])
                                                oframe = tfa.image.transform(resized, homography,
                                                             'BILINEAR', [np.uint32(np.rint(resized.shape[0]/homography[0])),
                                                                  np.uint32(np.rint(resized.shape[1]/homography[0]))]).numpy()
                                            else:
                                                start = time.time()
                                                homography, pce, rotation, scaling = calibration_GPU(np.zeros((3,3)), noise, 0, 0, 0, TA_tf, norm2, np.shape(TA_tf), [1,0,0,0,1,0,0,0], pceres)
                                                pce_array.append(pce)
                                                time_array.append(time.time() - start_run)
                                                print('time calibration: ', time.time()-start)
                                                oframe = tfa.image.transform(resized, homography,
                                                             'BILINEAR', [np.uint32(np.rint(resized.shape[0]/homography[0])),
                                                              np.uint32(np.rint(resized.shape[1]/homography[0]))]).numpy()
                                        else:
                                                start = time.time()
                                                homography, pce, rotation, scaling = calibration_GPU(np.zeros((3,3)), noise, 0, 0, 0, TA_tf, norm2, np.shape(TA_tf), [1,0,0,0,1,0,0,0], pceres)
                                                pce_array.append(pce)
                                                time_array.append(time.time() - start_run)
                                                print('time calibration: ', time.time()-start)
                                                oframe = tfa.image.transform(resized, homography,
                                                             'BILINEAR', [np.uint32(np.rint(resized.shape[0]/homography[0])),
                                                              np.uint32(np.rint(resized.shape[1]/homography[0]))]).numpy()
                                else:
                                    start = time.time()
                                    homography, pce, rotation, scaling = calibration_GPU(np.zeros((3,3)), noise, 0, 0, 0, TA_tf, norm2, np.shape(TA_tf), [1,0,0,0,1,0,0,0], pceres)
                                    pce_array.append(pce)
                                    time_array.append(time.time() - start_run)
                                    print('time calibration: ', time.time()-start)
                                    oframe = tfa.image.transform(resized, homography,
                                                     'BILINEAR', [np.uint32(np.rint(resized.shape[0]/homography[0])),
                                                          np.uint32(np.rint(resized.shape[1]/homography[0]))]).numpy()
                            else:
                                start = time.time()
                                homography, pce, rotation, scaling = calibration_GPU(np.zeros([3,3]), noise, 0, 0, 0, TA_tf, norm2, np.shape(TA_tf), [1,0,0,0,1,0,0,0], pceres)
                                pce_array.append(pce)
                                time_array.append(time.time() - start_run)
                                print('time calibration: ', time.time()-start)
                                oframe = tfa.image.transform(resized, homography,
                                                 'BILINEAR', [np.uint32(np.rint(resized.shape[0]/homography[0])),
                                                          np.uint32(np.rint(resized.shape[1]/homography[0]))]).numpy()
                            if cv2.waitKey(25) & 0xFF == ord('q'):
                                break
                        else:
                            break
                    if not os.path.exists(FLAGS.output):
                        os.mkdir(FLAGS.output)
                    mdir = {'pce': np.asarray(pce_array)}
                    out_name1 = FLAGS.output + '/' + device2 + '_' + video_path[len(test_set):-4] + '_PCE.mat'
                    savemat(out_name1, mdir)
                    mdir = {'time': np.asarray(time_array)}
                    out_name2 = FLAGS.output + '/' + device2 + '_' + video_path[len(test_set):-4] + '_time.mat'
                    savemat(out_name2, mdir)
