"""_summary_ Calibrate LED colors against known

TODO: 
    Make interactive version with manual sample location options
    send frames to workers to get colors then collect back together for a bit of a speedboost
"""
from itertools import product
from threading import Thread
from time import sleep
from tqdm import tqdm
from copy import copy
import queue
import numpy as np
import cv2
import os
import pickle
import example_record
from colors import colors
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Options & Constants
debug = True  #  False  #  
recording_file = 'recording.mp4'
sample_window = 20
height = 1080
width = 1920
max_cal_frames = 500
charuco_margin = 220
measure_loc = 0.6  # Multiple of charuco margin to go beyond images (for TV bevel)
aruco_type = cv2.aruco.DICT_5X5_50
time_shift_bounds = [0, 60]  # num frames, where: neg means truth is behind measured; pos means measured is behind truth
rm_duplicates = False
# n_workers = 4

# Globals
file = None
arucoDict = cv2.aruco.getPredefinedDictionary(aruco_type)
board = cv2.aruco.CharucoBoard((11, 5), 0.128, 0.0768, arucoDict)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
ret = None
mtx = None
dist = None
sample_x_inc = 170
measure_loc = measure_loc + 1

sample_a_y = -0.5*charuco_margin
sample_a_locs = [(sample_x_inc * i, sample_a_y, 0) for i in range(-1,10)]
sample_a_locs = np.array([np.array(loc) for loc in sample_a_locs]) / 1000

sample_b_y = -measure_loc*charuco_margin
sample_b_locs = [(sample_x_inc * i, sample_b_y, 0) for i in range(-1,10)]
sample_b_locs = np.array([np.array(loc) for loc in sample_b_locs]) / 1000


def main() -> None:
    """_summary_ Main entry function
    """
    global file, recording_file, ret, mtx, dist, arucoDict
    file = open('Calibration_Results.txt', 'w')
    
    # If no recording then get or make the example
    if not os.path.isfile(recording_file):
        recording_file = 'Example.avi'
        if not os.path.isfile(recording_file):
            example_record.main()

    # Get camera lens and object perspective calibrations
    ret, mtx, dist, _, _ = get_camera_calibration()
    paint_aoi(mtx, dist)
    
    # Get colors from samples & initial gamma guess
    colors_a, colors_b, partial_gamma = get_colors(mtx, dist)
    
    # Find and apply time shift correction fit (registration)
    print(colors.OKGREEN + "Find color calibration factors" + colors.ENDC)
    colors_a, colors_b, _, se = time_correct(colors_a, colors_b, partial_gamma)
    
    # Outlier rejection
    _, keep_mask = remove_outliers(se)
    colors_a = colors_a[keep_mask]
    colors_b = colors_b[keep_mask]
    colors_b_original = copy(colors_b)
    
    # Find and apply optimization corrections fit (shift, scale, gamma, minimum, maximum, increment function)
    colors_b, _, _, _ = optimze_params(colors_a, colors_b)
    
    # Find and apply hue corrections fit (affine transform)
    colors_b, _, _ = color_transform(colors_a, colors_b)
    
    # Export quality check video 
    export_quality_check(colors_a, colors_b, colors_b_original)
    
    file.close()


def get_camera_calibration():
    # Source: https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/sandbox/ludovic/aruco_calibration_rotation.html
    video_in = cv2.VideoCapture(recording_file)
    video_in_length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    allCorners = []
    allIds = []
    decimator = 0
    print(colors.OKGREEN + "Get camera lens & perspective calibration" + colors.ENDC)
    print(colors.OKBLUE + "  Process Frames" + colors.ENDC)
    cal_frame_count = min(video_in_length, max_cal_frames)
    pbar = tqdm(total=cal_frame_count, unit='Frames')
    
    # Loop through video
    while video_in.isOpened():
        ret, frame = video_in.read()
        
        if not ret:
            break
        
        # Get corners
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, arucoDict)

        # Get refined corners
        if len(corners)>0:
            for corner in corners:
                cv2.cornerSubPix(gray, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator+=1
        pbar.update(1)
        if decimator >= cal_frame_count:
            break
    pbar.close()
    video_in.release()

    imsize = gray.shape
    print(colors.OKBLUE + "  Calculate calibration" + colors.ENDC)
    print(colors.WARNING + "    Progress not availible" + colors.ENDC)  # TODO: maybe add something to show how long it took
    
    # Initial guess cam matrix
    cameraMatrixInit = np.array([[ 6e5,  0, imsize[1]/2],
                                 [ 0,  6e5, imsize[0]/2],
                                 [ 0,    0,           1]])

    # Get cam matrix and distortian coeff
    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)  # cv2.CALIB_USE_INTRINSIC_GUESS + 
    (ret, camera_matrix, distortion_coefficients,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                        charucoCorners=allCorners, charucoIds=allIds, board=board, imageSize=imsize,
                        cameraMatrix=cameraMatrixInit, distCoeffs=distCoeffsInit, flags=flags,
                        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
     
    file.write("Camera calibration \n")
    file.write("Fit error: \n" + str(ret) + "\n")
    file.write("Matrix: \n" + str(camera_matrix) + "\n")
    file.write("Distortion: \n" + str(distortion_coefficients) + "\n")
    file.write("Rotation: \n" + str(rotation_vectors[0]) + "\n")
    file.write("Translation: \n" + str(translation_vectors[0]) + "\n\n")
    
    return ret, camera_matrix, distortion_coefficients, rotation_vectors[0], translation_vectors[0]


def paint_aoi(camera_matrix, distortion_coefficients):
    video_in = cv2.VideoCapture(recording_file)
    video_in_length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    width_  = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_ = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out = cv2.VideoWriter('debug_markers.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (width_,height_))
    print(colors.OKGREEN + "Get marker locations" + colors.ENDC)
    print(colors.OKBLUE + "  Process frames" + colors.ENDC)
    cal_frame_count = min(video_in_length, max_cal_frames)
    pbar = tqdm(total=cal_frame_count, unit='Frames')
    count = 0
    
    # Loop through video
    while video_in.isOpened():
        ret, frame = video_in.read()
        
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers & Draw aruco axes
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, arucoDict)
        if ids is not None and len(ids)>0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.0768, camera_matrix, distortion_coefficients)
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camera_matrix, distortion_coefficients, rvecs[i], tvecs[i], 0.064)
            
        # Detect Board & Draw charuco axes
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, board.getDictionary())
        if ids is not None and len(ids)>0:
            retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board)
            
            if charucoIds is not None and len(charucoIds) > 0:
                cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds)
                
                rvec = np.array((0,0,0), dtype=np.float32)
                tvec = np.array((0,0,0), dtype=np.float32)
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, distortion_coefficients, rvec, tvec)
                
                if valid:
                    cv2.drawFrameAxes(frame, camera_matrix, distortion_coefficients, rvec, tvec, 0.064)
                    
                    # TODO:
                    #   Display TV screen bound, probably use just projectPoints like sample points, but lines not just point and square, though might need these too:
                    #       imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                    #       cv2.getPerspectiveTransform(pts1,pts2)  https://stackoverflow.com/questions/22656698/perspective-correction-in-opencv-using-python
                    #       cv2.warpPerspective(img, M, transformed.shape)
            
                    # Draw A sample locations
                    imgpoints, _ = cv2.projectPoints(sample_a_locs, rvec, tvec, camera_matrix, distortion_coefficients)
                    imgpoints = imgpoints.astype(np.uint32)
                    imgpoints = imgpoints.reshape((-1,2))
                    for imgpoint in imgpoints:
                        try:
                            frame = cv2.rectangle(frame, [imgpoint[0],imgpoint[1]], [imgpoint[0]+sample_window,imgpoint[1]+sample_window], (100, 50, 50), 5)
                        except:
                            pass
                    
                    # Draw B sample locations
                    imgpoints, _ = cv2.projectPoints(sample_b_locs, rvec, tvec, camera_matrix, distortion_coefficients)
                    imgpoints = imgpoints.astype(np.uint32)
                    imgpoints = imgpoints.reshape((-1,2))
                    for imgpoint in imgpoints:
                        try:
                            frame = cv2.rectangle(frame, [imgpoint[0],imgpoint[1]], [imgpoint[0]+sample_window,imgpoint[1]+sample_window], (50, 100, 50), 5)
                        except:
                            pass
        
        video_out.write(frame)
        count+=1
        pbar.update(1)
        
        if count >= max_cal_frames:
            break
        
    video_in.release() 
    video_out.release() 
    pbar.close()
    
    
def get_colors(camera_matrix, distortion_coefficients):

    try: 
        with open('colors.pkl', 'rb') as f:
            (colors_a, colors_b, colors_, video_in_length, n_charuco_detect, n) = pickle.load(f)
            loaded = True
    except:
        loaded = False
        
        
    if not loaded:
    
        video_in = cv2.VideoCapture(recording_file)
        video_in_length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        print(colors.OKGREEN + "Get color samples" + colors.ENDC)
        print(colors.OKBLUE + "  Process frames" + colors.ENDC)
        pbar = tqdm(total=video_in_length, unit='Frames')
        
        colors_a = np.empty((video_in_length,3), dtype=np.float32)
        colors_a[:] = np.nan
        colors_b = np.empty((video_in_length,3), dtype=np.float32)
        colors_b[:] = np.nan
        i = 0
        n_charuco_detect = 0
        
        # Loop through video
        while video_in.isOpened():
            ret, frame = video_in.read()
            
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            # Detect Board
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, board.getDictionary())
            
            if ids is not None and len(ids)>0:
                retval, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(corners, ids, frame, board)
                
                if charucoIds is not None and len(charucoIds) > 0:
                    rvec = np.array((0,0,0), dtype=np.float32)
                    tvec = np.array((0,0,0), dtype=np.float32)
                    valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, camera_matrix, distortion_coefficients, rvec, tvec)
                    
                    if valid:
                        n_charuco_detect += 1
                        
                        # TODO: might not need but keep just in case: img_undist = cv2.undistort(frame,mtx,dist,None)
                        #   Would allow the sample area to be more repesentative and not risk overlap with other areas
                        
                        # Get A sample locations
                        imgpoints, _ = cv2.projectPoints(sample_a_locs, rvec, tvec, camera_matrix, distortion_coefficients)
                        imgpoints = imgpoints.astype(np.uint32)
                        imgpoints = imgpoints.reshape((-1,2))
                        colors_a_ = np.empty((len(imgpoints),3))
                        colors_a_[:] = np.nan
                        
                        for j, imgpoint in enumerate(imgpoints):
                            try:
                                color_a = frame[imgpoint[1]:(imgpoint[1]+sample_window),imgpoint[0]:(imgpoint[0]+sample_window)]
                                colors_a_[j,:] = np.nanmean(color_a.reshape((-1,3)),axis=0)
                                # TODO rm outliers
                            except:
                                pass
                        
                        
                        # Get B sample locations
                        imgpoints, _ = cv2.projectPoints(sample_b_locs, rvec, tvec, camera_matrix, distortion_coefficients)
                        imgpoints = imgpoints.astype(np.uint32)
                        imgpoints = imgpoints.reshape((-1,2))
                        colors_b_ = np.empty((len(imgpoints),3))
                        colors_a_[:] = np.nan
                        
                        for j, imgpoint in enumerate(imgpoints):
                            try:
                                color_b = frame[imgpoint[1]:(imgpoint[1]+sample_window),imgpoint[0]:(imgpoint[0]+sample_window)]
                                colors_b_[j,:] = np.nanmean(color_b.reshape((-1,3)),axis=0)
                                # TODO rm outliers
                            except:
                                pass
                        
                        color_a = np.nanmean(colors_a_,axis=0)
                        color_b = np.nanmean(colors_b_,axis=0)
                        colors_a[i,:] = color_a
                        colors_b[i,:] = color_b
            
            i += 1
            pbar.update(1)
            
            if debug and i >= max_cal_frames*5:
                break
            
        video_in.release()
        pbar.close()
        
        # rm rows with missing vales and duplicates to reduce dataset... though repeats might be nice for weighting.
        colors_ = np.concatenate((colors_a, colors_b), axis=1)
        if rm_duplicates:
            colors_ = np.unique(colors_, axis=0)  # Many removed with constant example, fewer removed fromreal recording
        n = len(colors_)
        colors_ = colors_[~np.isnan(colors_).any(axis=1), :]
        colors_a = colors_[:,0:3]
        colors_b = colors_[:,3:6]
        
        with open('colors.pkl', 'wb+') as f:
            pickle.dump((colors_a, colors_b, colors_, video_in_length, n_charuco_detect, n), f)
            
    # Export logs
    file.write("Color Extract \n")
    file.write("Befor/after frame count: \n" + str((video_in_length, n_charuco_detect)) + "\n")
    file.write("Befor/after sample count: \n" + str((n, len(colors_))) + "\n")
    #file.write("partial gamma: \n" + str(partial_gamma) + "\n\n")
    
    def gamma_fit(x, gamma):
        return ((x / 255) ** (1 / gamma)) * 255
    
    # Export Plot
    fig, ax = plt.subplots(2,2)  # (B, G, R)
    partial_gamma = [0, 0, 0]
    
    g_opt, _ = curve_fit(gamma_fit, colors_a[:,0], colors_b[:,0])
    partial_gamma[0] = g_opt[0]
    a_fit = np.linspace(0, 255, 100)
    b_fit = [gamma_fit(x, g_opt) for x in a_fit]
    ax[0, 0].plot(colors_a[:,0], colors_b[:,0], "b.", markersize=2)
    ax[0, 0].plot(a_fit, b_fit, "k-")
    ax[0, 0].set_xlim([0, 255])
    ax[0, 0].set_ylim([0, 255])
    ax[0, 0].legend(['Points', "Fit γ={:0.2f}".format(g_opt[0])], loc='best')
    
    g_opt, _ = curve_fit(gamma_fit, colors_a[:,1], colors_b[:,1])
    partial_gamma[1] = g_opt[0]
    a_fit = np.linspace(0, 255, 100)
    b_fit = [gamma_fit(x, g_opt) for x in a_fit]
    ax[0, 1].plot(colors_a[:,1], colors_b[:,1], "g.", markersize=2)
    ax[0, 1].plot(a_fit, b_fit, "k-")
    ax[0, 1].set_xlim([0, 255])
    ax[0, 1].set_ylim([0, 255])
    ax[0, 1].legend(['Points', "Fit γ={:0.2f}".format(g_opt[0])], loc='best')
    
    g_opt, _ = curve_fit(gamma_fit, colors_a[:,2], colors_b[:,2])
    partial_gamma[2] = g_opt[0]
    a_fit = np.linspace(0, 255, 100)
    b_fit = [gamma_fit(x, g_opt) for x in a_fit]
    ax[1, 0].plot(colors_a[:,2], colors_b[:,2], "r.", markersize=2)
    ax[1, 0].plot(a_fit, b_fit, "k-")
    ax[1, 0].set_xlim([0, 255])
    ax[1, 0].set_ylim([0, 255])
    ax[1, 0].legend(['Points', "Fit γ={:0.2f}".format(g_opt[0])], loc='best')
    
    a_gray = cv2.cvtColor(colors_a.reshape((-1,1,3)), cv2.COLOR_BGR2GRAY).reshape((-1))
    b_gray = cv2.cvtColor(colors_b.reshape((-1,1,3)), cv2.COLOR_BGR2GRAY).reshape((-1))
    ax[1, 1].plot(a_gray, b_gray, "k.", markersize=2)
    ax[1, 1].set_xlim([0, 255])
    ax[1, 1].set_ylim([0, 255])
    
    fig.savefig("pre-calibration_colors.png", bbox_inches='tight')
    plt.close(fig)  # Gamma shown in image is inital notional before other corrections
    
    return colors_a, colors_b, partial_gamma


def test_fit_cost(a_, b_):
    se = np.array([(a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 for (a, b) in zip(a_, b_)]) ** 0.5
    sse = np.sum(se)
    return sse, se


def time_correct(colors_a, colors_b, partial_gamma):
    # NOTE: Fails to account for potential for dropped frames in the middle, different frame rates, rolling shutter, etc., just handles rough delay. 
    #   Small number of outliers should be fine, but also good to remove them if posible. 
    
    print(colors.OKBLUE + "  Calculate frame delay" + colors.ENDC)
    shift_range = list(range(time_shift_bounds[0], time_shift_bounds[1]))
    sses = np.empty((len(shift_range), 1))
    sses[:] = np.nan
    n = len(colors_a)
    
    # Temporarly adjust initial gamma to help get the best delay fit
    colors_bb = copy(colors_b)  # Prevent mutating back upstream
    colors_bb[0] = (np.float32(colors_bb[0]) / 255) ** (partial_gamma[0]) * 255
    colors_bb[1] = (np.float32(colors_bb[1]) / 255) ** (partial_gamma[1]) * 255
    colors_bb[2] = (np.float32(colors_bb[2]) / 255) ** (partial_gamma[2]) * 255
    
    def shift_n_chop(a_, b_, shift):
        if shift > 0:  # pos means measured is behind truth
            a_ = a_[0:n-shift]  # chop end
            b_ = b_[shift:-1]  # chop start
        elif shift < 0:  # num frames where neg means truth is behind measured
            a_ = a_[-shift:-1]  # chop start
            b_ = b_[0:n+shift]  # chop end
        return a_, b_
    
    for i, shift in enumerate(shift_range):
        a_, b_ = shift_n_chop(colors_a, colors_bb, shift)
        sses[i], _ = test_fit_cost(a_, b_)
    
    idx = np.argmin(sses)
    best_shift = shift_range[idx]
    
    colors_a_adjusted, colors_b_adjusted = shift_n_chop(colors_a, colors_b, best_shift)
    _, se = test_fit_cost(colors_a_adjusted, colors_b_adjusted)
    
    file.write("Frame shift \n")
    file.write("N frames: " + str(best_shift) + "\n")
    file.write("Befor/after sample count: \n" + str((n, len(colors_a))) + "\n\n")
    
    return colors_a_adjusted, colors_b_adjusted, best_shift, se
    

def remove_outliers(x, outlier_factor=1.5):
    # Source: https://blog.finxter.com/how-to-find-outliers-in-python-easily/
    print(colors.OKBLUE + "  Remove outliers" + colors.ENDC)
    n = len(x)
    upper_quartile = np.percentile(x, 75)
    lower_quartile = np.percentile(x, 25)
    IQR = (upper_quartile - lower_quartile)
    keep_range = (max(lower_quartile - IQR * outlier_factor, 0), upper_quartile + IQR * outlier_factor)
    keep_mask = (x >= keep_range[0]) & (x <= keep_range[1])
    x = x[keep_mask]
    
    file.write("Outlier Criteria \n")
    file.write("Keep range (SE): \n" + str(keep_range) + "\n")
    file.write("Befor/after sample count: \n" + str((n, len(x))) + "\n\n")
    
    return x, keep_mask
    
    
def optimze_params(colors_a, colors_b):
    print(colors.OKBLUE + "  Optomizing parameters" + colors.ENDC)
    
    def param_fit(x, shift, scale, gamma, minimum, maximum, increment):
        x = np.minimum(np.maximum(x * scale + shift, 0), 255)
        x = increment * (x / increment).astype('uint8')
        x = ((x.astype('float64') / 255) ** (1 / gamma)) * 255
        x = np.minimum(np.maximum(x, minimum), maximum)
        return x
    
    def param_invert(x, shift, scale, gamma, minimum, maximum, increment):
        # Correct: Shift, scale, gamma
        # TODO: maybe do something with min/max/inc
        x = ((x.astype('float64') / 255) ** (gamma)) * 255
        x = np.minimum(np.maximum((x - shift) / scale, 0), 255)
        return x
    
    p_opt_b, _ = curve_fit(param_fit, colors_a[:,0], colors_b[:,0])
    p_opt_g, _ = curve_fit(param_fit, colors_a[:,1], colors_b[:,1])
    p_opt_r, _ = curve_fit(param_fit, colors_a[:,2], colors_b[:,2])
    
    colors_b_adjusted = copy(colors_b)  # Prevent mutating back upstream
    colors_b_adjusted[:,0] = param_invert(colors_b[:,0], *p_opt_b)
    colors_b_adjusted[:,1] = param_invert(colors_b[:,1], *p_opt_g)
    colors_b_adjusted[:,2] = param_invert(colors_b[:,2], *p_opt_r)
    
    # Export Plot
    fig, ax = plt.subplots(2,2)  # (B, G, R)
    
    a_fit = np.linspace(0, 255, 100)
    b_fit = [param_fit(x, *p_opt_b) for x in a_fit]
    ax[0, 0].plot(colors_a[:,0], colors_b[:,0], "b.", markersize=2)
    ax[0, 0].plot(colors_a[:,0], colors_b_adjusted[:,0], "c.", markersize=1)
    ax[0, 0].plot(a_fit, b_fit, "k-")
    ax[0, 0].set_xlim([0, 255])
    ax[0, 0].set_ylim([0, 255])
    ax[0, 0].legend(['Original', 'Adjusted', "Fit"], loc='best')
    
    a_fit = np.linspace(0, 255, 100)
    b_fit = [param_fit(x, *p_opt_g) for x in a_fit]
    ax[0, 1].plot(colors_a[:,1], colors_b[:,1], "g.", markersize=2)
    ax[0, 1].plot(colors_a[:,1], colors_b_adjusted[:,1], "y.", markersize=1)
    ax[0, 1].plot(a_fit, b_fit, "k-")
    ax[0, 1].set_xlim([0, 255])
    ax[0, 1].set_ylim([0, 255])
    ax[0, 1].legend(['Original', 'Adjusted', "Fit"], loc='best')
    
    a_fit = np.linspace(0, 255, 100)
    b_fit = [param_fit(x, *p_opt_r) for x in a_fit]
    ax[1, 0].plot(colors_a[:,2], colors_b[:,2], "r.", markersize=2)
    ax[1, 0].plot(colors_a[:,2], colors_b_adjusted[:,2], "m.", markersize=1)
    ax[1, 0].plot(a_fit, b_fit, "k-")
    ax[1, 0].set_xlim([0, 255])
    ax[1, 0].set_ylim([0, 255])
    ax[1, 0].legend(['Original', 'Adjusted', "Fit"], loc='best')
    
    a_gray = cv2.cvtColor(colors_a.reshape((-1,1,3)), cv2.COLOR_BGR2GRAY).reshape((-1))
    b_gray = cv2.cvtColor(colors_b.reshape((-1,1,3)), cv2.COLOR_BGR2GRAY).reshape((-1))
    ba_gray = cv2.cvtColor(colors_b_adjusted.reshape((-1,1,3)), cv2.COLOR_BGR2GRAY).reshape((-1))
    ax[1, 1].plot(a_gray, b_gray, "k.", markersize=2)
    ax[1, 1].plot(a_gray, ba_gray, color='0.8', markersize=1)
    ax[1, 1].set_xlim([0, 255])
    ax[1, 1].set_ylim([0, 255])
    ax[1, 1].legend(['Original', 'Adjusted'], loc='best')
    
    fig.savefig("calibration_colors.png", bbox_inches='tight')
    plt.close(fig)
    
    # Check fit
    sse, se = test_fit_cost(colors_a, colors_b_adjusted)
    
    file.write("Paremeter Optimization \n")
    file.write("<shift, scale, gamma, minimum, maximum, increment> \n")
    file.write("Blue fit: \n" + str(p_opt_b) + "\n")
    file.write("Green fit: \n" + str(p_opt_g) + "\n")
    file.write("Red fit: \n" + str(p_opt_r) + "\n")
    file.write("Fit error (SSE, MSE): \n" + str((sse, np.mean(se))) + "\n\n") 
    
    return colors_b_adjusted, p_opt_b, p_opt_g, p_opt_r


def color_transform(colors_a, colors_b):
    print(colors.OKBLUE + "  Hue transform" + colors.ENDC)
    
    # Fit transform
    retval, transform, inliers = cv2.estimateAffine3D(colors_b, colors_a)
    colors_b_adjusted = np.zeros(colors_b.shape)
    
    # Apply transform
    for i, old in enumerate(colors_b):
        old = np.append(old, 1)
        colors_b_adjusted[i,:] = np.matmul(transform, old)
    
    # Scale to limits
    (lower, upper) = (np.min(colors_b_adjusted), np.max(colors_b_adjusted))
    colors_b_adjusted = (((colors_b_adjusted.astype('float64') - lower) / (upper - lower)) * 255).astype('uint8')
    
    # Check fit
    sse, se = test_fit_cost(colors_a, colors_b_adjusted)
    
    file.write("Color calibration \n")
    file.write("N inliers: \n" + str(np.sum(inliers)) + "\n")  
    file.write("Fit error: " + str(retval) + "\n")
    file.write("Transform: \n" + str(transform) + "\n")
    file.write("Limit fix: \n" + str((lower, upper)) + "\n")
    file.write("Transform error (SSE, MSE): " + str((sse, np.mean(se))) + "\n\n") 
    
    # TODO: Print the rest
    #   Hyperion fit: max /rgbwcmy, saturation gain, black threshold, brightness gain
    
    return colors_b_adjusted, retval, transform


def export_quality_check(a, b, o):
    print(colors.OKGREEN + "Export quality check video" + colors.ENDC)
    print(colors.OKBLUE + "  Process Frames" + colors.ENDC)
    
    video_out = cv2.VideoWriter('Quality Check.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))
    blank_image = np.ones((height,width,3), np.uint8)
    half_height = int(height/2)
    half_width = int(width/2)
    blank_top_image = np.ones((half_height,half_width,3), np.uint8)
    pbar = tqdm(total=len(a), unit='Frames')
    
    for a_, b_, o_ in zip(a, b, o):
        frame = (blank_image.reshape((-1,3))*a_.reshape((-1,3))).reshape((height,width,3)).astype(np.uint8)
        frame[0:half_height, 0:half_width] = (blank_top_image.reshape((-1,3))*o_.reshape((-1,3))).reshape((half_height,half_width,3)).astype(np.uint8)
        frame[0:half_height, half_width:] = (blank_top_image.reshape((-1,3))*b_.reshape((-1,3))).reshape((half_height,half_width,3)).astype(np.uint8)
        
        video_out.write(frame)
        pbar.update(1)
    
    video_out.release()
    
    colors_ = np.concatenate((a, o, b), axis=1)
    file.write("Corrected colors \n")
    file.write("Actual/before/after calibration: \n" + str(colors_) + "\n\n")  


if __name__ == "__main__":
    main()
