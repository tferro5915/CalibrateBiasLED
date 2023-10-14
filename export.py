"""_summary_ Make calibration video
"""
from itertools import product
from threading import Thread
from time import sleep
from tqdm import tqdm
from copy import copy
import scipy, scipy.stats
import queue
import numpy as np
import cv2
import random
from colors import colors

# Options & Constants
include_statics = True
include_fades = True
include_mixes = True
include_aruco = True
height = 1080
width = 1920
fade_count = 50
mix_count = 50
mix_max_combo = 500
frames_per_static = 60
frames_per_fade = 10
frames_per_mix = 30
cal_frames = 300
aruco_type = cv2.aruco.DICT_5X5_50
charuco_margin = 200
nap_timer = 0.00001
font = cv2.FONT_HERSHEY_SIMPLEX

# Globals
done_gen = False
done_write = False
frames = queue.Queue(maxsize=20)
video = None
file = None
charuco_board = None
blank_image = None
spans = {}
n_frames = 0


def main() -> None:
    """_summary_ Make calibration video with meta values
    """
    global video, file, charuco_board, blank_image
    
    # Get blank image (color channels: B, G, R)
    blank_image = np.ones((height,width,3), np.uint8)
    spans["Starter"] = ([[128]*3], cal_frames)
    
    # Get max chroma for each: R,G,B,Y,C,M,W,K
    if include_statics:
        static_span = sorted(list(set(product([0, 0, 255],repeat=3))))
        spans["static"] = (static_span, frames_per_static)
    
    # Get faded min-max brightness for each: R,G,B,Y,C,M,W
    if include_fades:
        fade_span_single = np.linspace(0, 255, fade_count)
        fade_span_mask = sorted(list(set(product([0,0,1],repeat=3))))
        fade_span_mask.remove((0,0,0))
        fade_span = np.concatenate([np.outer(fade_span_single,np.array(mask)) for mask in fade_span_mask]).tolist()
        spans["fade"] = (fade_span, frames_per_fade)
    
    # Get color mixes
    # TODO: Fix inherent probability trend towards medium brightness 
    if include_mixes:
        x = np.linspace(0, 255, mix_count)
        y = np.arange(0, len(x))
        dist = scipy.stats.binom.cdf(y, len(y)/2, 0.5)  # Helps reduce over clustering in middle brightness
        mix_span = list(set(product(dist*x,repeat=3)))
        random.shuffle(mix_span)
        mix_span = mix_span[0:mix_max_combo]
        spans["mix"] = (mix_span, frames_per_mix)
        
    # Generate ChAruco board
    if include_aruco:
        arucoDict = cv2.aruco.getPredefinedDictionary(aruco_type)
        charuco = cv2.aruco.CharucoBoard((11, 5), 0.128, 0.0768, arucoDict)
        charuco_board = charuco.generateImage((width-charuco_margin*2,height-charuco_margin*2), blank_image.copy(), 20, 1)
        # Charuco square height = (1080 - 200*2 - 20*2) / 5 = 128 px / square
        # Sqare is 0.128 "m", 1 px = 1 "mm" = 1/1000 "mm", not real units... just used to make math easier in calibration
        charuco_board = np.stack((charuco_board,)*3, axis=-1)
        
    # Setup video out
    video = cv2.VideoWriter('Calibration.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))
    file = open('Calibration.txt', 'w')
    
    # Spin up threads
    generator = Thread(target=do_frame_gen)
    writer = Thread(target=do_frame_write)
    generator.start()
    writer.start()
    
    # Watch threads
    while not done_gen or not done_write:
        sleep(nap_timer*100)
    generator.join()
    writer.join()
    
    # Let go of files
    video.release() 
    file.close()
    
    
def do_frame_gen():
    global frames, file, done_gen
    print(colors.OKGREEN + "Generate calibration" + colors.ENDC)
    
    for key, span_fpc in spans.items():
        print(colors.OKBLUE + "  " + key + colors.ENDC)
        span, fpc = span_fpc
        for color in tqdm(span, unit='Color Step'):
            frame = (blank_image.reshape((-1,3))*np.array(color)).reshape((height,width,3)).astype(np.uint8)
            if include_aruco:
                frame[charuco_margin:-charuco_margin,charuco_margin:-charuco_margin,:] = charuco_board
                
            while frames.full():
                sleep(nap_timer)
                
            frames.put((frame, fpc))
            
            line = ','.join(str(x) for x in color)
            file.write(key + "," + line + '\n')
            
            sleep(nap_timer)
                
    done_gen = True
    
    
def do_frame_write():
    global video, done_write
    cal_frames_ = cal_frames
    
    while not done_gen or not frames.empty():
        if not frames.empty():
            frame, fpc = frames.get()
            for _ in range(fpc):
                cal_frames_-=1
                if cal_frames_ >= 0:
                    frame_ = cv2.putText(copy(frame),"Calibration Remaining: " + str(int(cal_frames_/30)), 
                        (50,height-100), font, 0.98, (150,255,150), 2, 2)
                    video.write(frame_)
                else:
                    video.write(frame)
        sleep(nap_timer)
        
    done_write = True
            

if __name__ == "__main__":
    main()
