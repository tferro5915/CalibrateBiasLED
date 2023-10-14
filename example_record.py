"""_summary_ Make example calibration recording

TODO:
    time delay
"""
import os.path
from itertools import product
from threading import Thread
from time import sleep
from tqdm import tqdm
import queue
import numpy as np
import cv2
import export
from colors import colors


# Options & Constants
b = {"min": 15, "max": 240, "shift": 10, "scale": 0.90, "gamma": 1.5, "inc": 5}
g = {"min": 5, "max": 250, "shift": 5, "scale": 0.95, "gamma": 1, "inc": 10}
r = {"min": 10, "max": 245, "shift": -10, "scale": 1.10, "gamma": 0.75, "inc": 15}
dt = 2  # time delay in frames
dt_jitter = 2  # timeshift noise
color_jitter = 2  # Image noise
height = 1080
width = 1920
bevel = 200
roi = 200
bit_reflection = 0.01

# Globals
video_in = None
video_out = None
done_read = False
done_gen = False
done_write = False
em_stop = False
blank_top_image = None
video_in_length = 0
frames_in = queue.Queue(maxsize=20)
frames_out = queue.Queue(maxsize=20)
nap_timer = 0.00001
top_height = bevel+roi
total_height = height+top_height


def main() -> None:
    """_summary_ Main entry function
    """
    global video_out, video_in, blank_top_image, total_height, video_in_length
    
    # Get blank image (color channels: B, G, R)
    blank_wall_image = np.ones((roi,width,3), np.uint8)
    blank_bevel_image = np.ones((bevel,width,3), np.uint8) * bit_reflection
    blank_top_image = np.concatenate((blank_wall_image, blank_bevel_image), axis=0)

    # Setup videos
    if not os.path.isfile('Calibration.avi'):
        export.main()
    video_in = cv2.VideoCapture('Calibration.avi')
    video_in_length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
    video_out = cv2.VideoWriter('Example.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,total_height))
    
    # Spin up threads
    reader = Thread(target=do_frame_read)
    generator = Thread(target=do_frame_gen)
    writer = Thread(target=do_frame_write)
    reader.start()
    generator.start()
    writer.start()
    
    # Watch threads
    while not done_read or not done_gen or not done_write:
        sleep(nap_timer*100)
    reader.join()
    generator.join()
    writer.join()
    
    # Let go of files
    video_in.release()
    video_out.release()


def do_frame_read():
    global frames_in, done_read, em_stop
    
    while video_in.isOpened():
        ret, frame = video_in.read()
        
        if not ret:
            break
        
        while frames_in.full():
            sleep(nap_timer)
            
        frames_in.put(frame)
        
    done_read = True
 
    
def do_frame_gen():
    global frames_out, done_gen, em_stop
    print(colors.OKGREEN + "Generate example recording" + colors.ENDC)
    pbar = tqdm(total=video_in_length, unit='Frames')
    
    while not done_read or not frames_in.empty():
        if not frames_in.empty():
            frame_in = frames_in.get()
            color = frame_in[0,0,:]
            
            # Jitter
            noise = np.random.normal(color_jitter, color_jitter, color.shape)
            color = color + noise
            
            # Scale, shift, increment
            color[0] = b["inc"] * np.uint8(min(max(color[0] * b["scale"] + b["shift"], 0), 255) / b["inc"])
            color[1] = g["inc"] * np.uint8(min(max(color[1] * g["scale"] + g["shift"], 0), 255) / g["inc"])
            color[2] = r["inc"] * np.uint8(min(max(color[2] * r["scale"] + r["shift"], 0), 255) / r["inc"])
            
            # Gamma (works in 0-1 range)
            color[0] = ((np.float32(color[0]) / 255) ** (1 / b["gamma"])) * 255
            color[1] = ((np.float32(color[1]) / 255) ** (1 / g["gamma"])) * 255
            color[2] = ((np.float32(color[2]) / 255) ** (1 / r["gamma"])) * 255
            
            # Min/Max
            color[0] = min(max(color[0], b["min"]), b["max"])
            color[1] = min(max(color[1], g["min"]), g["max"])
            color[2] = min(max(color[2], r["min"]), r["max"])
            
            # Time Delay (dt) 
            # TODO: Add time delay & jitter in frames
                        
            top_image = (blank_top_image.reshape((-1,3))*color.reshape((-1,3))).reshape((top_height,width,3)).astype(np.uint8)
            frame_out = np.concatenate((top_image, frame_in), axis=0)
        
            while frames_out.full():
                sleep(nap_timer)
                
            frames_out.put(frame_out)
            pbar.update(1)
            
        sleep(nap_timer)
        
    pbar.close()
    done_gen = True
    
    
def do_frame_write():
    global video_out, done_write, em_stop
    
    while not done_gen or not frames_out.empty():
        if not frames_out.empty():
            frame = frames_out.get()
            video_out.write(frame)
        sleep(nap_timer)
        
    done_write = True


if __name__ == "__main__":
    main()
