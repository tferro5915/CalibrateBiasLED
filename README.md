# About

This is meant to help calibrate TV bias lights like Hyperion, Boblight, Adalight, etc. 

# How to use

0. Check bias light settings: 
    1. Turn off smoothing delay
    2. Turn off backlight min, gamma, gain, brightness compensation, etc. to defaults. 
    3. Make sure controller does not have any settings away from default either, if using an intermediate controller.
1. Run `export.py` to generate calibration video `Calibration.avi`
2. Record video
    1. Play calibration video on your backlit screen
    2. While recording the video with your camera
    3. During the initial few frames you can move your camera around slightly for better lens calibration
    4. After the on-screen countdown try to keep the camera still
    5. Alternatively, you can run `example_record.py` to generate an example of recording
    * *Note:* To use this with other applications not for backlight screens you might need to change sample locations in `calibration.py`
3. Copy recording to computer
4. Rename/convert to `recording.avi` or update `recording_file` variable in `calibration.py`
5. Run `calibration.py`
6. Open `Calibration_Results.txt` to get calibration results

# Taking a break
Sadly my TV broke in the middle of developing this code... So, it will be some time before I get bias lights setup on the new TV and resume working on this. 

It still has some bugs when I was testing it just before the TV met its end.
