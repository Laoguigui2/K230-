from libs.PipeLine import PipeLine
from libs.YOLO import YOLO11
from libs.Utils import *
from media.sensor import *
import os, sys, gc
import ulab.numpy as np
import image
import time

kmodel_path = "/sdcard/yolo11n_det_320.kmodel"
labels = {0: '钢球'}
model_input_size = [320, 320]

display = "lcd2_4"  # 2.4寸屏

if display == "hdmi":
    display_mode = "hdmi"
    display_size = [1920, 1080]
elif display == "lcd3_5":
    display_mode = "st7701"
    display_size = [800, 480]
elif display == "lcd2_4":
    display_mode = "st7701"
    display_size = [640, 480]

rgb888p_size = [640, 360]

pl = PipeLine(
    rgb888p_size=rgb888p_size,
    display_size=display_size,
    display_mode=display_mode
)
pl.create(sensor=Sensor(id=2, width=640, height=480))

display_size = pl.get_display_size()

confidence_threshold = 0.6
nms_threshold = 0.45

yolo = YOLO11(
    task_type="detect",
    mode="video",
    kmodel_path=kmodel_path,
    labels=labels,
    rgb888p_size=rgb888p_size,
    model_input_size=model_input_size,
    display_size=display_size,
    conf_thresh=confidence_threshold,
    nms_thresh=nms_threshold,
    max_boxes_num=50,
    debug_mode=0,
)

yolo.config_preprocess()

clock = time.clock()
frame_count = 0

try:
    while True:
        clock.tick()
        img = pl.get_frame()
        res = yolo.run(img)
        
        yolo.draw_result(res, pl.osd_img)
   
        obj_count = len(res[0])
        
        current_fps = clock.fps()
        
        info_text = f"FPS:{current_fps:.1f} Count:{obj_count}"
        pl.osd_img.draw_string(5, 5, info_text, color=(0, 255, 0), scale=2)
        
        pl.show_image()
        
        frame_count += 1
        if frame_count % 10 == 0:
            gc.collect()

except Exception as e:
    print(f"Error: {e}")
finally:
    yolo.deinit()
    pl.destroy()
