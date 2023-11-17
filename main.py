from tkinter import messagebox, Tk, ttk

import multiprocessing as mp
from pathlib import Path
from ctypes import c_wchar_p

import cv2
import numpy as np

from workers import DelegationWorker
import time

from utils import check_focus


def show_error_box(error_msg):
    messagebox.showerror("ERROR", error_msg)


if __name__ == "__main__":
    frame_width = 1440
    frame_height = 960

    GUI_UPDATE_INTERVAL = 1000  # ms

    save_dir = Path("./saved")
    save_dir.mkdir(parents=True, exist_ok=True)

    patch_size = 256

    cap = cv2.VideoCapture(0)
    cap.set(3, frame_width)
    cap.set(4, frame_height)

    process_event = mp.Event()
    done_event = mp.Event()

    manager = mp.Manager()
    file_str = manager.Value(c_wchar_p, str(save_dir / "frame.jpg"))

    """
        spi
        nec
        hea
        r_wri
        r_elb
        r_sho
        l_sho
        l_elb
        l_wri
    """
    pose_ret = mp.Array("i", [0] * 18)
    seg_ret = mp.Value(c_wchar_p, str(save_dir / "seg.png"))

    delegation = DelegationWorker(
        process_event, done_event, file_str, pose_ret, seg_ret
    )

    print("Starting main loop")

    root = Tk()
    frm = ttk.Frame(root, padding=10)
    frm.grid()
    ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
    ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)

    cur_time = time.time() * 1000.0

    while True:
        # update GUI
        if time.time() * 1000.0 - cur_time > GUI_UPDATE_INTERVAL:
            cur_time = time.time() * 1000.0
            root.update()

        ret, frame = cap.read()

        if not ret or not (focus := check_focus(frame)):
            show_error_box(
                f"Camera is not in focus or not connected, camera status={ret}, is focused={focus}"
            )
            continue

        frame = cv2.resize(
            frame, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC
        )
        cv2.imwrite(str(save_dir / "frame.jpg"), frame)
        process_event.set()  # we are ready to pass off the image to the Delegation worker
        done_event.wait()  # wait for the delegation worker to finish
        done_event.clear()  # clear the flag

        # pose_ret contains the pose information
        # seg_ret points to segmentation result
        print(seg_ret.value)
        print(pose_ret[:])
