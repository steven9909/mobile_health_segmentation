from tkinter import messagebox, Tk, ttk
import tkinter as tk

import multiprocessing as mp
from pathlib import Path

import cv2

from workers import DelegationWorker
import time

from utils import check_focus, print_d

from PIL import Image, ImageTk


def show_error_box(error_msg):
    messagebox.showerror("ERROR", error_msg)


def show_notification(notify_msg):
    messagebox.showinfo("NOTIFICATION", notify_msg)


if __name__ == "__main__":
    frame_width = 1440
    frame_height = 960

    GUI_UPDATE_INTERVAL = 500  # ms
    REST_DURATION = 5 * 60  # s
    skip_rest = True

    save_dir = Path("./saved")
    save_dir.mkdir(parents=True, exist_ok=True)

    patch_size = 256

    cap = cv2.VideoCapture(0)
    cap.set(3, frame_width)
    cap.set(4, frame_height)

    process_event = mp.Event()
    done_event = mp.Event()

    manager = mp.Manager()
    file_str = manager.Value("c", str(save_dir / "frame.jpg"))

    """
        spine
        neck
        head
        r_wrist
        r_elbow
        r_shoulder
        l_shoulder
        l_elbow
        l_wrist
    """
    pose_ret = manager.Array("i", [0] * 18)
    seg_ret = manager.Value("c", str(save_dir / "seg.png"))

    delegation = DelegationWorker(
        process_event, done_event, file_str, pose_ret, seg_ret
    )

    print_d("Starting main loop")

    root = Tk()
    frm = ttk.Frame(root, width=1000, height=1000)
    frm.grid(row=0, column=0, padx=10, pady=2)

    label = tk.Label(frm)
    label.grid(row=0, column=0)
    cap = cv2.VideoCapture(0)

    cur_time = time.time() * 1000.0

    first_launch = True

    while True:
        # update GUI
        if time.time() * 1000.0 - cur_time > GUI_UPDATE_INTERVAL:
            cur_time = time.time() * 1000.0
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            label.configure(image=imgtk)
            root.update()

        if first_launch and not skip_rest:
            show_notification("Please rest and sit still for 5 minutes")
            time.sleep(REST_DURATION)
            first_launch = False

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
