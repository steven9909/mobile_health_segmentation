from tkinter import messagebox, Tk, ttk
import tkinter as tk

import multiprocessing as mp
from pathlib import Path

import cv2

from workers import DelegationWorker, AudioWorker
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

    GUI_UPDATE_INTERVAL = 1  # ms
    REST_DURATION = 5 * 60  # s
    AUDIO_THRESHOLD = 50  # db
    SKIN_THRESHOLD = 15
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
    correct_pose = manager.Array("i", [0] * 8)
    seg_ret = manager.Value("c", str(save_dir / "seg.png"))
    audio_ret = manager.Value("i", 0)
    error_msg = manager.Value("c", "")

    delegation = DelegationWorker(
        process_event,
        done_event,
        file_str,
        pose_ret,
        correct_pose,
        seg_ret,
        error_msg,
        SKIN_THRESHOLD,
    )

    audio = AudioWorker(audio_ret)

    print_d("Starting main loop")

    root = Tk()
    frm = ttk.Frame(root, width=256, height=256)
    frm.grid(row=0, column=0, padx=10, pady=2)

    label = tk.Label(frm)
    label.grid(row=0, column=0)

    notify_label = tk.Label(frm)
    notify_label.grid(row=1, column=0)
    error_label = tk.Label(frm)
    error_label.grid(row=2, column=0)
    audio_label = tk.Label(frm)
    audio_label.grid(row=3, column=0)

    cap = cv2.VideoCapture(0)

    cur_time = time.time() * 1000.0

    first_launch = True

    if first_launch:
        pass

    prev_successful_frame = None

    while True:
        # update GUI
        if (
            time.time() * 1000.0 - cur_time > GUI_UPDATE_INTERVAL
            and prev_successful_frame is not None
        ):
            cur_time = time.time() * 1000.0
            for p in range(0, len(pose_ret), 2):
                cv2.circle(
                    prev_successful_frame, pose_ret[p : p + 2], 2, (0, 255, 0), 2
                )

            for p in range(0, len(correct_pose), 2):
                cv2.circle(
                    prev_successful_frame, correct_pose[p : p + 2], 2, (255, 0, 255), 1
                )

            img = Image.fromarray(prev_successful_frame)
            seg = Image.open(seg_ret.value).convert("L")

            img.paste(seg.convert("RGB"), (0, 0), seg)

            img = img.resize((512, 512), resample=Image.BICUBIC)

            imgtk = ImageTk.PhotoImage(image=img)
            label.configure(image=imgtk)
            root.update()

        ret, frame = cap.read()

        if not ret or not (focus := check_focus(frame)):
            notify_label.configure(text="Camera is not in focus or not connected")
            root.update()
            continue
        else:
            notify_label.configure(text="")

        frame = cv2.resize(
            frame, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC
        )

        prev_successful_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(save_dir / "frame.jpg"), frame)
        process_event.set()  # we are ready to pass off the image to the Delegation worker

        while not done_event.is_set():
            if audio_ret.value > AUDIO_THRESHOLD:
                audio_label.configure(
                    text=f"Audio level = {audio_ret.value}. Please do not talk or laugh."
                )
                audio_label.configure(fg="red")
            else:
                audio_label.configure(text=f"Audio level = {audio_ret.value}")
                audio_label.configure(fg="SystemButtonFace")
            root.update()

        if error_msg.value != "":
            error_label.configure(text=error_msg.value)
            error_label.configure(fg="red")
        else:
            error_label.configure(text="")
            error_label.configure(fg="SystemButtonFace")

        root.update()
        done_event.clear()
