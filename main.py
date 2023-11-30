import multiprocessing as mp
import time
import tkinter as tk
from pathlib import Path
from tkinter import Tk, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from utils import check_focus, print_d
from workers import AudioWorker, DelegationWorker


class LabelManager:
    def __init__(self, label_names, labels):
        self.label_names = label_names
        self.labels = labels
        self.timers = [0] * len(labels)

    def set_err_message(self, message):
        for i, label_name in enumerate(self.label_names):
            if label_name.lower() in message.lower():
                self.labels[i].configure(text=f"{label_name}: Not Compliant")
                self.labels[i].configure(foreground="red")
                self.timers[i] = time.time()
            else:
                self.update()

    def update(self):
        for i, timer in enumerate(self.timers):
            if time.time() - timer > 5:
                self.labels[i].configure(text=f"{self.label_names[i]}: Compliant")
                self.labels[i].configure(foreground="green")
                self.timers[i] = 0


if __name__ == "__main__":
    frame_width = 1440
    frame_height = 960

    GUI_UPDATE_INTERVAL = 500  # ms
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

    root.attributes("-fullscreen", True)

    cap = cv2.VideoCapture(0)

    cur_time = time.time() * 1000.0

    first_launch = True
    endtut = False
    if first_launch:
        canvas = tk.Canvas(root, width=1194, height=832)
        canvas.pack()

        # tutorial steps
        step1 = Image.open("./tutorial-illustrations/step1-beforehand.jpg")
        step1 = step1.resize((1194, 832))
        step1 = ImageTk.PhotoImage(image=step1)

        step2 = Image.open("./tutorial-illustrations/step2-toilet.jpg")
        step2 = step2.resize((1194, 832))
        step2 = ImageTk.PhotoImage(image=step2)

        step3 = Image.open("./tutorial-illustrations/step3-posture.jpg")
        step3 = step3.resize((1194, 832))
        step3 = ImageTk.PhotoImage(image=step3)

        step4 = Image.open("./tutorial-illustrations/step4-cuffposition.jpg")
        step4 = step4.resize((1194, 832))
        step4 = ImageTk.PhotoImage(image=step4)

        tutimgs = [step1, step2, step3, step4]
        maxind = len(tutimgs)
        curr_ind = 0

        def end_tut():
            print("skip tut")
            global endtut
            endtut = True

        # Buttons
        skip_button_photo = Image.open("./tutorial-illustrations/skip-button.jpg")
        skip_button_photo = skip_button_photo.resize((140, 61))
        skip_button_photo = ImageTk.PhotoImage(image=skip_button_photo)

        skip_button_label = tk.Label(canvas, image=skip_button_photo, border=0)
        skip_button_label.bind("<Button>", lambda e: end_tut())

        def next_slide():
            global curr_ind
            global prev_button_window
            global next_button_label

            if curr_ind == 0:
                prev_button_label.configure(image=prev_button_photo)
            if curr_ind < maxind - 1:
                curr_ind += 1
            canvas.create_image(597, 416, image=tutimgs[curr_ind])
            if curr_ind == maxind - 1:
                next_button_label.configure(image=done_button_photo)
                next_button_label.bind("<Button>", lambda e: end_tut())
            root.update()

        next_button_photo = Image.open("./tutorial-illustrations/next-button.jpg")
        next_button_photo = next_button_photo.resize((140, 61))
        next_button_photo = ImageTk.PhotoImage(image=next_button_photo)

        next_button_label = tk.Label(canvas, image=next_button_photo, border=0)
        next_button_label.bind("<Button>", lambda e: next_slide())

        def prev_slide():
            global curr_ind
            global prev_button_label
            global next_button_label

            if curr_ind > 0:
                if curr_ind == 1:
                    prev_button_label.configure(image=noprev_button_photo)

                if curr_ind == maxind - 1:
                    next_button_label.configure(image=next_button_photo)
                    next_button_label.bind("<Button>", lambda e: next_slide())

                curr_ind -= 1
                canvas.create_image(597, 416, image=tutimgs[curr_ind])
            root.update()

        noprev_button_photo = Image.open(
            "./tutorial-illustrations/no-previous-button.jpg"
        )
        noprev_button_photo = noprev_button_photo.resize((140, 61))
        noprev_button_photo = ImageTk.PhotoImage(image=noprev_button_photo)

        prev_button_photo = Image.open("./tutorial-illustrations/previous-button.jpg")
        prev_button_photo = prev_button_photo.resize((140, 61))
        prev_button_photo = ImageTk.PhotoImage(image=prev_button_photo)

        prev_button_label = tk.Label(canvas, image=noprev_button_photo, border=0)
        prev_button_label.bind("<Button>", lambda e: prev_slide())

        done_button_photo = Image.open("./tutorial-illustrations/done-button.jpg")
        done_button_photo = done_button_photo.resize((140, 61))
        done_button_photo = ImageTk.PhotoImage(image=done_button_photo)

        skip_button_window = canvas.create_window(
            180, 800, anchor="se", window=skip_button_label
        )
        next_button_window = canvas.create_window(
            1014, 32, anchor="nw", window=next_button_label
        )
        prev_button_window = canvas.create_window(
            180, 32, anchor="ne", window=prev_button_label
        )

        canvas.create_image(597, 416, image=step1)

        while not endtut:
            root.update()

        first_launch = False

    canvas.destroy()
    print("canvas destroyed")

    # center the widget
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)

    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)

    frm = ttk.Frame(root, width=1500, height=256)
    frm.grid(row=1, column=1)

    compliance_texts = [
        "Spine",
        "Neck",
        "Left wrist",
        "Left elbow",
        "Left shoulder",
        "Right wrist",
        "Right elbow",
        "Right shoulder",
        "Audio Level",
        "Clothing",
        "Cuff Position",
    ]

    compliance_labels = [
        tk.Label(frm, text=f"{text}: Compliant")
        for i, text in enumerate(compliance_texts)
    ]

    label_manager = LabelManager(compliance_texts, compliance_labels)

    label = tk.Label(frm)
    label.grid(row=0, column=0, rowspan=len(compliance_labels))

    notify_message = tk.Label(frm)
    notify_message.grid(row=1, column=0, columnspan=2)

    for index, l in enumerate(compliance_labels):
        l.grid(row=index, column=1, padx=10, pady=5)

    prev_successful_frame = None

    imgtk = None

    def update_gui(update_ret=False):
        global imgtk

        if update_ret:
            for p in range(0, len(pose_ret), 2):
                cv2.circle(frame, pose_ret[p : p + 2], 2, (0, 255, 0), 2)

            for p in range(0, len(correct_pose), 2):
                cv2.circle(frame, correct_pose[p : p + 2], 2, (255, 0, 255), 1)

            seg = Image.open(seg_ret.value).convert("L")

        img = Image.fromarray(frame)

        if update_ret:
            img.paste(seg.convert("RGB"), (0, 0), seg)

        img = img.resize((512, 512), resample=Image.BICUBIC)

        imgtk = ImageTk.PhotoImage(image=img)
        label.configure(image=imgtk)
        root.update()

    while True:
        ret, frame = cap.read()

        frame = cv2.resize(
            frame, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not ret or not (focus := check_focus(frame)):
            notify_message.configure(text="Camera is not in focus or not connected")
            label_manager.update()
            update_gui(update_ret=True)
            continue

        if time.time() * 1000.0 - cur_time > GUI_UPDATE_INTERVAL:
            cur_time = time.time() * 1000.0

            cv2.imwrite(str(save_dir / "frame.jpg"), frame)
            process_event.set()  # we are ready to pass off the image to the Delegation worker

            while not done_event.is_set():
                if audio_ret.value > AUDIO_THRESHOLD:
                    label_manager.set_err_message("Audio")

            if error_msg.value != "":
                label_manager.set_err_message(error_msg.value)
            else:
                label_manager.update()

            done_event.clear()
            update_gui(update_ret=True)
        else:
            label_manager.update()
            update_gui(update_ret=True)
