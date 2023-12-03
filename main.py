import multiprocessing as mp
import time
import tkinter as tk
from pathlib import Path
from tkinter import Tk, ttk
import math

import cv2
from PIL import Image, ImageTk
from PIL.ImageOps import mirror

from utils import check_focus, print_d, ComplianceLabelManager, TransientLabel
from workers import AudioWorker, DelegationWorker

if __name__ == "__main__":
    GUI_UPDATE_INTERVAL = 500  # ms
    REST_DURATION = 5 * 60  # s
    AUDIO_THRESHOLD = 55  # db
    SKIN_THRESHOLD = 20
    skip_rest = True

    save_dir = Path("./saved")
    save_dir.mkdir(parents=True, exist_ok=True)

    patch_size = 256

    cap = cv2.VideoCapture(0)

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
    correct_pose = manager.Array("i", [0] * 6)
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

    first_launch = False
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

        step4 = Image.open("./tutorial-illustrations/step4-cuffposition-artery.jpg")
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
        print("entering endtut")
        while not endtut:
            root.update()
        print("exit")

        if endtut:
            canvas.destroy()
        first_launch = False

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

    label_manager = ComplianceLabelManager(compliance_texts, compliance_labels)

    label = tk.Label(frm)
    label.grid(row=0, column=0, rowspan=len(compliance_labels))

    for index, l in enumerate(compliance_labels):
        l.grid(row=index, column=1, padx=10, pady=5)

    notify_message = TransientLabel(frm)
    notify_message.grid(row=len(compliance_labels), column=0, columnspan=2)

    instr_label = tk.Label(frm, text="Please align your body to the skeleton.")
    instr_label.grid(row=len(compliance_labels) + 1, column=0)

    done_label = tk.Label(frm)
    done_label.grid(row=len(compliance_labels) + 2, column=0)

    prev_successful_frame = None

    imgtk = None

    def update_gui(update_ret=False):
        global imgtk

        if update_ret:
            # Shoulder line
            cv2.line(frame, (161, 70), (91, 70), color=(255, 250, 255), thickness=2)

            # Left elbow line
            cv2.line(frame, (75, 180), (91, 70), color=(255, 250, 255), thickness=2)

            # Right elbow line
            cv2.line(frame, (181, 180), (161, 70), color=(255, 250, 255), thickness=2)

            # Left wrist line
            cv2.line(frame, (50, 230), (75, 180), color=(255, 250, 255), thickness=2)

            # Right wrist line
            cv2.line(frame, (216, 230), (181, 180), color=(255, 250, 255), thickness=2)

            for p in range(0, len(pose_ret), 2):
                curr_col = (0, 0, 255)
                if p >= 6:
                    if correct_pose[(p - 6) // 2] == 1:
                        curr_col = (0, 255, 0)  # Green for success
                    else:
                        curr_col = (255, 0, 0)
                cv2.circle(frame, pose_ret[p : p + 2], 2, curr_col, 2)

            """
                cv2.line(frame,
                         (int(correct_pose[2*p]), int(correct_pose[2*p + 1])),
                         (int(correct_pose[2*p + 2]), int(correct_pose[2*p + 3])),
                         color=(255, 250, 255),
                         thickness=9)
            """
            # cv2.circle(frame, correct_pose[p : p + 2], 2, (255, 0, 255), 1)

            seg_image = Image.open(seg_ret.value).convert("RGB")

        img = Image.fromarray(frame)

        if update_ret:
            img = Image.blend(img, seg_image, 0.5)

        img = mirror(img)

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
        # frame = cv2.flip(frame, 1)

        if not ret or not (focus := check_focus(frame)):
            notify_message.configure(text="Camera is not in focus or not connected")
            label_manager.update()
            update_gui(update_ret=True)
            continue

        if time.time() * 1000.0 - cur_time > GUI_UPDATE_INTERVAL:
            cur_time = time.time() * 1000.0

            cv2.imwrite(
                str(save_dir / "frame.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )
            process_event.set()  # we are ready to pass off the image to the Delegation worker

            while not done_event.is_set():
                if audio_ret.value > AUDIO_THRESHOLD:
                    notify_message.configure(text="Audio level is too high")
                    label_manager.set_err_message("Audio level is too high")

            if error_msg.value != "":
                notify_message.configure(text=error_msg.value)
                label_manager.set_err_message(error_msg.value)
                done_label.config(text="")
            else:
                label_manager.update()
                done_label.config(text="Ready to go!")

            done_event.clear()
            update_gui(update_ret=True)
        else:
            label_manager.update()
            update_gui(update_ret=True)
