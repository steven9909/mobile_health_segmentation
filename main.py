import cv2
from workers import DelegationWorker
import multiprocessing as mp
from pathlib import Path

if __name__ == "__main__":
    frame_width = 1440
    frame_height = 960

    save_dir = Path("./saved/")

    patch_size = 256

    cap = cv2.VideoCapture(0)
    cap.set(3, frame_width)
    cap.set(4, frame_height)

    process_event = mp.Event()
    done_event = mp.Event()

    delegation = DelegationWorker(process_event, done_event)

    while True:
        process_event.set()  # we are ready to pass off the image to the Delegation worker
        done_event.wait()  # wait for the delegation worker to finish
        done_event.clear()  # clear the flag
