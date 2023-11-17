import multiprocessing as mp
from pathlib import Path
from ctypes import c_wchar_p

import cv2

from workers import DelegationWorker
import time


def check_focus(image):
    return True


if __name__ == "__main__":
    frame_width = 1440
    frame_height = 960

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

    while True:
        ret, frame = cap.read()

        if not ret or not check_focus(frame):
            break

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
        time.sleep(2)
