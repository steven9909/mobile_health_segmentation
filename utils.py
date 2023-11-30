import cv2
import time
import tkinter as tk


def check_focus(image, threshold=10):
    """Check whether the image is in focus or not using variance of Laplacian

    Args:
        image (UMat): image array

    Returns:
        bool: True if the image is in focus, False otherwise
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    variance = cv2.Laplacian(image, cv2.CV_64F).var()

    return variance > threshold


class TransientLabel(tk.Label):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._job = None
        self.job_ongoing = False
        super().configure(fg="red")

    def set_transient(self, transient_time):
        self._job = self.after(transient_time, self._clear_transient)

    def _clear_transient(self):
        super().configure(text="")
        self.job_ongoing = False

    def configure(self, *args, **kwargs):
        if kwargs.get("text") != "" and self.job_ongoing:
            return
        elif kwargs.get("text") != "":
            self.job_ongoing = True
            self.set_transient(3000)
        super().configure(*args, **kwargs)


class ComplianceLabelManager:
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


def print_d(str):
    if __debug__:
        print(str)
