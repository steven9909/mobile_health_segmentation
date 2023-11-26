import multiprocessing as mp

import cv2
import numpy as np
import openvino as ov
from PIL import Image
import pyaudio
import wave

from utils import print_d

import torch
from torchvision.transforms.functional import normalize

import math


class ErrorState:
    def __init__(self, error_msg):
        self.error = False
        self.error_msg = error_msg

    def __str__(self):
        return self.error_msg


class SuccessState:
    def __init__(self):
        pass


class Worker:
    def __init__(self, args):
        self.process = mp.Process(target=self.run, args=args)
        self.start()

    def start(self):
        self.process.start()

    def join(self):
        self.process.join()

    def run(self, *args):
        raise NotImplementedError


class DelegationWorker(Worker):
    def __init__(self, process_event, done_event, file_str, pose_ret, seg_ret):
        super().__init__((process_event, done_event, file_str, pose_ret, seg_ret))

    def run(self, process_event, done_event, file_str, pose_ret, seg_ret):
        seg_process_event = mp.Event()
        pose_process_event = mp.Event()
        seg_done_event = mp.Event()
        pose_done_event = mp.Event()

        _ = SegmentationModelWorker(
            seg_process_event, seg_done_event, file_str, seg_ret
        )
        _ = PoseEstimatorWorker(pose_process_event, pose_done_event, file_str, pose_ret)

        while True:
            process_event.wait()
            process_event.clear()
            print_d("Delegation Worker Processing")

            image = cv2.imread(file_str.value)

            seg_process_event.set()
            pose_process_event.set()

            seg_done_event.wait()
            seg_done_event.clear()
            pose_done_event.wait()
            pose_done_event.clear()

            self.validate(pose_ret)

            # seg_image = cv2.imread(seg_ret.value, cv2.IMREAD_GRAYSCALE)
            seg_image = None

            self.skin_tone(pose_ret, seg_image, image)

            # everything is done - get the result from segmentation and pose estimation and validate, scale, decide...

            done_event.set()
            print_d("Delegation Done")

    def getoverlay(self, pose_ret, optang_es=5, optang_ew=25):
        r_wri = [pose_ret[6], pose_ret[7]]
        r_elb = [pose_ret[8], pose_ret[9]]
        r_sho = [pose_ret[10], pose_ret[11]]

        l_sho = [pose_ret[12], pose_ret[13]]
        l_elb = [pose_ret[14], pose_ret[15]]
        l_wri = [pose_ret[16], pose_ret[17]]

        r_rad_es = math.hypot(abs(r_sho[1] - r_elb[1]), abs(r_sho[0] - r_elb[0]))
        r_rad_ew = math.hypot(abs(r_wri[1] - r_elb[1]), abs(r_wri[0] - r_elb[0]))
        l_rad_es = math.hypot(abs(l_sho[1] - l_elb[1]), abs(l_sho[0] - l_elb[0]))
        l_rad_ew = math.hypot(abs(l_wri[1] - l_elb[1]), abs(l_wri[0] - l_elb[0]))

        l_opt_elbow = [
            l_sho[0] + l_rad_es * math.cos(math.radians(90 - optang_es)),
            l_sho[1] + l_rad_es * math.sin(math.radians(90 - optang_es)),
        ]
        r_opt_elbow = [
            r_sho[0] + r_rad_es * math.cos(math.radians(90 + optang_es)),
            r_sho[1] + r_rad_es * math.sin(math.radians(90 + optang_es)),
        ]

        l_opt_wrist = [
            l_elb[0] + l_rad_ew * math.cos(math.radians(90 - optang_ew)),
            l_elb[1] + l_rad_ew * math.sin(math.radians(90 + optang_ew)),
        ]
        r_opt_wrist = [
            r_elb[0] + r_rad_ew * math.cos(math.radians(90 + optang_ew)),
            r_elb[1] + r_rad_ew * math.sin(math.radians(90 + optang_ew)),
        ]

        return l_opt_elbow, r_opt_elbow, l_opt_wrist, r_opt_wrist

    def validate(self, pose_ret, optang_es=5, optang_ew=25):
        """Validate the output of segmentation model and pose estimation model

        Args:
            pose_ret (int []): array of size 18, containing the pose estimation result
        """
        max_ang = 10  # Leeway, position is ok +-10 degrees

        # 1. Check if all keypoints are in frame
        num_kpts = 0
        missing = []
        kpt_names = [
            "Spine",
            "Neck",
            "Head",
            "Right wrist",
            "Right elbow",
            "Right shoulder",
            "Left shoulder",
            "Left elbow",
            "Left wrist",
        ]

        for kp in range(0, 9, 2):
            # Skip head keypoint
            if kp == 2:
                continue

            # Check if keypoint exists, i.e. not zero; else note it down
            if pose_ret[2 * kp] != 0:
                num_kpts += 1
            else:
                missing.append(kpt_names[kp])

        # Report if the missing keypoints to user
        if num_kpts != 8:
            return ErrorState(f"{', '.join(missing)} not in frame.")

        # 2. Check if both shoulders are level

        # Find angle of shoulders from horizontal
        should_x = abs(pose_ret[12] - pose_ret[10])
        should_y = abs(pose_ret[13] - pose_ret[11])
        should_ang = math.degrees(math.atan(should_y / should_x))

        # If shoulders misaligned, send feedback for how to adjust
        if should_ang > max_ang:
            if pose_ret[13] > pose_ret[11]:
                higher, lower = "left shoulder", "right shoulder"
            else:
                higher, lower = "right shoulder", "left_shoulder"

            return ErrorState(
                f"Your {higher} is higher than your {lower} shoulder. Please make sure you are sitting straight."
            )

        # 3. Check if neck and spine are aligned

        # Get tilt of abdomen (i.e. neck and spine)
        abdomen_x = abs(pose_ret[2] - pose_ret[0])
        abdomen_y = abs(pose_ret[3] - pose_ret[1])
        abdomen_hyp = math.hypot(abdomen_x, abdomen_y)
        abdomen_ang = math.degrees(math.acos(abdomen_y / abdomen_hyp))

        # If abdomen is tilted, send error
        if abdomen_ang > max_ang:
            return ErrorState(
                "Your neck and your spine are not aligned. Please sit straight."
            )

        # 4. Check if valid arm positions

        # Get coordinates of left and right wrist, elbow, and shoulder
        r_wri = [pose_ret[6], pose_ret[7]]
        r_elb = [pose_ret[8], pose_ret[9]]
        r_sho = [pose_ret[10], pose_ret[11]]

        l_sho = [pose_ret[12], pose_ret[13]]
        l_elb = [pose_ret[14], pose_ret[15]]
        l_wri = [pose_ret[16], pose_ret[17]]

        # Calculate elbow-shoulder angles
        r_es_x = abs(r_elb[0] - r_sho[0])  # Right arm
        r_es_y = abs(r_elb[1] - r_sho[1])
        r_es_ang = math.degrees(math.atan(r_es_x / (r_es_y + 1)))

        l_es_x = abs(l_elb[0] - l_sho[0])  # Left arm
        l_es_y = abs(l_elb[1] - l_sho[1])
        l_es_ang = math.degrees(math.atan(l_es_x / (l_es_y + 1)))

        # Calculate elbow wrist angles
        r_ew_x = abs(l_elb[0] - l_wri[0])  # Right arm
        r_ew_y = abs(r_elb[1] - r_wri[1])
        r_ew_ang = math.degrees(math.atan(r_ew_x / (r_ew_y + 1)))

        l_ew_x = abs(l_elb[0] - l_wri[0])  # Left arm
        l_ew_y = abs(l_elb[1] - l_wri[1])
        l_ew_ang = math.degrees(math.atan(l_ew_x / (l_ew_y + 1)))

        # Check if all angles are appropriate
        check = [
            abs(r_es_ang - optang_es) < max_ang,
            abs(l_es_ang - optang_es) < max_ang,
            abs(r_ew_ang - optang_ew) < max_ang,
            abs(l_ew_ang - optang_ew) < max_ang,
        ]

        # If an angle is inappropriate, send back feedback
        if sum(check) > 0:
            check_labels = ["right elbow", "left elbow", "right wrist", "left wrist"]
            misaligned = [check_labels[i] for i, x in enumerate(check) if not x]
            return ErrorState(
                f"The following joints are in the wrong position: {', '.join(misaligned)}"
            )

        return SuccessState()

    def _compare_patches(self, patch1, patch2):
        # Convert patches to a color space better suited for skin color analysis
        patch1_cb = patch1[:, :, 1][patch1[:, :, 1] != 0]
        patch1_cr = patch1[:, :, 2][patch1[:, :, 2] != 0]

        patch2_cb = patch2[:, :, 1][patch2[:, :, 1] != 0]
        patch2_cr = patch2[:, :, 2][patch2[:, :, 2] != 0]

        # Calculate the mean color values of the patches
        mean1_1 = np.mean(patch1_cb)
        mean2_1 = np.mean(patch1_cr)

        mean1_2 = np.mean(patch2_cb)
        mean2_2 = np.mean(patch2_cr)

        # Calculate the Euclidean distance between the mean values
        distance = np.linalg.norm(mean2_1 - mean2_2)
        distance_2 = np.linalg.norm(mean1_1 - mean1_2)

        print(distance)
        print(distance_2)

    def skin_tone(self, pose_ret, seg_image, image, radius=5):
        """Get the skin tone of the person in the image

        Args:
            pose_ret (int []): array of size 18, containing the pose estimation result
            seg_ret (str): path to the segmentation result
            image (np.ndarray): the image
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        neck_x, neck_y = pose_ret[2:4]

        mask_neck = np.zeros(image.shape[:2], dtype="uint8")

        cv2.circle(mask_neck, (neck_x, neck_y - radius), radius, 255, -1)

        patch1 = cv2.bitwise_and(image, image, mask=mask_neck)

        mask_arm = np.zeros(image.shape[:2], dtype="uint8")

        r_elbow_x, r_elbow_y = pose_ret[8], pose_ret[9]
        r_shoulder_x, r_shoulder_y = pose_ret[10], pose_ret[11]

        cv2.line(mask_arm, (r_elbow_x, r_elbow_y), (r_shoulder_x, r_shoulder_y), 255, 3)

        patch2 = cv2.bitwise_and(image, image, mask=mask_arm)

        self._compare_patches(patch1, patch2)

    def scale(self, pose_ret, seg_ret):
        CUFF_LENGTH = 10  # cm
        seg_result = Image.open(seg_ret.value).convert("L")

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
        r_wrist_x, r_wrist_y = pose_ret[6], pose_ret[7]
        r_elbow_x, r_elbow_y = pose_ret[8], pose_ret[9]
        r_shoulder_x, r_shoulder_y = pose_ret[10], pose_ret[11]

    def decide(self):
        pass


class ModelWorker(Worker):
    def __init__(self, args):
        super().__init__(args)

    def block(self, *args):
        raise NotImplementedError

    def pre_block(self, *args):
        raise NotImplementedError

    def run(self, *args):
        self.pre_block(*args)

        process_event = args[0]
        done_event = args[1]

        while True:
            process_event.wait()
            process_event.clear()
            print_d(f"{self.__class__.__name__} Processing")

            self.block(*args)

            done_event.set()
            print_d(f"{self.__class__.__name__} Done")


class SegmentationModelWorker(ModelWorker):
    def __init__(self, process_event, done_event, file_str, seg_ret):
        super().__init__((process_event, done_event, file_str, seg_ret))

    def pre_block(self, process_event, done_event, file_str, seg_ret):
        self.device = torch.device("cpu")
        self.model = torch.hub.load(
            "milesial/Pytorch-UNet",
            "unet_carvana",
            pretrained=False,
            scale=1,
        )
        self.model.load_state_dict(
            torch.load(
                "./segmentation/output/unet.pth",
                map_location=self.device,
            )["model"]
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def block(self, process_event, done_event, file_str, seg_ret):
        r_image = Image.open(file_str.value).convert("RGB")

        image = np.asarray(r_image, dtype=np.float32) / 255

        r_image.close()

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).to(self.device)
        image = normalize(
            image, mean=(0.5687, 0.5434, 0.5152), std=(0.2508, 0.2399, 0.2307)
        )
        image = torch.unsqueeze(image, 0)

        output = self.model(image)

        output = np.transpose(
            np.squeeze(torch.sigmoid(output.detach()).cpu().numpy(), 0), (1, 2, 0)
        )[:, :, 1]

        out = Image.fromarray(output, mode="L")
        out.save(seg_ret.value)
        out.close()


class PoseEstimatorWorker(ModelWorker):
    def __init__(self, process_event, done_event, file_str, pose_ret):
        super().__init__((process_event, done_event, file_str, pose_ret))

    def pre_block(self, process_event, done_event, file_str, pose_ret):
        core = ov.Core()

        model = core.read_model(model="./pose/model/human-pose-estimation.xml")
        self.compiled_model = core.compile_model(model=model, device_name="CPU")

    def _extract_keypoints(self, heatmap, min_confidence=-100):
        ind = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
        if heatmap[ind] < min_confidence:
            ind = (-1, -1)
        else:
            ind = (int(ind[1]), int(ind[0]))
        return ind

    def block(self, process_event, done_event, file_str, pose_ret):
        infer_request = self.compiled_model.create_infer_request()

        image = Image.open(file_str.value)
        image.load()
        image = np.asarray(image, dtype=np.float32)

        image = np.expand_dims(image, axis=0)
        image = np.ascontiguousarray(np.transpose(image, (0, 3, 1, 2)))

        input_tensor = ov.Tensor(array=image, shared_memory=True)
        infer_request.set_input_tensor(input_tensor)

        infer_request.start_async()
        infer_request.wait()

        output = infer_request.get_output_tensor()
        output_buffer = output.data

        output_buffer = np.squeeze(output_buffer, axis=0)
        output_buffer = np.transpose(output_buffer, axes=(1, 2, 0))
        heatmaps = cv2.resize(
            output_buffer, (256, 256), fx=8, fy=8, interpolation=cv2.INTER_CUBIC
        )

        keypoints = []
        for kpt_idx in range(16):
            keypoints.append(self._extract_keypoints(heatmaps[:, :, kpt_idx]))

        for id in range(7, 16):
            keypoint = keypoints[id]
            if keypoint[0] != -1:
                pose_ret[2 * (id - 7)] = int(keypoint[0])
                pose_ret[2 * (id - 7) + 1] = int(keypoint[1])


class AudioWorker(Worker):
    def __init__(self, process_event, done_event, audio_ret):
        super().__init__((process_event, done_event, audio_ret))

    def run(process_event, done_event, audio_ret):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 3
        WAVE_OUTPUT_FILENAME = "recordedFile.wav"
        audio = pyaudio.PyAudio()

        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=0,
            frames_per_buffer=CHUNK,
        )
        record_frames = []

        while True:
            for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                data = stream.read(CHUNK)
                record_frames.append(data)

            stream.stop_stream()
            stream.close()
            audio.terminate()

            waveFile = wave.open(WAVE_OUTPUT_FILENAME, "wb")
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b"".join(record_frames))
            waveFile.close()
