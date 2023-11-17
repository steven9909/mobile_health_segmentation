import multiprocessing as mp

import cv2
import numpy as np
import openvino as ov
from PIL import Image

from utils import print_d

import torch
from torchvision.transforms.functional import normalize


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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            seg_process_event.set()
            pose_process_event.set()

            seg_done_event.wait()
            seg_done_event.clear()
            pose_done_event.wait()
            pose_done_event.clear()

            self.validate(pose_ret)

            # everything is done - get the result from segmentation and pose estimation and validate, scale, decide...

            done_event.set()
            print_d("Delegation Done")

    def validate(self, pose_ret):
        """Validate the output of segmentation model and pose estimation model

        Args:
            pose_ret (int []): array of size 18, containing the pose estimation result
        """

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
