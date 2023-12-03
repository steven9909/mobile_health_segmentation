import audioop
import math
import multiprocessing as mp

import cv2
import numpy as np
import openvino as ov
import pyaudio
import torch
from PIL import Image
from torchvision.transforms.functional import normalize

from utils import print_d


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
    def __init__(
        self,
        process_event,
        done_event,
        file_str,
        pose_ret,
        correct_pose,
        seg_ret,
        error_msg,
        skin_threshold,
    ):
        super().__init__(
            (
                process_event,
                done_event,
                file_str,
                pose_ret,
                correct_pose,
                seg_ret,
                error_msg,
                skin_threshold,
            )
        )

    def run(
        self,
        process_event,
        done_event,
        file_str,
        pose_ret,
        correct_pose,
        seg_ret,
        error_msg,
        skin_threshold,
    ):
        seg_process_event = mp.Event()
        pose_process_event = mp.Event()
        seg_done_event = mp.Event()
        pose_done_event = mp.Event()

        seg_bounding = mp.Array("i", [0] * 8)

        _ = SegmentationModelWorker(
            seg_process_event, seg_done_event, file_str, seg_ret, seg_bounding
        )
        _ = PoseEstimatorWorker(pose_process_event, pose_done_event, file_str, pose_ret)

        while True:
            process_event.wait()
            process_event.clear()
            print_d("Delegation Worker Processing")

            image = cv2.imread(file_str.value)

            seg_process_event.set()
            pose_process_event.set()

            pose_done_event.wait()
            pose_done_event.clear()

            seg_done_event.wait()
            seg_done_event.clear()

            ret = self.check_position(pose_ret)
            for i in range(len(ret)):
                correct_pose[i] = ret[i]

            state = self.validate(pose_ret)

            if isinstance(state, ErrorState):
                error_msg.value = str(state)
                done_event.set()
                continue
            else:
                error_msg.value = ""

            # for i, (x, y) in enumerate(self.get_overlay(pose_ret)):
            #    correct_pose[2 * i] = int(x)
            #    correct_pose[(2 * i) + 1] = int(y)

            if self.skin_tone(pose_ret, image) > skin_threshold:
                error_msg.value = (
                    "Please make sure you are not wearing any clothing below the cuff."
                )
                done_event.set()
                continue

            seg_image = cv2.imread(seg_ret.value, cv2.IMREAD_GRAYSCALE)

            state = self.scale(pose_ret, seg_bounding, seg_image)

            print_d("Delegation Done")
            done_event.set()

    def get_overlay(self, pose_ret, optang_es=5, optang_ew=25):
        r_wri = (pose_ret[6], pose_ret[7])
        r_elb = (pose_ret[8], pose_ret[9])
        r_sho = (pose_ret[10], pose_ret[11])

        l_sho = (pose_ret[12], pose_ret[13])
        l_elb = (pose_ret[14], pose_ret[15])
        l_wri = (pose_ret[16], pose_ret[17])

        r_rad_es = math.hypot(abs(r_sho[1] - r_elb[1]), abs(r_sho[0] - r_elb[0]))
        r_rad_ew = math.hypot(abs(r_wri[1] - r_elb[1]), abs(r_wri[0] - r_elb[0]))
        l_rad_es = math.hypot(abs(l_sho[1] - l_elb[1]), abs(l_sho[0] - l_elb[0]))
        l_rad_ew = math.hypot(abs(l_wri[1] - l_elb[1]), abs(l_wri[0] - l_elb[0]))

        l_opt_elbow = (
            l_sho[0] + l_rad_es * math.cos(math.radians(90 - optang_es)),
            l_sho[1] + l_rad_es * math.sin(math.radians(90 - optang_es)),
        )
        r_opt_elbow = (
            r_sho[0] + r_rad_es * math.cos(math.radians(90 + optang_es)),
            r_sho[1] + r_rad_es * math.sin(math.radians(90 + optang_es)),
        )

        l_opt_wrist = (
            l_elb[0] + l_rad_ew * math.cos(math.radians(90 - optang_ew)),
            l_elb[1] + l_rad_ew * math.sin(math.radians(90 + optang_ew)),
        )
        r_opt_wrist = (
            r_elb[0] + r_rad_ew * math.cos(math.radians(90 + optang_ew)),
            r_elb[1] + r_rad_ew * math.sin(math.radians(90 + optang_ew)),
        )

        return [l_opt_wrist, l_opt_elbow, l_sho, r_sho, r_opt_elbow, r_opt_wrist]

    def check_on_line(self, xp, yp, x1, y1, x2, y2, maxDistance):
        dxL, dyL = x2 - x1, y2 - y1  # line: vector from (x1,y1) to (x2,y2)
        dxP, dyP = xp - x1, yp - y1  # point: vector from (x1,y1) to (xp,yp)
        dxQ, dyQ = xp - x2, yp - y2  # extra: vector from (x2,y2) to (xp,yp)

        squareLen = dxL * dxL + dyL * dyL  # squared length of line
        dotProd = (
            dxP * dxL + dyP * dyL
        )  # squared distance of point from (x1,y1) along line
        crossProd = (
            dyP * dxL - dxP * dyL
        )  # area of parallelogram defined by line and point

        # perpendicular distance of point from line
        distance = abs(crossProd) / math.sqrt(squareLen)

        # distance of (xp,yp) from (x1,y1) and (x2,y2)
        distFromEnd1 = math.sqrt(dxP * dxP + dyP * dyP)
        distFromEnd2 = math.sqrt(dxQ * dxQ + dyQ * dyQ)

        # if the point lies beyond the ends of the line, check if
        # it's within maxDistance of the closest end point
        if dotProd < 0:
            return distFromEnd1 <= maxDistance
        if dotProd > squareLen:
            return distFromEnd2 <= maxDistance

        # else check if it's within maxDistance of the line
        return distance <= maxDistance

    def check_position(self, pose_ret, max_distance=15):
        # r_wri, r_elb, r_sho, l_sho, l_elb, l_wri,
        # optimal_position = [(216, 230), (181, 180), (161, 70), (91, 70),  (75, 180), (50, 230)]
        optimal_position = [
            (50, 230),
            (75, 180),
            (91, 70),
            (161, 70),
            (181, 180),
            (216, 230),
        ]

        # Get coordinates of left and right wrist, elbow, and shoulder
        curr_pos = [
            (pose_ret[6], pose_ret[7]),
            (pose_ret[8], pose_ret[9]),
            (pose_ret[10], pose_ret[11]),
            (pose_ret[12], pose_ret[13]),
            (pose_ret[14], pose_ret[15]),
            (pose_ret[16], pose_ret[17]),
        ]

        check = []
        for i in range(3):
            check.append(
                self.check_on_line(
                    curr_pos[i][0],
                    curr_pos[i][1],
                    optimal_position[i][0],
                    optimal_position[i][1],
                    optimal_position[i + 1][0],
                    optimal_position[i + 1][1],
                    max_distance,
                )
            )

        for i in range(3, 6):
            check.append(
                self.check_on_line(
                    curr_pos[i][0],
                    curr_pos[i][1],
                    optimal_position[i - 1][0],
                    optimal_position[i - 1][1],
                    optimal_position[i][0],
                    optimal_position[i][1],
                    max_distance,
                )
            )

        return [int(c) for c in check]

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

        for kp in range(0, 9):
            # Skip head keypoint
            if kp * 2 == 4:
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
                higher, lower = "right shoulder", "left shoulder"
            else:
                higher, lower = "left shoulder", "right shoulder"

            return ErrorState(
                f"Your {higher} is higher than your {lower}. Please make sure they are level."
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

        check_positions = self.check_position(pose_ret)

        if sum(check_positions) < 6:
            check_labels = [
                "Right wrist",
                "Right elbow",
                "Right shoulder",
                "Left shoulder",
                "Left elbow",
                "Left wrist",
            ]
            misaligned = [
                check_labels[i] for i, x in enumerate(check_positions) if not x
            ]
            return ErrorState(
                f"The following joints are in the wrong position: {', '.join(misaligned)}"
            )

        """
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
        if sum(check) < 4:
            check_labels = ["right elbow", "left elbow", "right wrist", "left wrist"]
            misaligned = [check_labels[i] for i, x in enumerate(check) if not x]
            return ErrorState(
                f"The following joints are in the wrong position: {', '.join(misaligned)}"
            )
        """

        return SuccessState()

    def _compare_patches(self, patch1, patch2):
        patch1_cb = patch1[:, :, 1][patch1[:, :, 1] != 0]
        patch1_cr = patch1[:, :, 2][patch1[:, :, 2] != 0]

        patch2_cb = patch2[:, :, 1][patch2[:, :, 1] != 0]
        patch2_cr = patch2[:, :, 2][patch2[:, :, 2] != 0]

        mean1_cb, std1_cb = np.mean(patch1_cb), np.std(patch1_cb)
        mean1_cr, std1_cr = np.mean(patch1_cr), np.std(patch1_cr)

        mean2_cb, std2_cb = np.mean(patch2_cb), np.std(patch2_cb)
        mean2_cr, std2_cr = np.mean(patch2_cr), np.std(patch2_cr)

        distance = np.sqrt(
            (mean1_cb - mean2_cb) ** 2
            + (mean1_cr - mean2_cr) ** 2
            + (std1_cb - std2_cb) ** 2
            + (std1_cr - std2_cr) ** 2
        )

        return distance

    def skin_tone(self, pose_ret, image, radius=5):
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

        return self._compare_patches(patch1, patch2)

    def _is_point_on_line_segment(self, x, y, x1, y1, x2, y2):
        return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

    def _arm_aligned_helper(self, m, c, rect):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = rect

        edges = [
            ((x1, y1), (x2, y2)),
            ((x2, y2), (x3, y3)),
            ((x3, y3), (x4, y4)),
            ((x4, y4), (x1, y1)),
        ]

        for (x1, y1), (x2, y2) in edges:
            if x1 == x2:
                y_int = m * x1 + c
                if self._is_point_on_line_segment(x1, y_int, x1, y1, x2, y2):
                    return True
            else:
                m_edge = (y2 - y1) / (x2 - x1)
                c_edge = y1 - m_edge * x1

                x_int = (c_edge - c) / (m - m_edge)
                y_int = m * x_int + c

                if self._is_point_on_line_segment(x_int, y_int, x1, y1, x2, y2):
                    return True

        return False

    def _is_arm_aligned(self, elbow_point, shoulder_point, cuff_bounding_box):
        x, y = elbow_point
        x1, y1 = shoulder_point

        m = (y1 - y) / (x1 - x) if x1 != x else float("inf")
        c = y - m * x if m != float("inf") else x

        return self._arm_aligned_helper(m, c, cuff_bounding_box)

    def _distance_from_elbow(self, elbow, cuff_line, seg_image):
        x0, y0 = elbow
        (x1, y1), (x2, y2) = cuff_line
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2

        m = (midpoint_y - y0) / (midpoint_x - x0) if midpoint_x != x0 else float("inf")
        c = y0 - m * x0 if m != float("inf") else x0

        for x in np.arange(x0 - 25, x0 + 25, 0.05):
            y = m * x + c
            if y < 0 or y > 255:
                continue

            if seg_image[int(y), int(x)] == 255:
                return math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        else:
            return -1

    def scale(self, pose_ret, seg_bounding, seg_image):
        CUFF_LENGTH = 138.35  # mm

        r_elbow_x, r_elbow_y = pose_ret[8], pose_ret[9]
        r_shoulder_x, r_shoulder_y = pose_ret[10], pose_ret[11]

        # [bottom-left, top-left, top-right, bottom-right]
        if not (
            self._is_arm_aligned(
                (r_elbow_x, r_elbow_y),
                (r_shoulder_x, r_shoulder_y),
                [
                    seg_bounding[0:2],
                    seg_bounding[2:4],
                    seg_bounding[4:6],
                    seg_bounding[6:8],
                ],
            )
        ):
            return ErrorState(
                "Please make sure your cuff is on your right upper arm at heart level."
            )

        dist_pixels = self._distance_from_elbow(
            (r_elbow_x, r_elbow_y), [seg_bounding[0:2], seg_bounding[6:8]], seg_image
        )

        if dist_pixels == -1:
            return ErrorState("Please make sure your cuff is aligned.")

        cuff_left_side = math.sqrt(
            (seg_bounding[1] - seg_bounding[3]) ** 2
            + (seg_bounding[0] - seg_bounding[2]) ** 2
        )

        cuff_right_side = math.sqrt(
            (seg_bounding[5] - seg_bounding[7]) ** 2
            + (seg_bounding[4] - seg_bounding[6]) ** 2
        )

        cuff_length_pixels = (
            cuff_left_side if (cuff_left_side > cuff_right_side) else cuff_right_side
        )

        print(f"Cuff length is: {cuff_length_pixels}")
        print(f"Distance from cuff to elbow is {dist_pixels}")

        actual_length = (dist_pixels / cuff_length_pixels) * CUFF_LENGTH

        if actual_length < 5:
            return ErrorState(
                "Please make sure your cuff position is higher on your arm."
            )
        elif actual_length > 50:
            return ErrorState(
                "Please make sure your cuff position is lower on your arm."
            )
        else:
            return SuccessState()


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
    def __init__(self, process_event, done_event, file_str, seg_ret, seg_bounding):
        super().__init__((process_event, done_event, file_str, seg_ret, seg_bounding))

    def _get_largest_island(self, binary_mask):
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        sizes = stats[1:, -1]
        if len(sizes) == 0:
            return np.zeros_like(binary_mask)

        max_label = np.argmax(sizes) + 1

        largest_island = np.zeros_like(binary_mask, dtype=np.uint8)
        largest_island[labels == max_label] = 255

        return largest_island

    def _find_minimum_rectangle(self, binary_mask):
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        all_points = np.vstack(contours)

        rect = cv2.minAreaRect(all_points)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        return box

    def pre_block(self, process_event, done_event, file_str, seg_ret, seg_bounding):
        self.device = torch.device("cpu")
        self.model = torch.hub.load(
            "milesial/Pytorch-UNet",
            "unet_carvana",
            pretrained=False,
            scale=1,
        )
        self.model.load_state_dict(
            torch.load(
                "./segmentation/output/unet_dice_12_02_late",
                map_location=self.device,
            )["model"]
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def block(self, process_event, done_event, file_str, seg_ret, seg_bounding):
        r_image = Image.open(file_str.value).convert("RGB")
        image = np.asarray(r_image, dtype=np.float32) / 255

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).to(self.device)
        image = normalize(
            image, mean=(0.5742, 0.5423, 0.5122), std=(0.2504, 0.2408, 0.2302)
        )
        image = torch.unsqueeze(image, 0)

        segmented = self.model(image)
        segmented = (
            np.transpose(
                np.squeeze((torch.sigmoid(segmented.detach()) > 0.2).cpu().numpy(), 0),
                (1, 2, 0),
            )[:, :, 1]
            * 255
        ).astype(np.uint8)

        largest_island_mask = self._get_largest_island(segmented)
        rectangle = self._find_minimum_rectangle(largest_island_mask)

        if rectangle is not None:
            for i in range(4):
                x = rectangle[i][0]
                y = rectangle[i][1]
                seg_bounding[2 * i] = x
                seg_bounding[2 * i + 1] = y

        output = Image.fromarray(segmented, mode="L")
        output.save(seg_ret.value)
        output.close()


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

        image = Image.open(file_str.value).convert("RGB")
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
    def __init__(self, audio_ret):
        super().__init__([audio_ret])

    def _measure_rms(self, data):
        rms = audioop.rms(data, 2)
        decibels = 20 * math.log10(rms) if rms > 0 else 0
        return int(decibels)

    def run(self, audio_ret):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024
        audio = pyaudio.PyAudio()

        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )

        while True:
            data = stream.read(CHUNK)
            audio_ret.value = self._measure_rms(data)
