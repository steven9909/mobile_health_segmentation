import multiprocessing as mp


class Worker:
    def __init__(self, *args):
        self.process = mp.Process(target=self.run, args=args)
        self.start()

    def start(self):
        self.process.start()

    def join(self):
        self.process.join()

    def run(self, *args):
        raise NotImplementedError


class DelegationWorker(Worker):
    def __init__(self, process_event, done_event):
        super().__init__(process_event, done_event)

    def run(self, process_event, done_event):
        seg_process_event = mp.Event()
        pose_process_event = mp.Event()
        seg_done_event = mp.Event()
        pose_done_event = mp.Event()

        _ = SegmentationModelWorker(seg_process_event, seg_done_event)
        _ = PoseEstimatorWorker(pose_process_event, pose_done_event)

        while True:
            process_event.wait()
            process_event.clear()
            print("Delegation Worker Processing")

            seg_process_event.set()
            pose_process_event.set()

            seg_done_event.wait()
            seg_done_event.clear()
            pose_done_event.wait()
            pose_done_event.clear()

            # everything is done - get the result from segmentation and pose estimation and validate, scale, decide...

            done_event.set()
            print("Delegation Done")

    def validate(self):
        pass

    def scale(self):
        pass

    def decide(self):
        pass


class ModelWorker(Worker):
    def __init__(self, process_event, done_event):
        super().__init__(process_event, done_event)

    def block(self):
        raise NotImplementedError

    def run(self, process_event, done_event):
        while True:
            process_event.wait()
            process_event.clear()
            print(f"{self.__class__.__name__} Processing")

            self.block()

            done_event.set()
            print(f"{self.__class__.__name__} Done")


class SegmentationModelWorker(ModelWorker):
    def __init__(self, process_event, done_event):
        super().__init__(process_event, done_event)

    def block(self):
        pass


class PoseEstimatorWorker(ModelWorker):
    def __init__(self, process_event, done_event):
        super().__init__(process_event, done_event)

    def block(self):
        pass
