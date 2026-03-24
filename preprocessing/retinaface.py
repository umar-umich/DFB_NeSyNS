# RetinaFace ONNX face detector (ported from GenD / InsightFace buffalo_l)
# Replaces dlib's FHOG-based detector with a CNN-based detector for better
# recall on profile/occluded/low-quality faces.

import os
import subprocess

import cv2
import numpy as np
import onnxruntime


def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class RetinaFace:
    def __init__(self, model_file=None, session=None, providers=["CPUExecutionProvider"]):
        self.model_file = model_file
        self.session = session
        self.taskname = "detection"
        if self.session is None:
            assert self.model_file is not None
            assert os.path.exists(self.model_file), f"Model file not found: {self.model_file}"
            sess_options = onnxruntime.SessionOptions()
            sess_options.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
            sess_options.inter_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
            self.session = onnxruntime.InferenceSession(self.model_file, sess_options)
            self.session.set_providers(providers)
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = [o.name for o in outputs]
        self.input_name = input_name
        self.output_names = output_names
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(["CPUExecutionProvider"])
        nms_thresh = kwargs.get("nms_thresh", None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get("det_thresh", None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get("input_size", None)
        if input_size is not None:
            if self.input_size is not None:
                print("warning: det_size is already set in detection model, ignore")
            else:
                self.input_size = input_size

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / self.input_std, input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + fmc] * stride
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None, max_num=0, metric="default"):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0],
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == "max":
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


def cuda_available():
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _ensure_cuda_libs_available():
    """
    Add nvidia pip package lib dirs to LD_LIBRARY_PATH and pre-load libcudnn
    so ONNX Runtime CUDAExecutionProvider can find all required .so files
    (cuDNN, cuBLAS, cuFFT, etc.) installed via pip nvidia-* packages.
    """
    import ctypes
    import glob as _glob
    try:
        import nvidia
        nvidia_base = os.path.dirname(nvidia.__file__)
        # Collect all lib/ dirs under nvidia pip packages
        lib_dirs = _glob.glob(os.path.join(nvidia_base, "*/lib"))
        if lib_dirs:
            extra = ":".join(lib_dirs)
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if extra not in ld_path:
                os.environ["LD_LIBRARY_PATH"] = extra + (":" + ld_path if ld_path else "")
        # Pre-load libcudnn so dlopen inside ONNX Runtime finds it
        cudnn_so = os.path.join(nvidia_base, "cudnn", "lib", "libcudnn.so.9")
        if os.path.exists(cudnn_so):
            ctypes.CDLL(cudnn_so, mode=ctypes.RTLD_GLOBAL)
    except (ImportError, OSError) as e:
        print(f"[RetinaFace] Could not set up CUDA libs: {e}. Will fall back to CPU.")


def prepare_model(det_thres=0.5, nms_thresh=0.4, model_dir=None) -> RetinaFace:
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), "weights", "models", "buffalo_l")
    model_file = os.path.join(model_dir, "det_10g.onnx")

    if not os.path.exists(model_file):
        os.makedirs(model_dir, exist_ok=True)
        print(f"Downloading RetinaFace model to {model_file} ...")
        os.system(
            f"wget -q https://huggingface.co/datasets/theanhntp/Liblib/resolve/"
            f"ae4357741af379482690fe3e0f2fa6fd32ba33b4/insightface/models/buffalo_l/"
            f"det_10g.onnx -O {model_file}"
        )

    if cuda_available():
        _ensure_cuda_libs_available()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    model = RetinaFace(model_file, providers=providers)
    model.prepare(
        ctx_id=0,
        nms_thresh=nms_thresh,
        input_size=(640, 640),
        det_thresh=det_thres,
    )
    return model
