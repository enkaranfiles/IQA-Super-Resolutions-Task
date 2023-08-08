import math
from os.path import join
import shutil
import traceback
import time
import cv2
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import lpips
from piq import brisque, total_variation
from eval_utils import normalize, append_timestamp, setup_output_folder, append_result, save_inferred_image
from eval_utils import append_result_int, cv2torch


class BaseMetric:
    """Base class for quantitative evaluation metrics"""

    def __init__(self, name, no_ref=False):
        self.scores = []
        self.name = name
        self.no_ref = no_ref
        self.updated = 0
        self.image_queue = []
        self.ref_queue = []
        self.batch_size = 4
        self.total_calculation_time = 0

    def reset(self):
        self.scores = []
        self.image_queue = []
        self.ref_queue = []
        self.updated = 0
        self.total_calculation_time = 0

    def finish_queue(self):
        self.updated = 0

    def get_num_updated(self):
        return self.updated

    def calculate(self, img, ref):
        raise NotImplementedError

    def update(self, img, ref=None):
        self.updated = 0
        start_time = time.time()
        score = self.calculate(img, ref)
        self.total_calculation_time += time.time() - start_time
        if not isinstance(score, list):
            score = [score]
        for s in score:
            if math.isfinite(s) and not math.isnan(s):
                self.updated += 1
                self.scores.append(s)

    def print_summary(self):
        print("Mean {}: {:.2f}".format(self.name, self.get_mean_score()))

    def get_num_scores(self):
        return len(self.scores)

    def get_all_scores(self):
        return self.scores

    def get_last_score(self):
        return self.scores[-1]

    def get_last_scores(self, n):
        return self.scores[-n:]

    def get_mean_score(self):
        if self.get_num_scores() == 0:
            return -1
        mean_score = sum(self.scores) / self.get_num_scores()
        return mean_score

    def get_name(self):
        return self.name

    def get_total_calculation_time(self):
        return self.total_calculation_time

    def get_mean_calculation_time(self):
        if self.get_num_scores() == 0:
            return -1
        return self.total_calculation_time / self.get_num_scores()


class RmsMetric(BaseMetric):

    def __init__(self):
        super().__init__(name='rms', no_ref=True)

    def calculate(self, img, ref=None):
        score = img.std()
        return score


class MseMetric(BaseMetric):

    def __init__(self):
        super().__init__(name='mse')

    def calculate(self, img, ref):
        score = mse(ref, img)
        return score


class SsimMetric(BaseMetric):

    def __init__(self, gaussian_weights=True, sigma=1.5, use_sample_covariance=False):
        super().__init__(name='ssim')
        self.gaussian_weights = gaussian_weights
        self.sigma = sigma
        self.use_sample_covariance = use_sample_covariance

    def calculate(self, img, ref):
        # Setting win_size and channel_axis explicitly
        score = ssim(ref.astype(np.float64), img.astype(np.float64), gaussian_weights=True, sigma=1.5,
                     use_sample_covariance=False, win_size=7, channel_axis=2, data_range=1.0)
        return score


lpips_fn = lpips.LPIPS()


class LpipsMetric(BaseMetric):

    def __init__(self, network='alex'):
        super().__init__(name='lpips')
        self.lpips_fn = lpips_fn

    def calculate(self, img, ref):
        image_for_lpips = torch.from_numpy(2 * img - 1)
        ref_frame_for_lpips = torch.from_numpy(2 * ref - 1)
        lpips_tensor = self.lpips_fn(image_for_lpips, ref_frame_for_lpips)
        score = lpips_tensor.mean().detach().numpy()
        return score


class BrisquePiqMetric(BaseMetric):

    def __init__(self):
        super().__init__(name='brisque_piq', no_ref=True)

    def calculate(self, img, ref=None):
        img_tensor = cv2torch(img)
        return brisque(img_tensor)


class BrisqueOpenCVMetric(BaseMetric):

    def __init__(self, model_path="brisque_model_live.yml", range_path="brisque_range_live.yml"):
        # pip install opencv-contrib-python
        # wget https://raw.githubusercontent.com/tomqingo/RDGAN/master/brisque_model_live.yml
        # wget https://raw.githubusercontent.com/tomqingo/RDGAN/master/brisque_range_live.yml
        super().__init__(name='brisque_cv', no_ref=True)
        self.cv_brisque = cv2.quality.QualityBRISQUE_create(model_path, range_path)

    def calculate(self, img, ref=None):
        stacked_img = np.stack((np.rint(img * 255),)*3, axis=-1).astype(np.uint8)
        return self.cv_brisque.compute(stacked_img)[0]


class TvMetric(BaseMetric):

    def __init__(self):
        super().__init__(name='tv', no_ref=True)

    def calculate(self, img, ref=None):
        img_tensor = cv2torch(img)
        return total_variation(img_tensor)


class PyIqaMetricFactory:

    def __init__(self):
        import pyiqa
        self.pyiqa = pyiqa
        self.list_of_metrics = pyiqa.list_models()
        self.batch_size = 4

    def get_metric(self, name):
        iqa_metric = self.pyiqa.create_metric(name)
        no_ref = True if iqa_metric.metric_mode == 'NR' else False
        metric_name = 'pyiqa_' + name.lower()

        def inference(self):
            if len(self.image_queue) < 1:
                return []
            img_tensor = torch.cat(self.image_queue[-self.batch_size:])
            if self.ref_queue[0] is not None:
                ref_tensor = torch.cat(self.ref_queue[-self.batch_size:])
                score_tensor = iqa_metric(img_tensor, ref_tensor)
            else:
                score_tensor = iqa_metric(img_tensor)
            self.image_queue = []
            self.ref_queue = []
            score_list = score_tensor.squeeze().tolist()
            if not isinstance(score_list, list):
                score_list = [score_list]
            return score_list

        def finish_queue(self):
            self.updated = 0
            score = self.inference()
            self.updated += len(score)
            self.scores.extend(score)

        def calculate(self, img, ref=None):
            if self.name in ['pyiqa_ahiq', 'pyiqa_maniqa']:
                img = cv2.resize(img, (0, 0), fx=2, fy=2)
                if ref is not None:
                    ref = cv2.resize(ref, (0, 0), fx=2, fy=2)
            img_tensor = cv2torch(img, num_ch=3)
            ref_tensor = None if ref is None else cv2torch(ref, num_ch=3)
            self.image_queue.append(img_tensor)
            self.ref_queue.append(ref_tensor)
            if len(self.image_queue) < self.batch_size:
                return []
            return self.inference()

        pyiqa_metric_class = type('PyIqa_' + name + '_Metric', (BaseMetric,), {'calculate': calculate,
                                                                               'inference': inference,
                                                                               'finish_queue': finish_queue})
        return pyiqa_metric_class(metric_name, no_ref)

    def get_metric_names(self):
        return list(self.list_of_metrics)

    def get_all_metrics(self):
        metrics = []
        for metric_name in self.list_of_metrics:
            if metric_name in ['fid', 'ahiq', 'maniqa']:
                continue
            metrics.append(self.get_metric(metric_name))
        return metrics


# pyiqa_metrics = PyIqaMetricFactory().get_all_metrics()
# pyiqa_metrics = [PyIqaMetricFactory().get_metric('lpips')]
# pyiqa_metrics = [PyIqaMetricFactory().get_metric('maniqa')]
pyiqa_metrics_list = ["brisque", "maniqa", "niqe"]
pyiqa_metrics = []
for metric_name in pyiqa_metrics_list:
    pyiqa_metrics.append(PyIqaMetricFactory().get_metric(metric_name))


class EvalMetricsTracker:
    """ Helper class to calculate and keep track of all the qualitative results and quantitative evaluation metrics."""

    def __init__(self, save_images=False, save_processed_images=False, save_tiff=False, save_scores=False, save_timestamps=False, output_dir=None, color=False,
                 norm='none', hist_eq='none', hist_match=False, quan_eval=True,  quan_eval_start_time=0, quan_eval_end_time=float('inf'),
                 quan_eval_ts_tol_ms=float('inf'), has_reference_frames=False):
        self.save_images = save_images
        self.save_processed_images = save_processed_images
        self.save_tiff = save_tiff
        self.save_scores = save_scores
        self.save_timestamps = save_timestamps
        self.output_dir = output_dir
        self.color = color
        self.norm = norm
        self.hist_eq = hist_eq
        self.hist_match = hist_match
        self.quan_eval = quan_eval
        self.quan_eval_start_time = quan_eval_start_time
        self.quan_eval_end_time = quan_eval_end_time
        self.quan_eval_ts_tol_ms = quan_eval_ts_tol_ms
        self.has_reference_frames = has_reference_frames

        if self.hist_eq == 'none' and self.hist_match is False:
            if self.save_processed_images:
                print("Can not save processed images when hist_match is False and  hist_eq is none")
                self.save_processed_images = False

        if self.save_scores:
            assert self.quan_eval is True, "quan_eval cannot be False when save_scores is True"

        self.metrics = []
        if self.quan_eval:
            mse = MseMetric()
            ssim = SsimMetric()
            lpips = LpipsMetric()
            rms = RmsMetric()
            # brisque_piq = BrisquePiqMetric()
            brisque_cv = BrisqueOpenCVMetric()
            tv = TvMetric()
            self.metrics = [mse, ssim, lpips, rms, brisque_cv, tv]
            self.metrics.extend(pyiqa_metrics)

            if not self.has_reference_frames:
                self.metrics = [m for m in self.metrics if m.no_ref]

        self.reset()

    def reset(self):
        self.setup_output_folders_and_files()
        for metric in self.metrics:
            metric.reset()

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir

    def set_start_end_times(self, quan_eval_start_time, quan_eval_end_time):
        self.quan_eval_start_time = quan_eval_start_time
        self.quan_eval_end_time = quan_eval_end_time

    def save_new_scores(self, idx, metric):
        if not self.save_scores:
            return
        metric_file_path = self.get_metric_file_path(metric)
        num_updated = metric.get_num_updated()
        if num_updated > 0:
            last_scores = metric.get_last_scores(num_updated)
            indices = list(np.arange(idx - num_updated + 1, idx + 1))
            append_result(metric_file_path, indices, last_scores)

    def finalize(self, idx):
        for metric in self.metrics:
            metric.finish_queue()
            self.save_new_scores(idx, metric)

    def update_quantitative_metrics(self, idx, img, ref):
        for metric in self.metrics:
            try:
                if not self.has_reference_frames or metric.no_ref:
                    metric.update(img)
                else:
                    metric.update(img, ref)
                self.save_new_scores(idx, metric)
            except Exception as e:
                print("Exception in metric " + metric.get_name() + ": " + str(e))
                print(traceback.format_exc())
                metric.reset()

    def update(self, idx, img, ref, img_ts, ref_ts=None):
        if ref_ts is None:
            ref_ts = img_ts

        if not self.color:
            img = self.post_process_normalization(img)
            img = np.clip(img, 0.0, 1.0)

        if self.save_timestamps:
            timestamps_file_name = self.get_timestamps_file_path()
            append_timestamp(timestamps_file_name, idx, img_ts)

        if self.save_images:
            save_inferred_image(self.output_dir, img, idx, self.save_tiff)

        if self.has_reference_frames:
            ref = np.clip(ref, 0.0, 1.0)

        if self.hist_match and self.has_reference_frames:
            from skimage import exposure
            img = exposure.match_histograms(img, ref).astype(np.float32)
        else:
            img = self.histogram_equalization(img)
            if self.has_reference_frames:
                ref = self.histogram_equalization(ref)

        if self.save_processed_images:
            save_inferred_image(self.processed_output_dir, img, idx, self.save_tiff)

        inside_eval_cut = self.quan_eval_start_time <= img_ts <= self.quan_eval_end_time
        img_ref_time_diff_ms = abs(ref_ts - img_ts)*1000
        inside_eval_ts_tolerance = img_ref_time_diff_ms <= self.quan_eval_ts_tol_ms
        if self.quan_eval and inside_eval_cut and inside_eval_ts_tolerance:
            self.update_quantitative_metrics(idx, img, ref)

    def save_custom_metric(self, idx, metric_name, metric_value, is_int=False):
        if self.save_scores:
            metric_file_path = join(self.output_dir, metric_name + '.txt')
            if idx == 0:
                open(metric_file_path, 'w').close()  # overwrite with emptiness
            if is_int:
                append_result_int(metric_file_path, idx, metric_value)
            else:
                append_result(metric_file_path, idx, metric_value)

    def get_num_evaluated(self):
        num_evaluated = 0
        for idx, metric in enumerate(self.metrics):
            metric_num_scores = metric.get_num_scores()
            if idx == 0:
                num_evaluated = metric_num_scores
            elif metric_num_scores > num_evaluated:
                print("Number of evaluated frames not equal for each metric")
                num_evaluated = metric_num_scores

        return num_evaluated

    def check_and_print_num_eval(self):
        num_evaluated = self.get_num_evaluated()
        print("Number of evaluated frames: {}".format(num_evaluated))

    def print_summary(self):
        self.check_and_print_num_eval()
        for metric in self.metrics:
            metric.print_summary()

    def get_mean_scores(self):
        mean_scores = {}
        for metric in self.metrics:
            name = metric.get_name()
            mean_score = metric.get_mean_score()
            mean_scores[name] = mean_score
        return mean_scores

    def get_mean_calculation_times(self):
        mean_calculation_times = {}
        for metric in self.metrics:
            name = metric.get_name()
            mean_calculation_time = metric.get_mean_calculation_time()
            mean_calculation_times[name] = mean_calculation_time
        return mean_calculation_times

    def get_metric_names(self):
        metric_names = []
        for metric in self.metrics:
            name = metric.get_name()
            metric_names.append(name)
        return metric_names

    def get_timestamps_file_path(self):
        timestamps_path = join(self.output_dir, 'timestamps.txt')
        return timestamps_path

    def get_metric_file_path(self, metric):
        metric_name = metric.get_name()
        metric_file_path = join(self.output_dir, metric_name + '.txt')
        return metric_file_path

    def setup_output_folders_and_files(self):
        if self.output_dir is None:
            assert self.save_images is False, "save_images cannot be True when output_dir is None"
            assert self.save_processed_images is False, "save_processed_images cannot be True when output_dir is None"
            assert self.save_scores is False, "save_results cannot be True when output_dir is None"
            assert self.save_timestamps is False, "save_timestamps cannot be True when output_dir is None"
        else:
            setup_output_folder(self.output_dir)
            if self.save_processed_images:
                self.processed_output_dir = self.output_dir + "_processed"
                setup_output_folder(self.processed_output_dir)
            if self.save_timestamps:
                timestamps_file_name = self.get_timestamps_file_path()
                open(timestamps_file_name, 'w').close()  # overwrite with emptiness
            if self.save_scores:
                for metric in self.metrics:
                    metric_file_path = self.get_metric_file_path(metric)
                    open(metric_file_path, 'w').close()  # overwrite with emptiness

    def post_process_normalization(self, img):
        if self.norm == 'robust':
            img = normalize(img, 1, 99)
        elif self.norm == 'standard':
            img = normalize(img, 0, 100)
        elif self.norm == 'none':
            pass
        else:
            raise Exception("Unrecognized normalization argument: {}".format(self.hist_eq))
        return img

    def histogram_equalization(self, img):
        if self.hist_eq == 'global':
            from skimage.util import img_as_ubyte, img_as_float32
            from skimage import exposure
            img = exposure.equalize_hist(img)
            img = img_as_float32(img)
        elif self.hist_eq == 'local':
            from skimage.morphology import disk, square
            from skimage.filters import rank
            from skimage.util import img_as_ubyte, img_as_float32
            footprint = disk(55)
            # footprint = square(8)
            img = img_as_ubyte(img)
            img = rank.equalize(img, footprint=footprint)
            img = img_as_float32(img)
        elif self.hist_eq == 'clahe':
            from skimage.util import img_as_ubyte, img_as_float32
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = img_as_ubyte(img)
            img = clahe.apply(img)
            img = img_as_float32(img)
        elif self.hist_eq == 'none':
            pass
        else:
            raise Exception("Unrecognized histogram equalization argument: {}".format(self.hist_eq))
        return img
