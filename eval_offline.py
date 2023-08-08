import argparse
import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from eval_utils import ensure_dir
from eval_metrics import EvalMetricsTracker

from eval_metrics import SsimMetric, LpipsMetric, MseMetric
def get_eval_start_end_indices(input_folder):
    mse_path = os.path.join(input_folder, "mse.txt")
    with open(mse_path, "r") as f:
        mse_lines = f.readlines()
    start_idx = int(mse_lines[0].split()[0])
    end_idx = int(mse_lines[-1].split()[0])
    return start_idx, end_idx


def get_eval_start_end_times(input_folder):
    start_idx, end_idx = get_eval_start_end_indices(input_folder)
    ts_path = os.path.join(input_folder, "timestamps.txt")
    with open(ts_path, "r") as f:
        ts_lines = f.readlines()
    start_time = float(ts_lines[start_idx].split()[1])
    end_time = float(ts_lines[end_idx].split()[1])
    return start_time, end_time


def get_timestamps(input_folder):
    ts_path = os.path.join(input_folder, "timestamps.txt")
    with open(ts_path, "r") as f:
        ts_lines = f.readlines()
    timestamps = [float(ts_line.split()[1]) for ts_line in ts_lines]
    return timestamps


def main(args):
    input_image_paths = sorted(glob.glob(os.path.join(args.input_folder, "*." + args.input_ext)))
    gt_image_paths = sorted(glob.glob(os.path.join(args.gt_folder, "*.png")))
    if args.output_folder is None:
        output_folder = args.input_folder + "_offline"
        if args.eval_histmatch:
            output_folder = output_folder + "_histmatch"
        elif args.eval_histeq != 'none':
            output_folder = output_folder + "_histeq_" + args.eval_histeq
    else:
        output_folder = args.output_folder
    ensure_dir(output_folder)

    #pred_timestamps = get_timestamps(args.input_folder)
    '''
    if args.eval_keepcut:
        eval_start_time, eval_end_time = get_eval_start_end_times(args.input_folder)
    else:
        eval_start_time, eval_end_time = args.eval_start_time, args.eval_end_time

    if args.post_process_robust_norm:
        norm = 'robust'
    elif args.post_process_norm:
        norm = 'standard'
    else:
        norm = 'none'
    '''


    '''
    eval_metrics_tracker = EvalMetricsTracker(save_images=True, save_processed_images=True,
                                              save_scores=True, save_timestamps=True,
                                              output_dir=output_folder, color=args.color, norm=norm,
                                              hist_eq=args.eval_histeq, hist_match=args.eval_histmatch,
                                              quan_eval=True,
                                              quan_eval_start_time=eval_start_time,
                                              quan_eval_end_time=eval_end_time,
                                              quan_eval_ts_tol_ms=args.eval_ts_tol)
    '''
    '''
    lpips_object = LpipsMetric()
    ssim_object = SsimMetric()
    count = 0
    lpips = 0.0
    ssim = 0.0
    for img_path, gt_path in zip(tqdm(input_image_paths), gt_image_paths):
        count+=1
        #frame_no = int(Path(img_path).stem.lstrip("frame_"))
        #pred_frame_ts = pred_timestamps[frame_no]
        img = cv2.imread(img_path, -1)
        if args.input_ext == 'png':
            img = img.astype(np.float32) / 255.0
        gt = cv2.imread(gt_path, -1)
        if len(gt.shape) == 2:  # if the image is grayscale
            gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
        gt = gt.astype(np.float32) / 255.0
        img_l = img.transpose((2, 0, 1))  # Re-order dimensions to CxHxW
        gt_l = gt.transpose((2, 0, 1))  # Re-order dimensions to CxHxW
        img_l = np.expand_dims(img_l, axis=0)  # Add an extra dimension for batch
        gt_l = np.expand_dims(gt_l, axis=0)  # Add an extra dimension for batch
        lpips += lpips_object.calculate(img_l, gt_l)
        ssim += ssim_object.calculate(img, gt)

        #eval_metrics_tracker.update(frame_no, img, gt, pred_frame_ts)
    print(f"Avg Lpips:" , lpips/count)
    print(f"Avg Ssim:", ssim/count)
    #eval_metrics_tracker.print_summary()
    '''
    lpips_object = LpipsMetric()
    ssim_object = SsimMetric()
    mse_object = MseMetric()

    for subfolder in os.listdir(args.input_folder):
        count = 0
        lpips = 0.0
        ssim = 0.0
        mse = 0.0
        input_subfolder = Path(args.input_folder) / subfolder
        gt_subfolder = Path(args.gt_folder) / subfolder

        input_image_paths = sorted(input_subfolder.glob('*'))  # Assuming all files in the folder are images
        gt_image_paths = sorted(gt_subfolder.glob('*'))
        print(f"\nProcessing folder: {subfolder}...")
        print(f"\nProcessing pair:\n{input_subfolder}\n{gt_subfolder}")
        for img_path, gt_path in zip(tqdm(input_image_paths), gt_image_paths):
            count += 1
            img = cv2.imread(str(img_path), -1)
            if img_path.suffix == '.png':
                img = img.astype(np.float32) / 255.0
            gt = cv2.imread(str(gt_path), -1)
            if len(gt.shape) == 2:  # if the image is grayscale
                gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
            gt = gt.astype(np.float32) / 255.0
            img_l = img.transpose((2, 0, 1))  # Re-order dimensions to CxHxW
            gt_l = gt.transpose((2, 0, 1))  # Re-order dimensions to CxHxW
            img_l = np.expand_dims(img_l, axis=0)  # Add an extra dimension for batch
            gt_l = np.expand_dims(gt_l, axis=0)  # Add an extra dimension for batch
            lpips += lpips_object.calculate(img_l, gt_l)
            ssim += ssim_object.calculate(img, gt)
            mse += mse_object.calculate(img,gt)

        print(f"\n{subfolder}:")
        print(f"Avg Lpips: {lpips / count if count > 0 else 'No images'}")
        print(f"Avg Mse: {mse / count if count > 0 else 'No images'}")
        print(f"Avg Ssim: {ssim / count if count > 0 else 'No images'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='event2im post process')
    parser.add_argument('--input_folder', required=True, type=str, help='path to input images')
    parser.add_argument('--gt_folder', required=True, type=str, help='path to ground truth images')
    parser.add_argument('--input_ext', default='png', type=str, help='input image extension', choices=['png', 'tiff'])
    parser.add_argument('--output_folder', type=str, help='path to output files')
    parser.add_argument('--color', action='store_true', default=False,
                        help='Perform color reconstruction')
    parser.add_argument('--post_process_norm', action='store_true', default=False,
                        help='Apply min max normalization to reconstructed images.')
    parser.add_argument('--post_process_robust_norm', action='store_true', default=False,
                        help='Apply robust normalization to reconstructed images as described in Rebecq20PAMI.')
    parser.add_argument('--eval_keepcut', action='store_true', default=False,
                        help='Use same eval cut (eval start and end times) as the input folder ')
    parser.add_argument('--eval_start_time', type=float, default=0,
                        help='Start time of given event sequence for quantitative evaluation (ignored if eval_keepcut)')
    parser.add_argument('--eval_end_time', type=float, default=float('inf'),
                        help='End time of given event sequence for quantitative evaluation (ignored if eval_keepcut)')
    parser.add_argument('--eval_histeq', default='none', type=str,
                        help='Histogram equalization that will be applied to ground truth and predicted image before quantitative evaluation (default None. ignored if eval_histmatch).',
                        choices=['none', 'global', 'local', 'clahe'])
    parser.add_argument('--eval_histmatch', action='store_true', default=False,
                        help='Match histograms of predicted images to ground truth images before quantitative evaluation (default False)')
    parser.add_argument('--eval_ts_tol', type=float, default=1.0,
                        help='Maximum allowed timestamp difference (in ms) between prediction and ground truth frames '
                             'for quantitative evaluation')
    args = parser.parse_args()
    main(args)
