'''
#  ----------------------------------------------------------------------
#  Executable code for mitosis detection result evaluation
#
#  ---------------------------------------------------------------------- 
#  Copyright (c) 2018 Yao Lu
#  ----------------------------------------------------------------------
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
#  OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#  HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#  OTHER DEALINGS IN THE SOFTWARE.
#  ---------------------------------------------------------------------- 
#  Contact: Yao Lu at <hiluyao@tju.edu.cn> or <hiluyao@gmail.com>
#  ----------------------------------------------------------------------
#  ----------------------------------------------------------------------

Given mitosis detection results and ground truth labels in the same form of an N x 3 array (t, x, y),
this code computes the precision, recall for the detection reuslts.
'''
import os
import sys
import math
import numpy as np
import pdb
from skimage import io
from pathlib import Path
from tqdm import tqdm


##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"

lap = 0
##

def make_paths(root_path, check_num, lap):
    ps = []
    # ps.append(f"/home/fujii/hdd/BF-C2DL-HSC/02/gt_plots_02.txt")
    ps.append(os.path.join(root_path, f"check{check_num}/testdata/GT"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_eval/peak/peak.txt"))

    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_eval/evaluations/detp"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_eval/evaluations/gtp"))
    ps.append(os.path.join(root_path, f"check{check_num}/lap{lap}/pred_to_eval/evaluations"))

    return ps

def load_files(paths):
    fs = []
    fs.append(sorted(Path(paths[0]).glob("*.txt")))
    # fs.append(sorted(Path(paths[1]).glob("*.txt")))
    
    fs_lens = [len(file) for file in fs]
    assert len(list(set(fs_lens))) == 1, "files number error"
    return fs

def evaluate_detection(detection_result: np.ndarray, gt_labels: np.ndarray, config=None):
    '''
    :param detection_result: detection result to be evaluated, an N x 3 array, with (t, x, y) in each row
    :param gt_labels: ground truth lables, M x 3 array, with the same format of detection_result
    :config: a dictionary containing necessary params for evaluation
    :return: precision, recall
    '''
    xy_dist_threshold_default = 15
    return_detection_result = False
    if config is not None:
        if 'xy_dist_threshold' not in config.keys():
            xy_dist_threshold = xy_dist_threshold_default
        else:
            xy_dist_threshold = config['xy_dist_threshold']
        if 'return_detection_result' in config.keys():
            return_detection_result = config['return_detection_result']
    else:
        xy_dist_threshold = xy_dist_threshold_default
        pass
    # check input format
    if len(detection_result.shape) != 2:
        raise TypeError('Invalid input shape: detection_result: ' + str(detection_result.shape))
    if detection_result.shape[1] != 3:
        if detection_result.shape[1] == 4:
            seq_id_list = detection_result[:, 3]
            detection_result = detection_result[:, 0:3]
            pass

    # add additional column to mark detection status in gt_labels and detection_result, [x, y, status_id, pair_id]
    # status id: 0 as False Positive, 1 as True Positive, 2 as False Negative
    gt_labels = np.concatenate((gt_labels, np.zeros((gt_labels.shape[0], 2))), axis=1)
    detection_result = np.concatenate((detection_result, np.zeros((detection_result.shape[0], 2))), axis=1)
    # pdb.set_trace()
    # search for nearest ground truth labels for each detection result coordinate
    for det_idx, detection_coord in enumerate(detection_result):
        det_gt_dist = []
        det_x = detection_coord[0]
        det_y = detection_coord[1]
        for gt_idx, gt_coord in enumerate(gt_labels):
            gt_x = gt_coord[0]
            gt_y = gt_coord[1]

            xy_dist = math.sqrt((det_x - gt_x) ** 2 + (det_y - gt_y) ** 2)

            det_gt_dist.append({'xy_dist': xy_dist, 'gt_idx': gt_idx})
            pass
        # find the nearest gt label for the current detection
        det_gt_dist.sort(key=lambda x: x['xy_dist'])
        #  if文を満たすようなdet_gt_distの結果を持つジェネレータを生成
        det_gt_dist_filter = [_ for _ in det_gt_dist if abs(_['xy_dist']) < xy_dist_threshold]
        if len(det_gt_dist_filter) == 0:
            # mark current detection as False Positive
            detection_result[det_idx, 2] = 0
            detection_result[det_idx, 3] = 0
            pass
        else:
            if False:
                pass
            else:
                gt_idx = det_gt_dist_filter[0]['gt_idx']
                gt_idx = int(gt_idx)
                if gt_labels[gt_idx, 2] > 0:
                    # compare distance with previous chosen detection
                    gt_x = gt_labels[gt_idx, 0]
                    gt_y = gt_labels[gt_idx, 1]

                    det_idx_pre = gt_labels[gt_idx, 3]
                    det_idx_pre = int(det_idx_pre)
                    det_x_pre = detection_result[det_idx_pre, 0]
                    det_y_pre = detection_result[det_idx_pre, 1]
                    xy_dist_pre = math.sqrt((det_x_pre - gt_x) ** 2 + (det_y_pre - gt_y) ** 2)

                    det_x_cur = detection_result[det_idx, 0]
                    det_y_cur = detection_result[det_idx, 1]
                    xy_dist_cur = math.sqrt((det_x_cur - gt_x) ** 2 + (det_y_cur - gt_y) ** 2)
                    if xy_dist_pre <= xy_dist_cur:
                        # mark current detection as False Positive
                        detection_result[det_idx, 2] = 0
                        detection_result[det_idx, 3] = 0
                        pass
                    else:
                        # mark current detection as True Positive
                        detection_result[det_idx, 2] = 1
                        detection_result[det_idx, 3] = gt_idx
                        gt_labels[gt_idx, 2] = 2
                        gt_labels[gt_idx, 3] = det_idx
                        # mark previous detection as False Positive
                        detection_result[det_idx_pre, 2] = 0
                        detection_result[det_idx_pre, 3] = 0
                        pass
                    pass
                else:
                    # mark current detection as True Positive
                    # mark the corresponding gt_label as detected
                    gt_labels[gt_idx, 2] = 2
                    gt_labels[gt_idx, 3] = det_idx
                    detection_result[det_idx, 2] = 1
                    detection_result[det_idx, 3] = gt_idx
                pass
            pass
        pass
    tp_list = np.argwhere(detection_result[:, 2] > 0)

    if len(tp_list) == 0:
        precision = 0
        recall = 0
    else:
        precision = float(len(tp_list)) / detection_result.shape[0]
        recall = float(len(tp_list)) / gt_labels.shape[0]

    return precision, recall, detection_result, gt_labels, len(tp_list), detection_result.shape[0], gt_labels.shape[0]
    pass

def main(root_path, check_num, lap):
    paths = make_paths(root_path, check_num, lap)
    ## path list ##
    # 0: gtp
    # 1: detp
    # -3: savepath(detp)
    # -2: savepath(gtp)
    # -1: savepath
    files = load_files(paths)
    ## file list ##
    # 0: gtp
    # 1: gtp_all
    os.makedirs(paths[-3], exist_ok=True)
    os.makedirs(paths[-2], exist_ok=True)
    os.makedirs(paths[-1], exist_ok=True)

    # gtp = np.loadtxt(paths[0], delimiter=",", comments="%", dtype="int32")
    detp = np.loadtxt(paths[1], dtype="int32")

    evaluations = np.empty((0, 3))
    # detection_result = np.empty((0, 5))
    # grandtruth_result = np.empty((0, 5))
    allTP, allTPFP, allTPFN = 0, 0, 0
    for frame, fn in enumerate(files[0]):
        gt = np.loadtxt(str(fn), comments="%", dtype="int32")
        detection = detp[detp[:, 0] == frame][:, 1:]

        ## 0: FP, 1: TP, 2: FN
        ## evaluate to sampled grandtruth
        precision, recall, det_result, gt_lab, TP, TPFP, TPFN = evaluate_detection(detection, gt)
        F_measure = (2 * recall * precision) / (recall + precision + 1e-10)
        # print(f"frame:{frame:04}/prec={precision:.3f}, recall={recall:.3f}, F={F_measure:.3f}")

        allTP += TP
        allTPFP += TPFP
        allTPFN += TPFN
        
        evals = np.array([precision, recall, F_measure])
        evaluations = np.concatenate([evaluations, evals[np.newaxis, :]])

        # det_result = np.insert(det_result, 0, frame, axis=1)
        # detection_result = np.concatenate([detection_result, det_result])

        # gt_lab = np.insert(gt_lab, 0, frame, axis=1)
        # grandtruth_result = np.concatenate([grandtruth_result, gt_lab])

        savepathvector = os.path.join(paths[-3], f"detection_result-f{frame:04}.txt")
        np.savetxt(savepathvector, det_result, fmt="%d")
        savepathvector = os.path.join(paths[-2], f"gt_labels-f{frame:04}.txt")
        np.savetxt(savepathvector, gt_lab, fmt="%d")
        pass

    allprecision = float(allTP) / allTPFP
    allrecall = float(allTP) / allTPFN
    allF_measure = (2 * allprecision * allrecall) / (allprecision + allrecall + 1e-10)
    allevals = np.array([allprecision, allrecall, allF_measure])
    evaluations = np.concatenate([evaluations, allevals[np.newaxis, :]])

    savepathvector = os.path.join(paths[-1], f"evaluation_result.txt")
    np.savetxt(savepathvector, evaluations, header="precision recall F-measure  *last is result of evaluation to all frame*", footer=f"↑last is result of evaluation to all frame")


if __name__ == "__main__":
    main(root_path, check_num, lap)
    print("finished")