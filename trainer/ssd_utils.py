"""Some utils for SSD."""

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import get_session

class BBoxUtility(object):
    """Utility class to do some stuff with bounding boxes and priors.

    # Arguments
        num_classes: Number of classes including background.
        priors: Priors and variances, numpy tensor of shape (num_priors, 8),
            priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
        overlap_threshold: Threshold to assign box to a prior.
        nms_thresh: Nms threshold.
        top_k: Number of total bboxes to be kept per image after nms step.

    # References
        https://arxiv.org/abs/1512.02325
    """
    # TODO add setter methods for nms_thresh and top_K
    def __init__(self, num_classes, priors=None, overlap_threshold=0.5,
                 nms_thresh=0.45, top_k=400):
        self.num_classes = num_classes
        self.priors = priors
        self.num_priors = 0 if priors is None else len(priors)
        self.overlap_threshold = overlap_threshold
        self._nms_thresh = nms_thresh
        self._top_k = top_k

    def iou(self, box):
        """Compute intersection over union for the box with all priors.

        # Arguments
            box: Box, numpy tensor of shape (4,).

        # Return
            iou: Intersection over union,
                numpy tensor of shape (num_priors).
        """
        # compute intersection
        inter_upleft = np.maximum(self.priors[:, :2], box[:2])
        inter_botright = np.minimum(self.priors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # compute union
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (self.priors[:, 2] - self.priors[:, 0])
        area_gt *= (self.priors[:, 3] - self.priors[:, 1])
        union = area_pred + area_gt - inter
        # compute iou
        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        """Encode box for training, do it only for assigned priors.

        # Arguments
            box: Box, numpy tensor of shape (4,).
            return_iou: Whether to concat iou to encoded values.

        # Return
            encoded_box: Tensor with encoded box
                numpy tensor of shape (num_priors, 4 + int(return_iou)).
        """
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_priors, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        assigned_priors = self.priors[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] +
                                        assigned_priors[:, 2:4])
        assigned_priors_wh = (assigned_priors[:, 2:4] -
                              assigned_priors[:, :2])
        # we encode variance
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh /
                                                  assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        """Assign boxes to priors for training.

        # Arguments
            boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
                num_classes without background.

        # Return
            assignment: Tensor with assigned boxes,
                numpy tensor of shape (num_boxes, 4 + num_classes + 8),
                priors in ground truth are fictitious,
                assignment[:, -8] has 1 if prior should be penalized
                    or in other words is assigned to some ground truth box,
                assignment[:, -7:] are all 0. See loss for more details.
        """
        assignment = np.zeros((self.num_priors, 4 + self.num_classes + 8))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_priors, 5)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,
                                                         np.arange(assign_num),
                                                         :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        return assignment

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])
        decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decode_bbox_center_x += prior_center_x
        decode_bbox_center_y = mbox_loc[:, 1] * prior_height * variances[:, 1]
        decode_bbox_center_y += prior_center_y
        decode_bbox_width = tf.exp(mbox_loc[:, 2] * variances[:, 2])
        decode_bbox_width *= prior_width
        decode_bbox_height = tf.exp(mbox_loc[:, 3] * variances[:, 3])
        decode_bbox_height *= prior_height
        decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
        decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
        decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
        decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
        decode_bbox = tf.concat([tf.expand_dims(decode_bbox_xmin, -1),
                                 tf.expand_dims(decode_bbox_ymin, -1),
                                 tf.expand_dims(decode_bbox_xmax, -1),
                                 tf.expand_dims(decode_bbox_ymax, -1), ], -1)
        decode_bbox = tf.maximum(decode_bbox, tf.zeros(tf.shape(decode_bbox)))
        decode_bbox = tf.minimum(decode_bbox, tf.ones(tf.shape(decode_bbox)))
        return decode_bbox

    def detection_out(self, predictions, keep_top_k, confidence_threshold,
                      original_size, background_label_id=0):
        """Do non maximum suppression (nms) on prediction results.
           And fit in original size.

        # Arguments
            predictions: Numpy array of predicted values.
            num_classes: Number of classes for prediction.
            keep_top_k: Number of total bboxes to be kept per image
                after nms step. (using tf.placeholder)
            confidence_threshold: Only consider detections,
                whose confidences are larger than a threshold. (using tf.placeholder)
            original_size: Size of target image (width, height).
                (using tf.placeholder)
            background_label_id: Label of background class.

        # Return
            results: List of predictions for every picture. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax]
        """
        predictions = tf.squeeze(predictions, [0])
        mbox_loc = predictions[:, :4]
        variances = predictions[:, -4:]
        mbox_priorbox = predictions[:, -8:-4]
        mbox_conf = predictions[:, 4:-8]

        decode_bbox = self.decode_boxes(mbox_loc, mbox_priorbox, variances)

        threshold = confidence_threshold[0] * tf.ones(tf.shape(mbox_conf[:, 0]))
        results = []
        for c in range(self.num_classes):
            if c == background_label_id:
                continue
            c_confs = mbox_conf[:, c]
            c_confs_m = tf.reshape(tf.where(tf.greater(c_confs, threshold)), [-1])
            boxes_to_process = tf.gather(decode_bbox, c_confs_m)
            confs_to_process = tf.gather(c_confs, c_confs_m)
            idx = tf.image.non_max_suppression(boxes_to_process, confs_to_process,
                                                self._top_k,
                                                iou_threshold=self._nms_thresh)
            good_boxes = tf.gather(boxes_to_process, idx)
            confs = tf.expand_dims(tf.gather(confs_to_process, idx), -1)
            labels = c * tf.ones(tf.shape(tf.expand_dims(idx, -1)))
            c_pred = tf.concat([labels, confs, good_boxes], 1)
            results.append(c_pred)

        results = tf.concat(results, 0)
        cond = tf.greater(tf.shape(results)[0], keep_top_k[0])
        top_k = tf.cond(cond, lambda: keep_top_k[0], lambda: tf.shape(results)[0])
        results = tf.gather(results, tf.nn.top_k(results[:, 1], k=top_k).indices)

        # Fit in original size
        fit_matrix = tf.concat([tf.ones([2]), original_size[0], original_size[0]], 0, name='fit_matrix')
        results = results * fit_matrix

        # WORKAROUND: Outer dimension for outputs must be unknown
        top_one = tf.cond(cond, lambda: tf.constant(1), lambda: tf.constant(1))
        results = tf.expand_dims(results, 0)
        s, _ = tf.nn.top_k(results[:, :, 1], k=top_one)
        _, indices = tf.nn.top_k(s, k=top_one)
        return tf.gather(results, indices[0])
