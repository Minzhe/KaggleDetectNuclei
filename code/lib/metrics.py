##############################################################################
###                             metrics.py                                 ###
##############################################################################
# Create Keras metric
# mean average precision at different intersection over union (IoU) thresholds metric
# Define IoU metric

import numpy as np
from skimage.morphology import label
import tensorflow as tf
from keras import backend as K

################# use tensorflow api ##################
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


################# self defined iou ####################
def precision_at(iou, threshold):
    '''
    Calculate TP, FP, FN in a given threshold.
    :param iou: IoU of each object in the image
    :param threshold: threshold of iou for a object to be considered as a true positive
    :return: precision TP / (TP + FP + FN)
    '''
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def iou_metric(y_true, y_pred, print_table=False):
    '''
    Calculate the mean iou in different threshold
    @param y_pred: predicted vector
    @param y_true: truth vector
    @print_table: if print results in different threshold
    @return: mean iou of all objects in a images
    '''
    # label adjacent regions with integer
    labels = label(y_true > 0)
    y_pred = label(y_pred > 0)

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    # compute intersection
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # compute union
    area_true = np.histogram(labels.flatten(), bins=true_objects)[0]
    area_pred = np.histogram(y_pred.flatten(), bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection

    # exclude background from the analysis (label with integer 0)
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # compute the intersection over union
    iou = intersection / union

    # loop over iou thresholds
    prec = []
    if print_table:
        print("Threshold\tTP\tFP\tFN\tPrecision.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(iou=iou, threshold=t)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("Ave.Prec\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batches(y_true, y_pred):
    '''
    Mean iou of all batches
    @param y_pred: predicted vector
    @param y_true: truth vector
    @return: mean iou of all batches
    '''
    batch_size = y_true.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true[batch], y_pred[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)


def iou_metric_tf(label, pred):
    '''
    Create tf value
    '''
    iou_metric_value = tf.py_func(iou_metric_batches, [label, pred], tf.float32)
    return iou_metric_value

