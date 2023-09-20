import tensorflow
import numpy as np
from keras import backend as K
from focal_loss import BinaryFocalLoss
from scipy import ndimage
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def get_loss(loss_name):
    if (loss_name == "surface_loss_with_mauer"):
        return surface_loss_with_mauer
    if (loss_name == "generalized_dice_loss"):
        return generalized_dice_loss
    if (loss_name == "dice_loss"):
        return dice_loss
    if (loss_name == "data_adaptive_loss"):
        return data_adaptive_loss
    if (loss_name == "erik_loss"):
        return erik_loss
    if (loss_name == "mime_loss"):
        return mime_loss
    if (loss_name == "dsfl"):
        return dsfl
    
    return tensorflow.keras.losses.get(loss_name)

def get_metric(metric_name):
    if (metric_name == "dice_loss"):
        return dice_loss
    if (metric_name == "data_adaptive_dice_metric"):
        return data_adaptive_dice_metric
    if (metric_name == "mean_error"):
        return mean_error
    if (metric_name == "hamming"):
        return hamming
    
    return tensorflow.keras.metrics.get(metric_name)

def hamming(y_true, y_pred):
    return - tensorflow.math.reduce_sum(y_true * y_pred)

def dsfl(y_true, y_pred, beta1=0.4, beta2=0.2, beta3=0.4, gamma=2):
    focal_loss = BinaryFocalLoss(gamma)
    dice_squared_l = dice_loss(y_true, y_pred)
    surface_l = 1 + (surface_loss(y_true, y_pred) / 12)
    focal_l_2 = focal_loss(y_true, y_pred)
    return beta1 * dice_squared_l + beta2 * surface_l + beta3 * focal_l_2


def surface_loss(y_true, y_pred):
    y_true_dist_map = tensorflow.py_function(func=calc_dist_map_batch, inp=[y_true], Tout=tensorflow.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)

def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    dist_maps = np.array([calc_dist_map(y) for y in y_true_numpy])
    return dist_maps.reshape(y_true.shape).astype(np.float32)

def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)
    if posmask.any():
        negmask = ~posmask
        neg_dist = ndimage.distance_transform_edt(negmask) * negmask
        pos_dist = (ndimage.distance_transform_edt(posmask) - 1) * posmask
        res = neg_dist - pos_dist
    return res



def data_adaptive_loss(y_true, y_pred):
    data_adaptive_loss = 0.0
    num_el = K.epsilon()
    y_true = K.cast(y_true, dtype='float32')
    for slc in range(np.shape(y_true)[0]):
        if (tensorflow.greater(y_true[slc, 0, 0, 0], 0.0)):
            data_adaptive_loss += data_adaptive_class_loss(y_true[slc:slc+1, :, :, 0], y_pred[slc:slc+1, :, :, 0], 1)
            num_el += 1
    return data_adaptive_loss / num_el

def mean_error(y_true, y_pred):
    return tensorflow.reduce_mean(tensorflow.subtract(y_true, y_pred))

def data_adaptive_dice_metric(y_true, y_pred):
    data_adaptive_l = []
    num_el = K.epsilon()
    y_true = K.cast(y_true, dtype='float32')
    for slc in range(np.shape(y_true)[0]):
        if (tensorflow.greater(y_true[slc, 0, 0, 0], 0.0)):
            data_adaptive_l.extend([data_adaptive_class_loss(y_true[slc:slc+1, :, :, 0], y_pred[slc:slc+1, :, :, 0], 1)])
            num_el += 1
        else:
            data_adaptive_l.extend([np.nan])
    return data_adaptive_l

def data_adaptive_class_loss(y_true, y_pred, delta=0.5):
    y_true_f = K.flatten(y_true)[1:]
    y_pred_f = K.flatten(y_pred)[1:]
    l = (delta*(1.-data_adaptive_dice_part(y_true_f, y_pred_f))) + ((1-delta)*data_adaptive_binary_crossentropy_part(y_true_f, y_pred_f))
    return l

def data_adaptive_dice_part(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    values = (2. * intersection + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())
    return values

def data_adaptive_binary_crossentropy_part(y_true, y_pred):
    cross = K.binary_crossentropy(y_true, y_pred)
    m = K.mean(cross)
    return m

def erik_loss(skip_value):
    import tensorflow as tf
    def loss_fn(y_true, y_pred):
        loss = 0.0
        for i in range(y_true.shape[0]):
            alpha = 0.001 if (y_true[i, 0, 0, 0] == skip_value) else 1.0
            loss += alpha * tf.losses.mean_absolute_error(y_true[i, ], y_pred[i, ])
        return loss / y_true.shape[0]
    return loss_fn

def mime_loss(y_true, y_pred):
    import tensorflow as tf
    loss = 0.0
    mask_a = tf.not_equal(y_true, False)
    mask_b = tf.equal(y_true, False)
    loss_a = - y_pred[mask_a]
    loss_b = y_pred[mask_b]
    if (~tf.math.is_nan(tf.reduce_mean(loss_a))):
        loss += tf.reduce_mean(loss_a)
    # if (~tf.math.is_nan(tf.reduce_mean(loss_b))):
    #     loss += tf.reduce_mean(loss_b)
    return loss

def surface_loss_with_mauer(y_true, y_pred):
    mask = y_true[:,0,:,:,:]
    map = y_true[:,1,:,:,:]
    map = map / tensorflow.reduce_sum(map)
    merr = tensorflow.abs(mask - y_pred)
    surface_loss = tensorflow.reduce_sum(merr * map)
    return surface_loss

def generalized_dice_coef_with_mauer(y_true, y_pred):
    y_true = tensorflow.cast(y_true, dtype='float32')
    smooth = 1e-8
    w = 1 / (tensorflow.einsum('bhwc->bc', y_true) + 1e-10)**2
    intersection = w * tensorflow.einsum('bwhc, bwhc->bc', y_true, y_pred)
    areas = w * ( tensorflow.einsum('bwhc->bc', y_true) + tensorflow.einsum('bwhc->bc', y_pred) )
    g_dice_coef = (2 * intersection + smooth) / (areas + smooth)
    return g_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coef_with_mauer(y_true, y_pred)

def combined_surface_gDice_loss_with_mauer(y_true, y_pred):
    w_factor = 174.5
    alpha = 0.5
    surface_loss = surface_loss_with_mauer(y_true, y_pred)
    gDice = generalized_dice_loss(y_true, y_pred)
    loss = alpha * gDice + (1 - alpha) * w_factor * surface_loss
    return loss

def dice_coef(y_true, y_pred):
    smooth = 1e-8
    y_true_f = K.flatten(K.cast(y_true, dtype='float32'))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_with_mauer(y_true, y_pred):
    smooth = 1e-8
    y_true_mask = y_true[:,0,:,:,:]
    y_true_f = K.flatten(y_true_mask)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss_with_mauer(y_true, y_pred):
    return -dice_coef_with_mauer(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)