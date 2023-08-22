import tensorflow
import numpy as np
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def data_adaptive_loss(y_true, y_pred):
    data_adaptive_loss = 0.0
    num_el = K.epsilon()
    y_true = K.cast(y_true, dtype='float32')
    for slc in range(np.shape(y_true)[0]):
        if (tensorflow.greater(tensorflow.reduce_sum(y_true[slc, 0, 0, :]), 0.0)):
            idx = tensorflow.random.shuffle(tensorflow.where(tensorflow.greater(y_true[slc, 0, 0, :], 0.0)))[0][0]
            data_adaptive_loss += data_adaptive_class_loss(y_true[slc:slc+1, :, :, idx], y_pred[slc:slc+1, :, :, idx])
            num_el += 1
    return data_adaptive_loss / num_el

def data_adaptive_dice_metric(y_true, y_pred):
    y_true = K.cast(y_true, dtype='float32')
    for slc in range(np.shape(y_true)[0]):
        if (tensorflow.greater(tensorflow.reduce_sum(y_true[slc, 0, 0, 0]), 0.0)):
            data_adaptive_loss = data_adaptive_class_loss(y_true[slc:slc+1, :, :, 0], y_pred[slc:slc+1, :, :, 0], 1)
        else:
            return np.nan
    return data_adaptive_loss

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

def get_loss(loss_name):
    if (loss_name == "surface_loss_with_mauer"):
        return surface_loss_with_mauer
    if (loss_name == "dice_loss"):
        return dice_loss
    if (loss_name == "data_adaptive_loss"):
        return data_adaptive_loss
    if (loss_name == "mime_loss"):
        return mime_loss
    
    return tensorflow.keras.losses.get(loss_name)

def get_metric(metric_name):
    if (metric_name == "dice_loss"):
        return dice_loss
    if (metric_name == "data_adaptive_dice_metric"):
        return data_adaptive_dice_metric
    
    return tensorflow.keras.metrics.get(metric_name)

def surface_loss_with_mauer(y_true, y_pred):
    mask = y_true[:,0,:,:,:]
    map = y_true[:,1,:,:,:]
    map = map / tensorflow.reduce_sum(map)
    merr = tensorflow.abs(mask - y_pred)
    surface_loss = tensorflow.reduce_sum(merr * map)
    return surface_loss

def generalized_dice_coef_with_mauer(y_true, y_pred):
    smooth = 1e-8
    y_true_mask = y_true[:,0,:,:,:]
    w = 1 / (tensorflow.einsum('bhwc->bc', y_true_mask) + 1e-10)**2
    intersection = w * tensorflow.einsum('bwhc, bwhc->bc', y_true_mask, y_pred)
    areas = w * ( tensorflow.einsum('bwhc->bc', y_true_mask) + tensorflow.einsum('bwhc->bc', y_pred) )
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