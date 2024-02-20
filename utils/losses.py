import tensorflow as tf
import numpy as np
from keras import backend as K
from focal_loss import BinaryFocalLoss
from scipy import ndimage
import math
from sklearn.utils.extmath import cartesian

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
    if (loss_name == "gsl"):
        return gsl
    
    return tf.keras.losses.get(loss_name)

def get_metric(metric_name):
    if (metric_name == "dice_score"):
        return dice_score
    if (metric_name == "data_adaptive_dice_metric"):
        return data_adaptive_dice_metric
    if (metric_name == "mean_error"):
        return mean_error
    if (metric_name == "hamming"):
        return hamming
    if (metric_name == "accuracy"):
        return accuracy
    
    return tf.keras.metrics.get(metric_name)

def accuracy(y_true, y_pred):
    return - tf.keras.metrics.binary_accuracy(y_true, y_pred)

def hamming(y_true, y_pred):
    return - tf.math.reduce_sum(y_true * y_pred)

def dsfl(y_true, y_pred, beta1=0.4, beta2=0.2, beta3=0.4, gamma=2):
    focal_loss = BinaryFocalLoss(gamma)
    dice_squared_l = dice_loss(y_true, y_pred)
    surface_l = 1 + (surface_loss(y_true, y_pred) / 12)
    focal_l_2 = focal_loss(y_true, y_pred)
    return beta1 * dice_squared_l + beta2 * surface_l + beta3 * focal_l_2


def diceCEloss(y_true, y_pred):
    loss_ce = K.mean(K.binary_crossentropy(y_true, y_pred))
    loss_dice = dice_loss(y_true, y_pred)

    return loss_ce + loss_dice

class LinearSchedule:
    def __init__(self, num_epochs, init_pause):
        if num_epochs <= init_pause:
            raise ValueError("The number of epochs must be greater than the initial pause.")
        self.num_epochs = num_epochs - 1
        self.init_pause = init_pause

    def __call__(self, epoch):
        if epoch > self.num_epochs:
            raise ValueError("The current epoch is greater than the total number of epochs.")
        if epoch > self.init_pause:
            return min(1, max(0, 1.0 - (float(epoch - self.init_pause) / (self.num_epochs - self.init_pause))))
        else:
            return 1.0
        
def gsl(n_epochs, epoch):
    epoch = epoch
    n_epochs = n_epochs
    alpha = LinearSchedule(n_epochs, 5)
    # Compute region based loss
    smooth = 1e-6

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        loss_gsl = 0.0
        for idx in range(y_true.shape[0]):
            region_loss = diceCEloss(y_true[idx, ...], y_pred[idx, ...])

            # with tf.compat.v1.Session() as sess:
            #     y_true_np = sess.run(y_true[idx, ...])
            dtm = calc_dist_map(y_true.numpy()[idx, ...])

            class_weight = tf.reduce_sum(y_true[idx, ...])
            class_weight = 1. / (tf.square(class_weight) + 1.)

            y_worst = tf.square(1.0 - y_true[idx, ...])

            num = tf.reduce_sum(tf.square(dtm * (y_worst - y_pred[idx, ...])))
            num *= class_weight

            den = tf.reduce_sum(tf.square(dtm * (y_worst - y_true[idx, ...])))
            den *= class_weight
            den += smooth

            boundary_loss = tf.reduce_sum(num) / tf.reduce_sum(den)
            boundary_loss = tf.reduce_mean(boundary_loss)
            boundary_loss = 1. - boundary_loss
            
            loss_gsl += alpha(epoch) * region_loss + (1. - alpha(epoch)) * boundary_loss
        return loss_gsl
    return loss_fn

def surface_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch, inp=[y_true], Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)

def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    dist_maps = np.array([calc_dist_map(y) for y in y_true_numpy])
    return dist_maps.reshape(y_true.shape).astype(np.float32)

def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = np.array(seg > 0.5, dtype=np.bool_)
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
        if (tf.greater(y_true[slc, 0, 0, 0], 0.0)):
            data_adaptive_loss += data_adaptive_class_loss(y_true[slc:slc+1, :, :, 0], y_pred[slc:slc+1, :, :, 0], 1)
            num_el += 1
    return data_adaptive_loss / num_el

def mean_error(y_true, y_pred):
    return tf.reduce_mean(tf.subtract(y_true, y_pred))

def data_adaptive_dice_metric(y_true, y_pred):
    data_adaptive_l = []
    num_el = K.epsilon()
    y_true = K.cast(y_true, dtype='float32')
    for slc in range(np.shape(y_true)[0]):
        if (tf.greater(y_true[slc, 0, 0, 0], 0.0)):
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
    def loss_fn(y_true, y_pred):
        loss = 0.0
        for i in range(y_true.shape[0]):
            alpha = 0.001 if (y_true[i, 0, 0, 0] == skip_value) else 1.0
            loss += alpha * tf.losses.mean_absolute_error(y_true[i, ], y_pred[i, ])
        return loss / y_true.shape[0]
    return loss_fn

def mime_loss(y_true, y_pred):
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
    map = map / tf.reduce_sum(map)
    merr = tf.abs(mask - y_pred)
    surface_loss = tf.reduce_sum(merr * map)
    return surface_loss

def generalized_dice_coef_with_mauer(y_true, y_pred):
    y_true = tf.cast(y_true, dtype='float32')
    smooth = 1e-8
    w = 1 / (tf.einsum('bhwc->bc', y_true) + 1e-10)**2
    intersection = w * tf.einsum('bwhc, bwhc->bc', y_true, y_pred)
    areas = w * ( tf.einsum('bwhc->bc', y_true) + tf.einsum('bwhc->bc', y_pred) )
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

def dice_score(y_true, y_pred):
    return -100 * dice_coef(y_true, y_pred)

def tf_repeat(tensor, repeats):
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor
