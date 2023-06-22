import tensorflow
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def get_loss(loss_name):
    if (loss_name == "surface_loss_with_mauer"):
        return surface_loss_with_mauer
    
    return tensorflow.keras.losses.get(loss_name)

def get_metric(metric_name):
    if (metric_name == "dice_loss"):
        return dice_loss
    
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

def dice_coef_with_mauer(y_true, y_pred):
    smooth = 1e-8
    y_true_mask = y_true[:,0,:,:,:]
    y_true_f = K.flatten(y_true_mask)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coef_with_mauer(y_true, y_pred)