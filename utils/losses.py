import tensorflow
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def data_adaptive_loss(y_true, y_pred):
    l, p = data_adaptive_class_loss(tensorflow.cast(y_true, dtype=tensorflow.float32), y_pred)
    return l/(tensorflow.cast((p), dtype=tensorflow.float32))

def data_adaptive_dice_metric(y_true, y_pred):
    l, p = data_adaptive_class_loss(tensorflow.cast(y_true[i], dtype=tensorflow.float32), y_pred[i], 0)
    return l/(tensorflow.cast((p), dtype=tensorflow.float32))

def data_adaptive_class_loss(y_true, y_pred, delta=0.5):
    # Batch size
    s = K.shape(y_true)[0]
    # Weight
    w = y_true[:,0,0,0]
    # Set weight to zero
    z = K.zeros_like(y_true[:,:,0:1,:])
    y_part = y_true[:,:,1:,:]
    y_true = K.concatenate([z, y_part], axis=2)
    # How many times the mask accures
    a = K.sum(w)
    # Chech if masks are present at all
    p = K.switch(K.equal(a,0), 0, 1)
    # Calculating loss
    y_true_f = K.reshape(y_true, (s, -1))
    y_pred_f = K.reshape(y_pred, (s, -1))
    l = (delta*(1.-data_adaptive_dice_part(y_true_f, y_pred_f))) + ((1-delta)*data_adaptive_binary_crossentropy_part(y_true_f, y_pred_f))
    # Set loss to zero if mask does not excist
    l = w*l
    # Sum and div by number of present masks
    l = K.sum(l)/(a + K.epsilon())
    return l, p

def data_adaptive_dice_part(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=1)
    values = (2. * intersection + K.epsilon()) / (K.sum(y_true, axis=1) + K.sum(y_pred,axis=1) + K.epsilon())
    return values

def data_adaptive_binary_crossentropy_part(y_true, y_pred):
    cross = K.binary_crossentropy(y_true, y_pred)
    m = K.mean(cross, axis=1)
    return m

def get_loss(loss_name):
    if (loss_name == "surface_loss_with_mauer"):
        return surface_loss_with_mauer
    if (loss_name == "data_adaptive_loss"):
        return data_adaptive_loss
    
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

def dice_coef_with_mauer(y_true, y_pred):
    smooth = 1e-8
    y_true_mask = y_true[:,0,:,:,:]
    y_true_f = K.flatten(y_true_mask)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return -dice_coef_with_mauer(y_true, y_pred)