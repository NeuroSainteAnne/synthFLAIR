import tensorflow as tf



bce_loss_object = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2, reduction=tf.keras.losses.Reduction.NONE)
mae_loss_object = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
    
def generator_loss(disc_generated_output, pseudo_flair, real_flair, lambda_l1,
                   pseudo_sobel, real_sobel, lambda_sobel, weights, pooled_weights):
    gan_loss = tf.reduce_mean(
        bce_loss_object(
            tf.ones_like(disc_generated_output), 
            disc_generated_output, 
            sample_weight=pooled_weights), 
        axis=(1,2))
    l1_loss = mae_loss_object(tf.reshape(real_flair*weights, (real_flair.shape[0],-1)),
                              tf.reshape(pseudo_flair*weights, (pseudo_flair.shape[0],-1)))
    edge_loss = mae_loss_object(tf.reshape(real_sobel*weights, (real_sobel.shape[0],-1)),
                              tf.reshape(pseudo_sobel*weights, (pseudo_sobel.shape[0],-1)))
    total_gen_loss = gan_loss + (lambda_l1 * l1_loss) + (lambda_sobel * edge_loss)
    return total_gen_loss, gan_loss, l1_loss, edge_loss

def discriminator_loss(disc_real_output, disc_generated_output, weights, pooled_weights):
    real_loss = tf.reduce_mean(
        bce_loss_object(
            tf.ones_like(disc_real_output), 
            disc_real_output, 
            sample_weight=pooled_weights), 
        axis=(1,2))
    generated_loss = tf.reduce_mean(
        bce_loss_object(
            tf.zeros_like(disc_generated_output),
            disc_generated_output,
            sample_weight=pooled_weights),
        axis=(1,2))
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss