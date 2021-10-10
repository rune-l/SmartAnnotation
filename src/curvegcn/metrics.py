
import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.l2_loss(preds - labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

if __name__ == '__main__':
    from gluoncv.data import ADE20KSegmentation
    data_dir = './data'
    train_dataset = ADE20KSegmentation(root=data_dir, split='train')
    val_dataset = ADE20KSegmentation(root=data_dir, split='val')
    print('Training images:', len(train_dataset))
    print('Validation images:', len(val_dataset))

