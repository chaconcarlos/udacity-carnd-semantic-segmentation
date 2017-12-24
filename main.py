import helper
import os.path
import time
import warnings
import project_tests as tests
import tensorflow    as tf

from datetime          import timedelta
from distutils.version import LooseVersion

STD_DEV_INITIALIZER = 0.01
REGULARIZER_WEIGHT  = 1e-3
LEARNING_RATE       = 0.00001
KEEP_PROB           = 0.5
EPOCHS              = 50
BATCH_SIZE          = 5
DEFAULT_PADDING     = 'same'
EPOCH_LOG_FORMAT    = 'Epoch : {} / {} - loss: {} - Time: {}'

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag                    = 'vgg16'
    vgg_input_tensor_name      = 'image_input:0'
    vgg_keep_prob_tensor_name  = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    
    graph       = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob   = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out  = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out  = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out  = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out

tests.test_load_vgg(load_vgg, tf)

def get_convolution2D(previous_layer, num_classes):
    # returns a 2D convolution
    initializer = tf.random_normal_initializer(stddev = STD_DEV_INITIALIZER)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_WEIGHT)
    conv        = tf.layers.conv2d(
        previous_layer, 
        num_classes, 
        1, 
        padding = DEFAULT_PADDING, 
        kernel_initializer = initializer, 
        kernel_regularizer = regularizer)
    return conv

def get_convolution2D_transpose(previous_layer, num_classes, kernel_size, strides):
    # returns a 2D convolution
    initializer = tf.random_normal_initializer(stddev = STD_DEV_INITIALIZER)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_WEIGHT)
    conv        = tf.layers.conv2d_transpose(
        previous_layer, 
        num_classes, 
        kernel_size,
        strides, 
        padding = DEFAULT_PADDING, 
        kernel_initializer = initializer, 
        kernel_regularizer = regularizer)
    return conv

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    l7_conv1x1   = get_convolution2D(vgg_layer7_out, num_classes)
    l7_upsample1 = get_convolution2D_transpose(l7_conv1x1, num_classes, 4, strides = (2, 2))
    l4_conv1x1   = get_convolution2D(vgg_layer4_out, num_classes)
    l4_skip      = tf.add(l7_upsample1, l4_conv1x1)
    l4_upsample2 = get_convolution2D_transpose(l4_skip, num_classes, 4, strides = (2, 2))
    l3_conv1x1   = get_convolution2D(vgg_layer3_out, num_classes)
    l3_skip      = tf.add(l4_upsample2, l3_conv1x1)
    l3_upsample3 = get_convolution2D_transpose(l3_skip, num_classes, 16, strides = (8, 8))
    
    return l3_upsample3

tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits             = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label      = tf.reshape(correct_label, (-1,num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    optimizer          = tf.train.AdamOptimizer(learning_rate= learning_rate)
    train_op           = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())

    print('Training for {} epochs...'.format(epochs))
    
    training_time = time.time()
    
    for epoch in range(epochs):
        epoch_time = time.time()
        loss_fmt   = ""
        
        for image, label in get_batches_fn(batch_size):
            feed_dict = { input_image: image, correct_label: label, keep_prob: KEEP_PROB, learning_rate: LEARNING_RATE }
            _, loss   = sess.run([train_op, cross_entropy_loss], feed_dict)
            loss_fmt ='{:3f}'.format(loss)
        
        print(EPOCH_LOG_FORMAT.format(epoch + 1, epochs, loss_fmt, str(timedelta(seconds=(time.time() - epoch_time)))))

    print()
    print('Training finished. Training time: {}'.format(str(timedelta(seconds=(time.time() - training_time)))))
    
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir    = './data'
    runs_dir    = './runs'
    
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name = 'correct_label')
        learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
        
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        
        fcnn = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
          
        logits, train_op, cross_entropy_loss = optimize(fcnn, correct_label, learning_rate, num_classes)
        
        # TODO: Train NN using the train_nn function
        
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)
        
        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
