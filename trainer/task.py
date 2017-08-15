import keras
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import get_session
from keras.models import Model
from keras.preprocessing import image
from keras.layers import Input, Lambda
import os
import tarfile
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils
import tensorflow.python.debug as tf_debug

from ssd import SSD300
from ssd_training import MultiboxLoss
from ssd_utils import BBoxUtility
from generator import Generator

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epoch', 100, 'Number of epoch to runs.')
flags.DEFINE_integer('num_classes', 21, 'Number of classes (including background class).')
flags.DEFINE_string('annotation_path', 'data/annotation.pkl', 'Path to annotation data.')
flags.DEFINE_string('prior_path', 'data/prior.pkl', 'Path to prior data.')
flags.DEFINE_string('weight_path', 'data/weight.hdf5', 'Path to weight data.')
flags.DEFINE_string('images_path', 'data/images.tar.gz', 'Path to image data.')
flags.DEFINE_string('model_dir', 'data', 'Directory to put the model into.')

def build_signature(inputs, outputs):
    signature_inputs = {key: saved_model_utils.build_tensor_info(tensor)
                        for key, tensor in inputs.items()}
    signature_outputs = {key: saved_model_utils.build_tensor_info(tensor)
                         for key, tensor in outputs.items()}

    signature_def = signature_def_utils.build_signature_def(
        signature_inputs, signature_outputs,
        signature_constants.PREDICT_METHOD_NAME)

    return signature_def

def export(sess, inputs, outputs, output_dir):
    if file_io.file_exists(output_dir):
        file_io.delete_recursively(output_dir)

    signature_def = build_signature(inputs, outputs)

    signature_def_map = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
    }
    builder = saved_model_builder.SavedModelBuilder(output_dir)
    builder.add_meta_graph_and_variables(
        sess,
        tags=[tag_constants.SERVING],
        signature_def_map=signature_def_map)

    builder.save()

def run_training():
    file_io.create_dir(FLAGS.model_dir)

    np.set_printoptions(suppress=True)

    input_shape = (300, 300, 3)

    prior_filename = os.path.basename(FLAGS.prior_path)
    file_io.copy(FLAGS.prior_path, prior_filename)
    priors = pickle.load(open(prior_filename, 'rb'))
    bbox_util = BBoxUtility(FLAGS.num_classes, priors)

    annotation_filename = os.path.basename(FLAGS.annotation_path)
    file_io.copy(FLAGS.annotation_path, annotation_filename)
    gt = pickle.load(open(annotation_filename, 'rb'))
    keys = sorted(gt.keys())
    num_train = int(round(0.8 * len(keys)))
    train_keys = keys[:num_train]
    val_keys = keys[num_train:]
    num_val = len(val_keys)

    images_filename = os.path.basename(FLAGS.images_path)
    file_io.copy(FLAGS.images_path, images_filename)
    tar = tarfile.open(images_filename)
    tar.extractall()
    tar.close()

    path_prefix = images_filename.split('.')[0] + '/'
    gen = Generator(gt, bbox_util, 4, path_prefix,
                    train_keys, val_keys,
                    (input_shape[0], input_shape[1]), do_crop=False)

    net, model = SSD300(input_shape, num_classes=FLAGS.num_classes)

    weight_filename = os.path.basename(FLAGS.weight_path)
    file_io.copy(FLAGS.weight_path, weight_filename)
    model.load_weights(weight_filename, by_name=True)

    freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
              'conv2_1', 'conv2_2', 'pool2',
              'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
              'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

    for L in model.layers:
        if L.name in freeze:
            L.trainable = False

    base_lr = 3e-4
    optim = keras.optimizers.Adam(lr=base_lr)
    model.compile(optimizer=optim,
                  loss=MultiboxLoss(FLAGS.num_classes, neg_pos_ratio=2.0).compute_loss)

    # train
    model.fit_generator(gen.generate(True), gen.train_batches,
            FLAGS.epoch,
            validation_data=gen.generate(False),
            nb_val_samples=gen.val_batches,
            nb_worker=1)

    # define prediction layer
    keys_placeholder = tf.placeholder(tf.string, shape=[None])
    keep_top_k_placeholder = tf.placeholder(dtype='int32', shape=(None))
    original_size_placeholder = tf.placeholder(dtype='float32', shape=(None, 2))
    confidence_threshold_placeholder = tf.placeholder(dtype='float32', shape=(None))

    detection_out = Lambda(bbox_util.detection_out, arguments={
        'keep_top_k': keep_top_k_placeholder,
        'confidence_threshold': confidence_threshold_placeholder,
        'original_size': original_size_placeholder
        })(net['predictions'])

    # export
    inputs  = {'key': keys_placeholder,
               'data': model.input,
               'keep_top_k': keep_top_k_placeholder,
               'confidence_threshold': confidence_threshold_placeholder,
               'original_size': original_size_placeholder}
    outputs = {'key': tf.identity(keys_placeholder), 'objects': detection_out}
    export(get_session(), inputs, outputs, FLAGS.model_dir)

def main(_):
    run_training()

if __name__ == '__main__':
    tf.app.run()
