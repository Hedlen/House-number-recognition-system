import os
import tensorflow as tf
from meta import Meta
from evaluator import Evaluator
from train import read_h5py_file
tf.app.flags.DEFINE_string('data_dir_test', './data', 'Directory to read TFRecords files')
tf.app.flags.DEFINE_string('checkpoint_dir', './logs/train', 'Directory to read checkpoint files')
tf.app.flags.DEFINE_string('eval_logdir', './logs/train/eval', 'Directory to write evaluation logs')
FLAGS = tf.app.flags.FLAGS


def _eval(path_to_checkpoint_dir, image_test, length_test,digits_test,path_to_test_eval_log_dir):
    evaluator = Evaluator(path_to_test_eval_log_dir)

    checkpoint_paths = tf.train.get_checkpoint_state(path_to_checkpoint_dir).all_model_checkpoint_paths
    for global_step, path_to_checkpoint in [(path.split('-')[-1], path) for path in checkpoint_paths]:
        print(global_step)
        try:
            global_step_val = int(global_step)
        except ValueError:
            continue
        accuracy = evaluator.evaluate(path_to_checkpoint, image_test, length_test,digits_test,global_step_val)
        print ('Evaluate %s on Test Set, accuracy = %f' % (path_to_checkpoint, accuracy))


def main(_):
    path_to_test_h5py_file = os.path.join(FLAGS.data_dir_test, 'test_set.h5')
    path_to_checkpoint_dir = FLAGS.checkpoint_dir
    path_to_test_eval_log_dir = os.path.join(FLAGS.eval_logdir, 'test')
    image_test,length_test,digits_test=read_h5py_file(path_to_test_h5py_file,Flag=0)
    _eval(path_to_checkpoint_dir, image_test,length_test,digits_test,path_to_test_eval_log_dir)


if __name__ == '__main__':
    tf.app.run(main=main)
