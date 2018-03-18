import os
from datetime import datetime
import time
import tensorflow as tf
from meta import Meta
from donkey import Donkey
from SVHNmodel import Model
from verify import Evaluator
import h5py
import numpy as np
tf.app.flags.DEFINE_string('data_dir', './data', 'Directory to read TFRecords files')
tf.app.flags.DEFINE_string('train_logdir', './logs/train', 'Directory to write training logs')
tf.app.flags.DEFINE_string('restore_checkpoint', None,
                           'Path to restore checkpoint (without postfix), e.g. ./logs/train/model.ckpt-100')
tf.app.flags.DEFINE_integer('batch_size', 32, 'Default 32')
tf.app.flags.DEFINE_float('learning_rate', 1e-2, 'Default 1e-2')
tf.app.flags.DEFINE_integer('patience', 100, 'Default 100, set -1 to train infinitely')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'Default 10000')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'Default 0.9')
FLAGS = tf.app.flags.FLAGS

def read_h5py_file(path_to_h5py_file,Flag):
    file=h5py.File(path_to_h5py_file,'r')
    if Flag==0:
        image=file['test_set'][:]
        length=file['test_length'][:]
        labels=file['test_labels'][:]
    if Flag==1:
        image=file['train_set'][:]
        length=file['train_length'][:]
        labels=file['train_labels'][:]
    if Flag==2:
        image=file['val_set'][:]
        length=file['val_length'][:]
        labels=file['val_labels'][:]
    return image,length,labels
def train(path_to_train_h5py_file, path_to_val_h5py_file,
           path_to_train_log_dir, path_to_restore_checkpoint_file, training_options):
    image_train,length_train,digits_train=read_h5py_file(path_to_train_h5py_file,Flag=1)
    image_val,length_val,digits_val=read_h5py_file(path_to_val_h5py_file,Flag=2)
    batch_size = training_options['batch_size']
    initial_patience = training_options['patience']
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000
    image_size=54
    num_channels=3
    digits_nums=5
    with tf.Graph().as_default():
        image_batch=tf.placeholder(
        tf.float32, shape=[None, image_size, image_size, num_channels])
        length_batch=tf.placeholder(tf.int32,shape=[None])
        digits_batch=tf.placeholder(tf.int32,shape=[None,digits_nums])
        # length_logtis, digits_logits = Model.inference(image_batch, drop_rate=0.2)
        length_logtis, digits_logits = Model.forward(image_batch, 0.8)

        loss = Model.loss(length_logtis, digits_logits, length_batch, digits_batch)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(training_options['learning_rate'], global_step=global_step,
                                                   decay_steps=training_options['decay_steps'], decay_rate=training_options['decay_rate'], staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.image('image', image_batch)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(path_to_train_log_dir, sess.graph)
            evaluator = Evaluator(os.path.join(path_to_train_log_dir, 'eval/val'))

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if path_to_restore_checkpoint_file is not None:
                assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file), \
                    '%s not found' % path_to_restore_checkpoint_file
                saver.restore(sess, path_to_restore_checkpoint_file)
                print('Model restored from file: %s' % path_to_restore_checkpoint_file)

            print('Start training')
            patience = initial_patience
            best_accuracy = 0.0
            duration = 0.0

            while True:
                start_time = time.time()
                image_batch_input, length_batch_input, digits_batch_input= Donkey.build_batch(image_train,length_train,digits_train,
                                                                     batch_size=batch_size)
                feed_dict={image_batch:image_batch_input,length_batch:length_batch_input,digits_batch:digits_batch_input}
                _, loss_val, summary_val, global_step_val, learning_rate_val = sess.run([train_op, loss, summary, global_step, learning_rate],feed_dict=feed_dict)
                duration += time.time() - start_time

                if global_step_val % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
                    duration = 0.0
                    print ('=> %s: step %d, loss = %f (%.1f examples/sec)' % (
                        datetime.now(), global_step_val, loss_val, examples_per_sec))

                if global_step_val % num_steps_to_check != 0:
                    continue

                summary_writer.add_summary(summary_val, global_step=global_step_val)

                print ('=> Evaluating on validation dataset...')
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'latest.ckpt'))
                accuracy = evaluator.evaluate(path_to_latest_checkpoint_file, image_val,length_val,digits_val,
                                              global_step_val)
                print ('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))

                if accuracy > best_accuracy:
                    path_to_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'model.ckpt'),
                                                         global_step=global_step_val)
                    print ('=> Model saved to file: %s' % path_to_checkpoint_file)
                    patience = initial_patience
                    best_accuracy = accuracy
                else:
                    patience -= 1

                print ('=> patience = %d' % patience)
                if patience == 0:
                    break
            print ('Finished')


def main(_):
    path_to_train_h5py_file = os.path.join(FLAGS.data_dir, 'train_set.h5')
    path_to_val_h5py_file = os.path.join(FLAGS.data_dir, 'val_set.h5')
    path_to_train_log_dir = FLAGS.train_logdir
    path_to_restore_checkpoint_file = FLAGS.restore_checkpoint
    training_options = {
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'patience': FLAGS.patience,
        'decay_steps': FLAGS.decay_steps,
        'decay_rate': FLAGS.decay_rate
    }
    train(path_to_train_h5py_file, path_to_val_h5py_file, 
           path_to_train_log_dir, path_to_restore_checkpoint_file,
           training_options)


if __name__ == '__main__':
    tf.app.run(main=main)
