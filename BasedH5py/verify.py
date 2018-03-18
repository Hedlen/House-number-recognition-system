import tensorflow as tf
from donkey import Donkey
from SVHNmodel import Model
import math

class Evaluator(object):
    def __init__(self, path_to_eval_log_dir):
        self.summary_writer = tf.summary.FileWriter(path_to_eval_log_dir)
        self.image_size=54
        self.num_channels=3
        self.digits_nums=5
    def evaluate(self, path_to_checkpoint, image_eval,length_eval,digits_eval, global_step):
        batch_size=128
        needs_include_length = False
        accuracy_val=0.0
        with tf.Graph().as_default():
            image_batch=tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, self.num_channels])
            length_batch=tf.placeholder(tf.int32,shape=[None])
            digits_batch=tf.placeholder(tf.int32,shape=[None,self.digits_nums])
            num_examples=image_eval.shape[0]
            num_batches = num_examples / batch_size
            # length_logits, digits_logits = Model.inference(image_batch, drop_rate=0.0)
            length_logits, digits_logits = Model.forward(image_batch, 1.0)
            length_predictions = tf.argmax(length_logits, axis=1)
            digits_predictions = tf.argmax(digits_logits, axis=2)
            
            if needs_include_length:
                labels = tf.concat([tf.reshape(length_batch, [-1, 1]), digits_batch], axis=1)
                predictions = tf.concat([tf.reshape(length_predictions, [-1, 1]), digits_predictions], axis=1)
            else:
                labels = digits_batch
                predictions = digits_predictions
            #correct_pre = tf.equal(tf.argmax(labels,axis=1),tf.argmax(predictions,axis=1))
            #accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))
            labels_string = tf.reduce_join(tf.as_string(labels), axis=1)
            predictions_string = tf.reduce_join(tf.as_string(predictions), axis=1)
            correct_pre = tf.equal(labels_string,predictions_string)
            accuracy = tf.reduce_mean(tf.cast(correct_pre, tf.float32))

            tf.summary.image('image', image_batch)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram('variables',
                                tf.concat([tf.reshape(var, [-1]) for var in tf.trainable_variables()], axis=0))
            summary = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
               
                restorer = tf.train.Saver()
                restorer.restore(sess, path_to_checkpoint)
                for _ in range(math.floor(num_examples / batch_size)):
                    image_batch_input,length_batch_input, digits_batch_input= Donkey.build_batch(image_eval,length_eval,digits_eval,
                                                                         batch_size=batch_size)
                    feed_dict={image_batch:image_batch_input,length_batch:length_batch_input,digits_batch:digits_batch_input}
                    accuracy_step,summary_val= sess.run([accuracy,summary],feed_dict=feed_dict)
                    accuracy_val+=accuracy_step
                self.summary_writer.add_summary(summary_val, global_step=global_step)
                accuracy_val=accuracy_val/math.floor(num_examples / batch_size)
        return accuracy_val*100
