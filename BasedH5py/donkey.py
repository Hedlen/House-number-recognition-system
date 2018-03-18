import tensorflow as tf
import numpy as np
import os
import random
class Donkey(object):
    @staticmethod
    def shuffled_batch(image,lengths,digits,bach_size,num_examples):
        image_list=[]
        lengths_list=[]
        digits_list=[]
        for _ in range(bach_size):
            index=random.randrange(num_examples)
            image_list.append(image[index])
            lengths_list.append(lengths[index])
            digits_list.append(digits[index])
        image_batch=np.array(image_list)
        lengths_batch=np.array(lengths_list)
        digits_batch=np.array(digits_list)
        return image_batch,lengths_batch,digits_batch
    @staticmethod
    def build_batch(image,length,digits,batch_size):
        num_examples=image.shape[0]
        #print("image numbers is %d" % num_examples)
        image_batch,length_batch,digits_batch = Donkey.shuffled_batch(image,length,digits,batch_size,num_examples)
        return image_batch, length_batch, digits_batch
