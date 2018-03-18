import os
import h5py
import numpy as np
import glob
import random
#import tensorflow as tf
from PIL import Image
import  matplotlib.pyplot as plt
#DATA_DIR='E:\\Git_Project\\data'
DATA_DIR='./data'
numbers_count=0
def draw_picture(image_ori):
    #image = Image.open(image_ori)
    plt.figure()
    currentAxis = plt.gca()
    currentAxis.imshow(image_ori)
    #currentAxis.add_patch(Rectangle((cropped_left, cropped_top), cropped_width, cropped_height, fill=False, edgecolor='red'))
    #currentAxis.add_patch(Rectangle((bbox_left, bbox_top), bbox_width, bbox_height, fill=False, edgecolor='green'))
    #for attr_left, attr_top, attr_width, attr_height in zip(attrs_left, attrs_top, attrs_width, attrs_height):
    #currentAxis.add_patch(Rectangle((attr_left, attr_top), attr_width, attr_height, fill=False, edgecolor='white', linestyle='dotted'))
    plt.show()
def step_process(image, bbox_left, bbox_top, bbox_width, bbox_height):
    cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                    int(round(bbox_top - 0.15 * bbox_height)),
                                                                    int(round(bbox_width * 1.3)),
                                                                    int(round(bbox_height * 1.3)))
    image = image.crop([cropped_left, cropped_top, cropped_left + cropped_width, cropped_top + cropped_height])
    image = image.resize([64, 64])

    left_x=np.random.randint(0,image.size[0]-54-1)  
    left_y= np.random.randint(0,image.size[1]-54-1)  
    slide_image_copy1=image.crop([left_x,left_y,left_x+54,left_y+54]) 
    right_x=np.random.randint(0,image.size[0]-54-1)  
    right_y=np.random.randint(0,image.size[1]-54-1)  
    slide_image_copy2=image.crop([10-right_x,10-right_y,64-right_x,64-right_y])
    print ('Slide_image Size is %d, %d' % (slide_image_copy1.size[0],slide_image_copy2.size[0]))
    #draw_picture( slide_image_copy1)
    #draw_picture( slide_image_copy2)
    return slide_image_copy1,slide_image_copy2

def read_and_convert(digit_struct_mat_file,path_to_image_files):
    numbers_image=len(path_to_image_files)
    global numbers_count
    if numbers_image==numbers_count:
        return 0,0,0,0
    print(numbers_count)
    path_to_image_file = path_to_image_files[numbers_count]#visit every picture.
    index = int(path_to_image_file.split('\\')[-1].split('.')[0]) - 1 #!! '\\' windows env and '/' Linux env
    numbers_count += 1
    # read the .mat data

    #f = h5py.File(path_digitstruct_file_mat,'r') 
    attrs_dict = {} #Extract the contents in .mat file.
    f = digit_struct_mat_file #h5py object
    item = f['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = f[item][key]
        values = [f[attr.value[i].item()].value[0][0]
                for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs_dict[key] = values
    #attrs = read_mat_data(digit_struct_mat_file, index)
    
    attrs=attrs_dict
    label_of_digits = attrs['label']
    length = len(label_of_digits)
    if length > 5:#If length of label over 5
        # skip this example
        #print(numbers_count)
        return read_and_convert(digit_struct_mat_file,path_to_image_files)
    digits = [10, 10, 10, 10, 10]   # digit 10 represents no digit
    for idx_evpicture, label_of_digit in enumerate(label_of_digits):
        digits[idx_evpicture] = int(label_of_digit if label_of_digit != 10 else 0)    # label 10 is essentially digit zero

    attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x], [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
    min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                min(attrs_top),
                                                max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                max(map(lambda x, y: x + y, attrs_top, attrs_height)))
    center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                    (min_top + max_bottom) / 2.0,
                                    max(max_right - min_left, max_bottom - min_top))
    bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                    center_y - max_side / 2.0,
                                                    max_side,
                                                    max_side)
    slide_image1,slide_image2=step_process(Image.open(path_to_image_file), bbox_left, bbox_top, bbox_width, bbox_height)

    return slide_image1,slide_image2,length,digits
def convert_to_reformat(path_to_dataset_dir_and_digit_struct_mat_file_tuples,Flag):
    num_examples = []
    count_train=0
    count_val=0
    count_test=0
    other_image=[]
    images_examples={}
    characters=['length','image','labels']
    if Flag==True:
            for i in range(2):
                images_examples[i]={}
                for indej in characters:
                    images_examples[i][indej]=[]
    else:
            images_examples[0]={}
            for indej in characters:
                images_examples[0][indej]=[]
    for path_to_dataset_dir, path_to_digit_struct_mat_file in path_to_dataset_dir_and_digit_struct_mat_file_tuples:
        path_to_image_files=[]
        global numbers_count
        numbers_count=0
        print(path_to_dataset_dir)
        file_glob= os.path.join(path_to_dataset_dir, '*.png')
        #file_glob=path_to_dataset_dir+'/*.png'
        print(file_glob)
        path_to_image_files.extend(glob.glob(file_glob))
        #path_to_image_files = tf.gfile.Glob(os.path.join(path_to_dataset_dir, '*.png'))
        total_files = len(path_to_image_files)
        print ('%d files found in %s' % (total_files, path_to_dataset_dir))

        with h5py.File(path_to_digit_struct_mat_file, 'r') as digit_struct_mat_file:
            for index, path_to_image_file in enumerate(path_to_image_files):
                print ('(%d/%d) processing %s' % (index + 1, total_files, path_to_image_file))
                example1,example2,length,digits=read_and_convert(digit_struct_mat_file,path_to_image_files)
                if length==0:
                    break
                else:
                    if Flag==True:
                        if random.random() > 0.1:
                            id=0
                            #for j in range(2):
                            for i in characters: 
                                images_examples[id][i].append(0)
                            image_array1=np.asanyarray(example1,'float32')
                            print(count_train)
                            images_examples[id]['image'][count_train]=image_array1
                            images_examples[id]['length'][count_train]=length
                            images_examples[id]['labels'][count_train]=digits
                            count_train+=1
                        else:
                            id=1
                            for i in characters: 
                                images_examples[id][i].append(0)
                            image_array1=np.asanyarray(example1,'float32')
                            images_examples[id]['image'][count_val]=image_array1
                            images_examples[id]['length'][count_val]=length
                            images_examples[id]['labels'][count_val]=digits
                            count_val+=1
                    else:
                        id=0
                        for i in characters: 
                            images_examples[id][i].append(0)
                        image_array1=np.asanyarray(example1,'float32')
                        images_examples[id]['image'][count_test]=image_array1
                        images_examples[id]['length'][count_test]=length
                        images_examples[id]['labels'][count_test]=digits
    return images_examples

def main():
    path_to_train_dir =os.path.join(DATA_DIR, 'train')
    path_to_test_dir = os.path.join(DATA_DIR, 'test')
    path_to_extra_dir = os.path.join(DATA_DIR, 'extra')
    path_to_train_digit_struct_mat_file = os.path.join(path_to_train_dir, 'digitStruct.mat')
    path_to_test_digit_struct_mat_file = os.path.join(path_to_test_dir, 'digitStruct.mat')
    path_to_extra_digit_struct_mat_file = os.path.join(path_to_extra_dir, 'digitStruct.mat')

    path_to_h5_train_file = os.path.join(DATA_DIR, 'train_set.h5')
    path_to_h5_test_file = os.path.join(DATA_DIR, 'test_set.h5')
    path_to_h5_val_file = os.path.join(DATA_DIR, 'val_set.h5')
    train_flag=True
    test_flag=True
    val_flag=True
    if os.path.exists(path_to_h5_train_file):
        print('The file %s already exists' % path_to_h5_train_file)
        train_flag=False
    if os.path.exists(path_to_h5_test_file):
        print('The file %s already exists' % path_to_h5_test_file)
        test_flag=False
    if os.path.exists(path_to_h5_val_file):
        print('The file %s already exists' % path_to_h5_val_file)
        val_flag=False
    print ('Processing train and val data')
    if train_flag==True:
        train_val_set= convert_to_reformat([(path_to_train_dir, path_to_train_digit_struct_mat_file),
                                                                   (path_to_extra_dir, path_to_extra_digit_struct_mat_file)],True)
        file = h5py.File(path_to_h5_train_file,'w')
        file.create_dataset('train_set', data = np.array(train_val_set[0]['image']))
        file.create_dataset('train_labels', data = np.array(train_val_set[0]['labels']))
        file.create_dataset('train_length',data = np.array(train_val_set[0]['length'])) 
        file.close()
    print ('Processing test data')
    if test_flag==True:
        test_set= convert_to_reformat([(path_to_test_dir, path_to_test_digit_struct_mat_file)],False)
        file = h5py.File(path_to_h5_test_file,'w')
        file.create_dataset('test_set', data = np.array(test_set[0]['image'])) 
        file.create_dataset('test_labels', data = np.array(test_set[0]['labels']))
        file.create_dataset('test_length',data = np.array(test_set[0]['length'])) 
        file.close()
    print ('Processing val data')
    if val_flag==True:
        #val_set,val_labels,val_length= convert_to_reformat([(path_to_val_dir, path_to_val_digit_struct_mat_file)])
        file = h5py.File(path_to_h5_val_file,'w')
        file.create_dataset('val_set', data = np.array(train_val_set[1]['image'])) 
        file.create_dataset('val_labels', data = np.array(train_val_set[1]['labels']))
        file.create_dataset('val_length',data = np.array(train_val_set[1]['length'])) 
        file.close()

    print('Done!')
if __name__ == '__main__':
    main()                    







