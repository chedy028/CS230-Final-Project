import numpy as np
import skimage.io
import skimage
from imgaug import augmenters as iaa

class data_generator_4channels:

    ''' to generate batch of image data with the corrsponding labels,

    stack 4 channels'''

    def create_train(dataset, batch_size, shape, augument=True):
        while True:
            random_indexes = np.random.choice(len(dataset), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, index in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset[index]['path'], shape)
                #if augument:
                    #image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset[index]['labels']] = 1
            yield batch_images, batch_labels



    def load_image(path, shape):
        image_red_channel = skimage.io.imread(path+'_red.png')
        image_yellow_channel =skimage.io.imread(path+'_yellow.png')
        image_green_channel =skimage.io.imread(path+'_green.png')
        image_blue_channel =skimage.io.imread(path+'_blue.png')
        image = np.stack((image_red_channel, image_green_channel, image_blue_channel, image_yellow_channel),-1)
        image = skimage.transform.resize(image, (shape[0], shape[1]), mode= 'reflect')
        return image

    
class data_generator_3channels:
     def create_train(dataset, batch_size, shape, augument=True):
        while True:
            random_indexes = np.random.choice(len(dataset), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, index in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset[index]['path'], shape)
                #if augument:
                    #image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset[index]['labels']] = 1
            yield batch_images, batch_labels



    def load_image(path, shape):
        image_red_ch = skimage.io.imread(path+'_red.png')
        image_yellow_ch = skimage.io.imread(path+'_yellow.png')
        image_green_ch = skimage.io.imread(path+'_green.png')
        image_blue_ch = skimage.io.imread(path+'_blue.png')

        image_red_ch += (image_yellow_ch/3).astype(np.uint8) 
        image_green_ch += (image_yellow_ch/3).astype(np.uint8)
        image_blue_ch += (image_yellow_ch/3).astype(np.uint8)
        image = np.stack((
            image_red_ch, 
            image_green_ch, 
            image_blue_ch), -1)
        image = resize(image, (shape[0], shape[1]), mode='reflect')
        return image
                
