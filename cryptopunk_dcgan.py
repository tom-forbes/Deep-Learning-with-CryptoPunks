# Recommened running this script in Google COLAB with a GPU.

import numpy as np
import pandas as pd
import os

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.fashion_mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate

from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from keras.models import load_model
from matplotlib import pyplot

import glob
import imageio

# Read in image and label arrays
x_train = np.load('x_punk.npy', allow_pickle=True)
y_train = np.load('y_punk.npy', allow_pickle=True)

# Make directory for storing training epoch images
try:
  os.mkdir('pics')
except:
  pass

# Make mutliclass array for punk type
d = pd.DataFrame(y_train[:,:2])
d[0] = d[0]*1
d[1] = d[1]*2
d['class'] = d.sum(axis=1)
y = np.array(d['class'])
y

class_embedding = 10 
n_classes = 2

# define the standalone discriminator model
def define_discriminator(in_shape=(42,42,3), n_classes=n_classes, class_embedding=class_embedding):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, class_embedding)(in_label)
	# scale up to image dimensions with linear activation
	n_nodes = in_shape[0] * in_shape[1]
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((in_shape[0], in_shape[1], 1))(li)
	# image input
	in_image = Input(shape=in_shape)
	# concat label as a channel
	merge = Concatenate()([in_image, li])
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
	fe = LeakyReLU(alpha=0.2)(fe)
	# downsample
	fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# dropout
	fe = Dropout(0.4)(fe)
	# output
	out_layer = Dense(1, activation='sigmoid')(fe)
	# define model
	model = Model([in_image, in_label], out_layer)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
 
# define the standalone generator model
def define_generator(latent_dim, n_classes=n_classes, class_embedding=class_embedding):
	# label input
	in_label = Input(shape=(1,))
	# embedding for categorical input
	li = Embedding(n_classes, class_embedding)(in_label)
	# linear multiplication
	n_nodes = 7 * 7
	li = Dense(n_nodes)(li)
	# reshape to additional channel
	li = Reshape((7, 7, 1))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	gen = Dense(n_nodes)(in_lat)
	gen = LeakyReLU(alpha=0.2)(gen)
	gen = Reshape((7, 7, 128))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 14x14
	gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
	gen = LeakyReLU(alpha=0.2)(gen)
	# upsample to 28x28
	gen = Conv2DTranspose(128, (4,4), strides=(3,3), padding='same')(gen)
	gen = LeakyReLU(alpha=0.2)(gen)
	# output
	out_layer = Conv2D(3, (7,7), activation='tanh', padding='same')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)
	return model
 
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# get noise and label inputs from generator model
	gen_noise, gen_label = g_model.input
	# get image output from the generator model
	gen_output = g_model.output
	# connect image output and label input from generator as inputs to discriminator
	gan_output = d_model([gen_output, gen_label])
	# define gan model as taking noise and label and outputting a classification
	model = Model([gen_noise, gen_label], gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
 
# load fashion mnist images
def load_real_samples():
    # load dataset
    #(trainX, trainy), (_, _) = load_data()
    X = x_train
    trainy = y
    # expand to 3d, e.g. add channels
    #X = expand_dims(trainX, axis=-1)
    # convert from ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 0.5) * 2
    return [X, trainy]
 
# # select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=n_classes):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(1, n_classes+1, n_samples)
	return [z_input, labels]

 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y

 
# create and save a plot of generated images
def save_plot(examples, n, epoch):
    # plot images
    for i in range( 2* n * n):  # 2*n*n
      # define subplot
      pyplot.subplot( 2* n, n,1+i) # n, 2* n , 1+i
      # turn off axis
      pyplot.axis('off')
      # plot raw pixel data
      pyplot.imshow(examples[i, :, :, :])#, cmap='RdYlBu')

    pyplot.savefig(f'pics/{epoch}.png')
    pyplot.show()
 
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
      # enumerate batches over the training set
      for j in range(bat_per_epo):
        # get randomly selected 'real' samples
        [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
        # update discriminator model weights
        d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
        # generate 'fake' examples
        [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator model weights
        d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
        # prepare points in latent space as input for the generator
        [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
        # summarize loss on this batch
        print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
          (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
      # save the generator model
      g_model.save(f'cgan_generator{i}.h5')
    
      model = load_model(f'cgan_generator{i}.h5')
      # generate images
      latent_points, labels = generate_latent_points(100, 100)
      # specify labels
      labels = asarray([x for _ in range(50) for x in range(1,3)])
      # generate images
      X  = model.predict([latent_points, labels])
      # scale from [-1,1] to [0,1]
      X = (X + 1) / 2.0
      # plot the result
      save_plot(X, 2, i)
 
 
# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator(n_classes=n_classes, class_embedding=class_embedding)
# create the generator
g_model = define_generator(latent_dim, n_classes=n_classes, class_embedding=class_embedding)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim, 100)
 

n_batch = 128
n_epochs = 100
bat_per_epo = int(dataset[0].shape[0] / n_batch)
half_batch = int(n_batch / 2)
# manually enumerate epochs
for i in range(n_epochs):
# enumerate batches over the training set
	for j in range(bat_per_epo):
		# get randomly selected 'real' samples
		[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
		# update discriminator model weights
		d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
		# generate 'fake' examples
		[X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model weights
		d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
		# prepare points in latent space as input for the generator
		[z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
		# summarize loss on this batch
		print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
		(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))

anim_file = 'dcgan_gender.gif'
# Create GIF using epoch for each frame.
with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('pics/*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

# Display GIF 
# !pip install -q git+https://github.com/tensorflow/docs 
import tensorflow_docs
from tensorflow_docs.vis import embed
embed.embed_file(anim_file)