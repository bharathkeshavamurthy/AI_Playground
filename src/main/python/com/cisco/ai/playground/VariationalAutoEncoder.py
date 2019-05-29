# This entity captures the design of a Variational Auto Encoder (VAE) using TensorFlow and the high-level Keras API
# Author: Bharath Keshavamurthy
# Organization: CISCO Systems
# Copyright (c) 2019. All Rights Reserved.

import os
import glob
import numpy
import imageio
import IPython
import tensorflow
import matplotlib.pyplot as plt

tensorflow.enable_eager_execution()


# This class defines the inference network and the generator network employed in the design of the CVAE
class ConvolutionalVariationalAutoEncoder(tensorflow.keras.Model):

    # The initialization sequence
    # Latent variable \theta
    # Input \textbf{X}
    # Reconstructed input \hat{\textbf{X}}
    def __init__(self, _latent_dimension):
        super(ConvolutionalVariationalAutoEncoder, self).__init__()
        self.latent_dimension = _latent_dimension
        # The encoder - posterior probability distribution p(\theta|\textbf{X})
        self.encoder = tensorflow.keras.Sequential(
            [tensorflow.keras.layers.InputLayer(input_shape=(28, 28, 1)),
             tensorflow.keras.layers.Conv2D(filters=32, strides=(2, 2),
                                            kernel_size=3,
                                            activation=tensorflow.nn.relu),
             tensorflow.keras.layers.Conv2D(filters=64, strides=(2, 2),
                                            kernel_size=3,
                                            activation=tensorflow.nn.relu),
             tensorflow.keras.layers.Flatten(),
             tensorflow.keras.layers.Dense(
                 self.latent_dimension + self.latent_dimension)])
        # The decoder - likelihood probability distribution p(\textbf{X}|\theta)
        self.decoder = tensorflow.keras.Sequential(
            [tensorflow.keras.layers.InputLayer(input_shape=(self.latent_dimension,)),
             tensorflow.keras.layers.Dense(units=7 * 7 * 32, activation=tensorflow.nn.relu),
             tensorflow.keras.layers.Reshape(target_shape=(7, 7, 32)),
             tensorflow.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2),
                                                     activation=tensorflow.nn.relu,
                                                     padding='SAME'),
             tensorflow.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2),
                                                     activation=tensorflow.nn.relu,
                                                     padding='SAME'),
             tensorflow.keras.layers.Conv2DTranspose(filters=1, strides=(1, 1), kernel_size=3, padding='SAME')
             ])

    # The encode routine within the encoder
    def encode(self, x):
        # Sample mean and log_var from the posterior Gaussian distribution
        mean, log_var = tensorflow.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, log_var

    # The decode routine within the decoder
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        return (lambda: logits, lambda: tensorflow.sigmoid(logits))[apply_sigmoid]()

    # The sample routine
    def sample(self, eps=None):
        if eps is None:
            eps = tensorflow.random_normal(shape=(100, self.latent_dimension))
        return self.decode(eps, apply_sigmoid=True)

    # The reparameterization routine
    # Sample from a unit Gaussian, multiply by std and add the mean so that the gradients could pass through the...
    # ...sample to the inference network parameters
    @staticmethod
    def reparameterize(mean, log_var):
        eps = tensorflow.random_normal(shape=mean.shape)
        return eps * tensorflow.exp(log_var * 0.5) + mean


# This class encapsulates the training of a Variational Auto Encoder in order to generate handwritten digits
class VariationalAutoEncoder(object):

    # The initialization sequence
    def __init__(self):
        print('[INFO] VariationalAutoEncoder Initialization: Bringing things up...')
        (self.training_data, self.training_labels), (self.test_data, self.test_labels) = \
            tensorflow.keras.datasets.mnist.load_data()
        self.training_data = self.training_data.reshape(self.training_data.shape[0], 28, 28, 1).astype('float32')
        self.test_data = self.test_data.reshape(self.test_data.shape[0], 28, 28, 1).astype('float32')
        self.training_data = self.training_data / 255.0
        self.test_data = self.test_data / 255.0
        self.training_data[self.training_data >= 0.5] = 1.0
        self.training_data[self.training_data < 0.5] = 0.0
        self.test_data[self.test_data >= 0.5] = 1.0
        self.test_data[self.test_data < 0.5] = 0.0
        # The training dataset which can be fed into the tensorflow model for training
        self.training_data_set = tensorflow.data.Dataset.from_tensor_slices(
            self.training_data).shuffle(60000).batch(100)
        # The evaluation dataset which can be fed into the tensorflow model for evaluation
        self.test_data_set = tensorflow.data.Dataset.from_tensor_slices(self.test_data).shuffle(10000).batch(100)
        # The Optimizer
        self.optimizer = tensorflow.train.AdamOptimizer(1e-4)

    # Develop a log normal and return it for computing the loss
    @staticmethod
    def log_normal(sample, mean, log_var, raxis=1):
        return tensorflow.reduce_sum(-0.5 * ((sample - mean) ** 2.0) + log_var + tensorflow.log(2.0 * numpy.pi),
                                     axis=raxis)

    # Compute the loss
    # log p(x|\theta) + log p(\theta) - log q(\theta|x)
    def compute_loss(self, x, model):
        mean, log_var = model.encode(x)
        theta = model.reparameterize(mean, log_var)
        x_hat = model.decode(theta)
        # The decoder loss
        cross_entropy = tensorflow.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x)
        # The decoder loss is encapsulated in this log
        log_p_x_theta = -tensorflow.reduce_sum(cross_entropy, axis=[1, 2, 3])
        # The prior is encapsulated in this log_normal
        log_p_theta = self.log_normal(theta, 0.0, 0.0)
        # The posterior is encapsulated in this log_normal
        log_p_theta_x = self.log_normal(theta, mean, log_var)
        return -tensorflow.reduce_mean(log_p_x_theta + log_p_theta - log_p_theta_x)

    # Compute the gradients
    def compute_gradients(self, x, model):
        with tensorflow.GradientTape() as gradient_tape:
            loss = self.compute_loss(x, model)
        return gradient_tape.gradient(loss, model.trainable_variables), loss

    # Gradient Descent step
    def apply_gradients(self, gradients, variables, global_step=None):
        self.optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

    # Generate and save images
    @staticmethod
    def generate_and_save_images(model, epoch, test_input):
        predictions = model.sample(test_input)
        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')
        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        plt.close(fig)

    # Build and Train the CVAE model
    def build_and_train_the_model(self):
        random_vector_for_generation = tensorflow.random_normal(shape=[16, 50])
        model = ConvolutionalVariationalAutoEncoder(50)
        self.generate_and_save_images(model, 0, random_vector_for_generation)
        # Start the Gradient Descent
        for epoch in range(1, 11):
            for train_x in self.training_data_set:
                gradients, loss = self.compute_gradients(train_x, model)
                self.apply_gradients(gradients, model.trainable_variables)
                self.generate_and_save_images(model, epoch, random_vector_for_generation)

    # Make a gif from the saved images and show it
    @staticmethod
    def display():
        with imageio.get_writer('cvae.gif', mode='I') as writer:
            filenames = glob.glob('image*.png')
            filenames = sorted(filenames)
            last = -1
            for i, filename in enumerate(filenames):
                frame = 2 * (i ** 0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)
        # this is a hack to display the gif inside the notebook
        os.system('cp cvae.gif cvae.gif.png')
        IPython.display.Image(filename='cvae.gif.png')

    # The termination sequence
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[INFO] VariationalAutoEncoder Termination: Tearing things down...')


# Run Trigger
if __name__ == '__main__':
    print('[INFO] VariationalAutoEncoder Trigger: Starting system assessment!')
    VAE = VariationalAutoEncoder()
    VAE.build_and_train_the_model()
    VAE.display()
