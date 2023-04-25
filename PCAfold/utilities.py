"""utilities.py: module for manifold optimization utilities."""

__author__ = "Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__copyright__ = "Copyright (c) 2020-2023, Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__credits__ = ["Department of Chemical Engineering, University of Utah, Salt Lake City, Utah, USA", "Universite Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Brussels, Belgium"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = ["Kamila Zdybal", "Elizabeth Armstrong"]
__email__ = ["kamilazdybal@gmail.com", "Elizabeth.Armstrong@chemeng.utah.edu", "James.Sutherland@chemeng.utah.edu"]
__status__ = "Production"

import numpy as np
import copy as cp
from termcolor import colored
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from keras import layers, models
from PCAfold.styles import *
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis

################################################################################
#
# Manifold optimization utilities
#
################################################################################

class QoIAwareProjection:
    """
    Enables computing QoI-aware encoder-decoder projections.

    """

    def __init__(self,
                input_data,
                n_components,
                projection_independent_outputs=None,
                projection_dependent_outputs=None,
                activation_decoder='linear',
                decoder_interior_architecture=(),
                encoder_weights_init=None,
                decoder_weights_init=None,
                hold_initialization=None,
                hold_weights=None,
                transformed_projection_dependent_outputs=None,
                loss='MSE',
                optimizer='Adam',
                batch_size=200,
                n_epochs=1000,
                learning_rate=0.001,
                validation_data=None,
                random_seed=None,
                verbose=False):

        __activations = ['linear', 'sigmoid', 'tanh', 'relu', 'selu']
        __projection_dependent_outputs_transformations = ['symlog', 'signed-square-root']
        __optimizers = ['Adam', 'Nadam']
        __losses = ['MSE', 'MAE']

        if not isinstance(input_data, np.ndarray):
            raise ValueError("Parameter `input_data` has to be of type `numpy.ndarray`.")

        (n_input_observations, n_input_variables) = np.shape(input_data)

        if projection_independent_outputs is None:
            if projection_dependent_outputs is None:
                raise ValueError("At least one of the parameters, `projection_independent_outputs` or `projection_dependent_outputs`, has to be of type `numpy.ndarray`.")

        if projection_independent_outputs is not None:
            if not isinstance(projection_independent_outputs, np.ndarray):
                raise ValueError("Parameter `projection_independent_outputs` has to be of type `numpy.ndarray`.")

            (n_projection_independent_output_observations, n_projection_independent_output_variables) = np.shape(projection_independent_outputs)

            if n_input_observations != n_projection_independent_output_observations:
                raise ValueError("Parameter `projection_independent_outputs` has to have the same number of observations as parameter `input_data`.")

        else:
            n_projection_independent_output_variables = 0

        if projection_dependent_outputs is not None:
            if not isinstance(projection_dependent_outputs, np.ndarray):
                raise ValueError("Parameter `projection_dependent_outputs` has to be of type `numpy.ndarray`.")

            n_projection_dependent_output_variables = n_components

            (n_projection_dependent_output_observations, _) = np.shape(projection_dependent_outputs)

            if n_projection_dependent_output_observations != n_input_observations:
                raise ValueError("Parameter `projection_dependent_outputs` has to have the same number of observations as parameter `input_data`.")

        else:
            n_projection_dependent_output_variables = 0

        if not isinstance(decoder_interior_architecture, tuple):
            raise ValueError("Parameter `decoder_interior_architecture` has to be of type `tuple`.")

        # Determine initialization of weights in the encoder
        if encoder_weights_init is not None:
            encoder_kernel_initializer = tf.constant_initializer(encoder_weights_init)
        else:
            encoder_kernel_initializer = 'glorot_uniform'

        # Determine initialization of weights in the decoder:
        if decoder_weights_init is not None:
            decoder_kernel_initializer = tf.constant_initializer(decoder_weights_init)
        else:
            decoder_kernel_initializer = 'glorot_uniform'

        if hold_initialization is not None:
            if not isinstance(hold_initialization, int):
                raise ValueError("Parameter `hold_initialization` has to be of type `int`.")

        if hold_weights is not None:
            if not isinstance(hold_weights, int):
                raise ValueError("Parameter `hold_weights` has to be of type `int`.")

        if transformed_projection_dependent_outputs is not None:
            if not isinstance(transformed_projection_dependent_outputs, str):
                raise ValueError("Parameter `transformed_projection_dependent_outputs` has to be of type `str`.")
            if transformed_projection_dependent_outputs not in __projection_dependent_outputs_transformations:
                raise ValueError("Parameter `transformed_projection_dependent_outputs` has to be 'symlog' or 'signed-square-root'.")

            n_transformed_projection_dependent_output_variables = n_components

        else:
            n_transformed_projection_dependent_output_variables = 0

        # Set the loss:
        if not isinstance(loss, str):
            raise ValueError("Parameter `loss` has to be of type `str`.")

        if loss not in __losses:
            raise ValueError("Parameter `loss` has to be 'MAE' or 'MSE'.")

        if loss == 'MSE':
            model_loss = tf.keras.losses.MeanSquaredError()
        elif loss == 'MAE':
            model_loss = tf.keras.losses.MeanAbsoluteError()

        # Set the optimizer:
        if not isinstance(optimizer, str):
            raise ValueError("Parameter `optimizer` has to be of type `str`.")

        if optimizer not in __optimizers:
            raise ValueError("Parameter `optimizer` has to be 'Adam' or 'Nadam'.")

        if optimizer == 'Adam':
            model_optimizer = tf.optimizers.Adam(learning_rate)
        elif optimizer == 'Nadam':
            model_optimizer = tf.optimizers.Nadam(learning_rate)

        if not isinstance(batch_size, int):
            raise ValueError("Parameter `batch_size` has to be of type `int`.")

        if not isinstance(n_epochs, int):
            raise ValueError("Parameter `n_epochs` has to be of type `int`.")

        if not isinstance(learning_rate, float):
            raise ValueError("Parameter `learning_rate` has to be of type `float`.")

        if validation_data is not None:
            if not isinstance(validation_data, tuple):
                raise ValueError("Parameter `validation_data` has to be of type `tuple`.")

        # Set random seed for neural network training reproducibility:
        if random_seed is not None:
            tf.random.set_seed(random_seed)

        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be a boolean.")

        n_total_outputs = n_projection_independent_output_variables + n_projection_dependent_output_variables + n_transformed_projection_dependent_output_variables

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Evaluate the architecture string:
        if len(decoder_interior_architecture)==0:
            architecture = str(n_input_variables) + '-' + str(n_components) + '-' + str(n_total_outputs)
        else:
            architecture = str(n_input_variables) + '-' + str(n_components) + '-' + '-'.join([str(i) for i in decoder_interior_architecture]) + '-' + str(n_total_outputs)

        # Create an encoder-decoder neural network with a given architecture:
        qoi_aware_encoder_decoder = models.Sequential()
        qoi_aware_encoder_decoder.add(layers.Dense(n_components, input_dim=n_input_variables, activation='linear', kernel_initializer=encoder_kernel_initializer))
        for n_neurons in decoder_interior_architecture:
            qoi_aware_encoder_decoder.add(layers.Dense(n_neurons, activation=activation_decoder, kernel_initializer=decoder_kernel_initializer))
        qoi_aware_encoder_decoder.add(layers.Dense(n_total_outputs, activation=activation_decoder, kernel_initializer=decoder_kernel_initializer))

        # Compile the neural network model:
        qoi_aware_encoder_decoder.compile(model_optimizer, loss=model_loss)

        # Attributes coming from user inputs:
        self.__input_data = input_data
        self.__n_components = n_components
        self.__projection_independent_outputs = projection_independent_outputs
        self.__projection_dependent_outputs = projection_dependent_outputs
        self.__activation_decoder = activation_decoder
        self.__decoder_interior_architecture = decoder_interior_architecture
        self.__hold_initialization = hold_initialization
        self.__hold_weights = hold_weights
        self.__transformed_projection_dependent_outputs = transformed_projection_dependent_outputs
        self.__loss = model_loss
        self.__optimizer = model_optimizer
        self.__batch_size = batch_size
        self.__n_epochs = n_epochs
        self.__learning_rate = learning_rate
        self.__validation_data = validation_data
        self.__random_seed = random_seed

        # Attributes computed at class object initialization:
        self.__architecture = architecture
        self.__qoi_aware_encoder_decoder = qoi_aware_encoder_decoder
        self.__weights_and_biases_init = qoi_aware_encoder_decoder.get_weights()
        self.__epochs_list = [e for e in range(0, n_epochs)]

        # Attributes available after model training:
        self.__training_loss = None
        self.__validation_loss = None
        self.__bases_across_epochs = None

    @property
    def input_data(self):
        return self.__input_data

    @property
    def output_data(self):
        return self.__output_data

    @property
    def n_components(self):
        return self.__n_components

    @property
    def architecture(self):
        return self.__architecture

    @property
    def qoi_aware_encoder_decoder(self):
        return self.__qoi_aware_encoder_decoder

    @property
    def weights_and_biases_init(self):
        return self.__weights_and_biases_init

    @property
    def training_loss(self):
        return self.__training_loss

    @property
    def validation_loss(self):
        return self.__validation_loss

# ------------------------------------------------------------------------------

    def summary(self):

        print('- '*40)
        print('Architecture:\n')
        print(self.architecture)
        print('- '*40)
        print('')

# ------------------------------------------------------------------------------

    def print_weights_and_biases_init(self):

        for i in range(0,len(self.weights_and_biases_init)):
            if i%2==0: print('Layers ' + str(int(i/2) + 1) + ' -- ' + str(int(i/2) + 2) + ': ' + '- '*20)
            if i%2==0:
                print('\nWeight:')
            else:
                print('Bias:')
            print(self.weights_and_biases_init[i])
            print()

# ------------------------------------------------------------------------------

    def train(self):

        pass
