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
import warnings
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

    More information can be found in :cite:`Zdybal2023`.

    **Example:**

    .. code:: python

        from PCAfold import center_scale, QoIAwareProjection
        import numpy as np

        # Generate dummy dataset:
        X = np.random.rand(100,8)
        S = np.random.rand(100,8)

        # Create 2D encoder-decoder projection of the dataset:
        n_components = 2

        # Preprocess the dataset before passing it to the encoder-decoder:
        (input_data, centers, scales) = center_scale(X, scaling='0to1')
        projection_dependent_outputs = S / scales

        # Instantiate QoIAwareProjection class object:
        qoi_aware = QoIAwareProjection(input_data,
                                       n_components,
                                       projection_independent_outputs=input_data[:,0:3],
                                       projection_dependent_outputs=projection_dependent_outputs,
                                       activation_decoder=('tanh', 'tanh', 'linear'),
                                       decoder_interior_architecture=(5,8),
                                       encoder_weights_init=None,
                                       decoder_weights_init=None,
                                       hold_initialization=10,
                                       hold_weights=2,
                                       transformed_projection_dependent_outputs='signed-square-root',
                                       loss='MSE',
                                       optimizer='Adam',
                                       batch_size=100,
                                       n_epochs=200,
                                       learning_rate=0.001,
                                       validation_perc=10,
                                       random_seed=100,
                                       verbose=True)

        # Begin model training:
        qoi_aware.train()

    A summary of the current QoI-aware encoder-decoder model and its hyperparameter settings can be printed using the ``summary()`` function:

    .. code:: python

        # Print the QoI-aware encoder-decoder model summary
        qoi_aware.summary()

    .. code-block:: text

        QoI-aware encoder-decoder model summary...

        (Model has been trained.)


        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Projection dimensionality:

        	- 2D projection

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Encoder-decoder architecture:

        	8-2-5-8-7

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Activation functions:

        	(8)--linear--(2)--tanh--(5)--tanh--(8)--linear--(7)

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Variables at the decoder output:

        	- 3 projection independent variables
        	- 2 projection dependent variables
        	- 2 transformed projection dependent variables using signed-square-root

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Model validation:

        	- Using 10% of input data as validation data.
        	- Model will be trained on 90% of input data.

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Hyperparameters:

        	- Batch size:		100
        	- # of epochs:		200
        	- Optimizer:		Adam
        	- Learning rate:	0.001
        	- Loss function:	MSE

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Weights updates in the encoder:

        	- Initial weights in the encoder will be kept for 10 first epochs.
        	- Weights in the encoder will change once every 2 epochs.

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Results reproducibility:

        	- Reproducible neural network training will be assured using random seed: 100.

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    :param input_data:
        ``numpy.ndarray`` specifying the data set used as the input to the encoder-decoder. It should be of size ``(n_observations,n_variables)``.
    :param n_components:
        ``int`` specifying the dimensionality of the QoI-aware encoder-decoder projection. This is equal to the number of neurons in the bottleneck layer.
    :param projection_independent_outputs: (optional)
        ``numpy.ndarray`` specifying any projection-independent outputs at the decoder. It should be of size ``(n_observations,n_projection_independent_outputs)``.
    :param projection_dependent_outputs: (optional)
        ``numpy.ndarray`` specifying any projection-dependent outputs at the decoder. During training, ``projection_dependent_outputs`` is projected onto the current basis matrix and the decoder outputs are updated accordingly. It should be of size ``(n_observations,n_projection_dependent_outputs)``.
    :param activation_decoder: (optional)
        ``str`` or ``tuple`` specifying activation functions in all the decoding layers. If set to ``str``, the same activation function is used in all decoding layers.
        If set to a ``tuple`` of ``str``, a different activation function can be set at different decoding layers. The number of elements in the ``tuple`` should match the number of decoding layers!
        ``str`` and ``str`` elements of the ``tuple`` can only be ``'linear'``, ``'sigmoid'``, or ``'tanh'``. Note, that the activation function in the encoder is hardcoded to ``'linear'``.
    :param decoder_interior_architecture: (optional)
        ``tuple`` of ``int`` specifying the number of neurons in the interior architecture of a decoder.
        For example, if ``decoder_interior_architecture=(4,5)``, two interior decoding layers will be created and the overal network architecture will be ``(Input)-(Bottleneck)-(4)-(5)-(Output)``.
        If set to an empty tuple, ``decoder_interior_architecture=()``, the overal network architecture will be ``(Input)-(Bottleneck)-(Output)``.
        Keep in mind that if you'd like to create just one interior layer, you should use a comma after the integer: ``decoder_interior_architecture=(4,)``.
    :param encoder_weights_init: (optional)
        ``numpy.ndarray`` specifying the custom initalization of the weights in the encoder. If set to ``None``, weights in the encoder will be initialized using the Glorot uniform distribution.
    :param decoder_weights_init: (optional)
        ``numpy.ndarray`` specifying the custom initalization of the weights in the decoder. If set to ``None``, weights in the encoder will be initialized using the Glorot uniform distribution.
    :param hold_initialization: (optional)
        ``int`` specifying the number of first epochs during which the initial weights in the encoder are held constant. If set to ``None``, weights in the encoder will change at the first epoch. This parameter can be used in conjunction with ``hold_weights``.
    :param hold_weights: (optional)
        ``int`` specifying how frequently the weights should be changed in the encoder. For example, if set to ``hold_weights=2``, the weights in the encoder will only be updated once every two epochs throught the whole training process. If set to ``None``, weights in the encoder will change at every epoch. This parameter can be used in conjunction with ``hold_initialization``.
    :param transformed_projection_dependent_outputs: (optional)
        ``str`` specifying if any nonlinear transformation of the projection-dependent outputs should be added at the decoder output. It can be ``'symlog'`` or ``'signed-square-root'``.
    :param loss: (optional)
        ``str`` specifying the loss function. It can be ``'MAE'`` or ``'MSE'``.
    :param optimizer: (optional)
        ``str`` specifying the optimizer used during training. It can be ``'Adam'`` or ``'Nadam'``.
    :param batch_size: (optional)
        ``int`` specifying the batch size.
    :param n_epochs: (optional)
        ``int`` specifying the number of epochs.
    :param n_epochs: (optional)
        ``float`` specifying the learning rate passed to the optimizer.
    :param validation_perc: (optional)
        ``int`` specifying the percentage of the input data to be used as validation data during training. It should be a number between 0 and 100. Note, that if it is set above 0, not all of the input data will be used as training data. Note, that validation data does not impact model training!
    :param random_seed: (optional)
        ``int`` specifying the random seed to be used for any random operations. It is highly recommended to set a fixed random seed, as this allows for complete reproducibility of the results.
    :param verbose: (optional)
        ``bool`` for printing verbose details.

    **Attributes:**

    - **input_data** - (read only) ``numpy.ndarray`` specifying the data set used as the input to the encoder-decoder.
    - **n_components** - (read only) ``int`` specifying the dimensionality of the QoI-aware encoder-decoder projection.
    - **projection_independent_outputs** - (read only) ``numpy.ndarray`` specifying any projection-independent outputs at the decoder.
    - **projection_dependent_outputs** - (read only) ``numpy.ndarray`` specifying any projection-dependent outputs at the decoder.
    - **architecture** - (read only) ``str`` specifying the QoI-aware encoder-decoder architecture.
    - **n_total_outputs** - (read only) ``int`` counting the total number of outputs at the decoder.
    - **qoi_aware_encoder_decoder** - (read only) object of ``Keras.models.Sequential`` class that stores the QoI-aware encoder-decoder neural network.
    - **weights_and_biases_init** - (read only) ``list`` of ``numpy.ndarray`` specifying weights and biases with which the QoI-aware encoder-decoder was intialized.
    - **weights_and_biases_trained** - (read only) ``list`` of ``numpy.ndarray`` specifying weights and biases after training the QoI-aware encoder-decoder. Only available after calling ``QoIAwareProjection.train()``.
    - **training_loss** - (read only) ``list`` of losses computed on the training data. Only available after calling ``QoIAwareProjection.train()``.
    - **validation_loss** - (read only) ``list`` of losses computed on the validation data. Only available after calling ``QoIAwareProjection.train()`` and only when ``validation_perc`` was not equal to 0.
    - **bases_across_epochs** - (read only) ``list`` of ``numpy.ndarray`` specifying all basis matrices from all epochs. Only available after calling ``QoIAwareProjection.train()``.
    """

    def __init__(self,
                input_data,
                n_components,
                projection_independent_outputs=None,
                projection_dependent_outputs=None,
                activation_decoder='tanh',
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
                validation_perc=10,
                random_seed=None,
                verbose=False):

        __activations = ['linear', 'sigmoid', 'tanh']
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

        if not isinstance(activation_decoder, str) and not isinstance(activation_decoder, tuple):
            raise ValueError("Parameter `activation_decoder` has to be of type `str` or `tuple`.")

        if isinstance(activation_decoder, str):
            if activation_decoder not in __activations:
                raise ValueError("Parameter `activation_decoder` can only be 'linear' 'sigmoid' or 'tanh'.")

        if not isinstance(decoder_interior_architecture, tuple):
            raise ValueError("Parameter `decoder_interior_architecture` has to be of type `tuple`.")

        if isinstance(activation_decoder, tuple):
            for i in activation_decoder:
                if not isinstance(i, str):
                    raise ValueError("Parameter `activation_decoder` has to be a tuple of `str`.")
                if i not in __activations:
                    raise ValueError("Elements of the parameter `activation_decoder` can only be 'linear' 'sigmoid' or 'tanh'.")
            if len(activation_decoder) != len(decoder_interior_architecture) + 1:
                raise ValueError("Parameter `activation_decoder` has to have as many elements as there are layers in the decoder.")

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

            # The transformed projection dependent outputs can only be added IN ADDITION to the projection dependent outputs.
            # This can be modified in future implementations.
            if projection_dependent_outputs is not None:
                n_transformed_projection_dependent_output_variables = n_components
            else:
                n_transformed_projection_dependent_output_variables = 0
                warnings.warn("Parameter `transformed_projection_dependent_outputs` has been set, but no projection dependent outputs have been given! Transformed projection-dependent outputs will not be used at the decoder output.")

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

        if not isinstance(validation_perc, int):
            raise ValueError("Parameter `validation_perc` has to be of type `int`.")

        if (validation_perc < 0) or (validation_perc > 100):
            raise ValueError("Parameter `validation_perc` has to be an integer between 0 and 100`.")

        # Set random seed for neural network training reproducibility:
        if random_seed is not None:
            tf.random.set_seed(random_seed)

        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be a boolean.")

        self.__n_total_outputs = n_projection_independent_output_variables + n_projection_dependent_output_variables + n_transformed_projection_dependent_output_variables

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Evaluate the architecture string:
        if len(decoder_interior_architecture)==0:
            architecture = str(n_input_variables) + '-' + str(n_components) + '-' + str(self.__n_total_outputs)
            neuron_count = [n_input_variables, n_components, self.__n_total_outputs]
        else:
            architecture = str(n_input_variables) + '-' + str(n_components) + '-' + '-'.join([str(i) for i in decoder_interior_architecture]) + '-' + str(self.__n_total_outputs)
            neuron_count = [n_input_variables, n_components] + [i for i in decoder_interior_architecture] + [self.__n_total_outputs]

        self.__neuron_count = neuron_count

        # Create an encoder-decoder neural network with a given architecture:
        qoi_aware_encoder_decoder = models.Sequential()
        qoi_aware_encoder_decoder.add(layers.Dense(n_components, input_dim=n_input_variables, activation='linear', kernel_initializer=encoder_kernel_initializer))
        for i, n_neurons in enumerate(decoder_interior_architecture):
            if isinstance(activation_decoder, str):
                qoi_aware_encoder_decoder.add(layers.Dense(n_neurons, activation=activation_decoder, kernel_initializer=decoder_kernel_initializer))
            elif isinstance(activation_decoder, tuple):
                qoi_aware_encoder_decoder.add(layers.Dense(n_neurons, activation=activation_decoder[i], kernel_initializer=decoder_kernel_initializer))
        if isinstance(activation_decoder, str):
            qoi_aware_encoder_decoder.add(layers.Dense(self.__n_total_outputs, activation=activation_decoder, kernel_initializer=decoder_kernel_initializer))
        elif isinstance(activation_decoder, tuple):
            qoi_aware_encoder_decoder.add(layers.Dense(self.__n_total_outputs, activation=activation_decoder[-1], kernel_initializer=decoder_kernel_initializer))

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
        self.__loss = loss
        self.__loss_function = model_loss
        self.__optimizer = optimizer
        self.__model_optimizer = model_optimizer
        self.__batch_size = batch_size
        self.__n_epochs = n_epochs
        self.__learning_rate = learning_rate
        self.__validation_perc = validation_perc
        self.__random_seed = random_seed
        self.__verbose = verbose

        # Attributes computed at class object initialization:
        self.__architecture = architecture
        self.__qoi_aware_encoder_decoder = qoi_aware_encoder_decoder
        self.__weights_and_biases_init = qoi_aware_encoder_decoder.get_weights()
        self.__epochs_list = [e for e in range(0, n_epochs)]
        self.__trained = False

        # Attributes available after model training:
        self.__training_loss = None
        self.__validation_loss = None
        self.__bases_across_epochs = None
        self.__weights_and_biases_trained = None

    @property
    def input_data(self):
        return self.__input_data

    @property
    def n_components(self):
        return self.__n_components

    @property
    def projection_independent_outputs(self):
        return self.__projection_independent_outputs

    @property
    def projection_dependent_outputs(self):
        return self.__projection_dependent_outputs

    @property
    def architecture(self):
        return self.__architecture

    @property
    def n_total_outputs(self):
        return self.__n_total_outputs

    @property
    def qoi_aware_encoder_decoder(self):
        return self.__qoi_aware_encoder_decoder

    @property
    def weights_and_biases_init(self):
        return self.__weights_and_biases_init

    @property
    def weights_and_biases_trained(self):
        return self.__weights_and_biases_trained

    @property
    def training_loss(self):
        return self.__training_loss

    @property
    def validation_loss(self):
        return self.__validation_loss

    @property
    def bases_across_epochs(self):
        return self.__bases_across_epochs

# ------------------------------------------------------------------------------

    def summary(self):
        """
        Prints the QoI-aware encoder-decoder model summary.
        """

        print('QoI-aware encoder-decoder model summary...\n')
        if self.__trained:
            print('(Model has been trained.)\n\n')
        else:
            print('(Model has not been trained yet.)\n\n')

        print('- '*60)

        print('Projection dimensionality:\n')
        print('\t- ' + str(self.__n_components) + 'D projection')
        print('\n' + '- '*60)

        print('Encoder-decoder architecture:\n')
        print('\t' + self.architecture)
        print('\n' + '- '*60)

        print('Activation functions:\n')
        activation_function_string = ''
        for i, n_neurons in enumerate(self.__neuron_count):
            activation_function_string = activation_function_string + '(' + str(n_neurons) + ')'
            if i == 0:
                activation_function_string = activation_function_string + '--linear--'
            elif i < len(self.__neuron_count) - 1:
                if isinstance(self.__activation_decoder, str):
                    activation_function_string = activation_function_string + '--' + self.__activation_decoder + '--'
                elif isinstance(self.__activation_decoder, tuple):
                    activation_function_string = activation_function_string + '--' + self.__activation_decoder[i-1] + '--'
        print('\t' + activation_function_string)
        print('\n' + '- '*60)

        print('Variables at the decoder output:\n')
        if self.projection_independent_outputs is not None:
            print('\t- ' + str(self.projection_independent_outputs.shape[1]) + ' projection independent variables')
        if self.projection_dependent_outputs is not None:
            print('\t- ' + str(self.n_components) + ' projection dependent variables')
            if self.__transformed_projection_dependent_outputs is not None:
                print('\t- ' + str(self.n_components) + ' transformed projection dependent variables using ' + self.__transformed_projection_dependent_outputs)
        print('\n' + '- '*60)

        print('Model validation:\n')
        if self.__validation_perc != 0:
            print('\t- ' + 'Using ' + str(self.__validation_perc) + '% of input data as validation data.')
        else:
            print('\t- ' + 'No validation data is used at model training.')

        print('\t- ' + 'Model will be trained on ' + str(100 - self.__validation_perc) + '% of input data.')

        print('\n' + '- '*60)

        print('Hyperparameters:\n')
        print('\t- ' + 'Batch size:\t\t' + str(self.__batch_size))
        print('\t- ' + '# of epochs:\t\t' + str(self.__n_epochs))
        print('\t- ' + 'Optimizer:\t\t' + self.__optimizer)
        print('\t- ' + 'Learning rate:\t' + str(self.__learning_rate))
        print('\t- ' + 'Loss function:\t' + self.__loss)
        print('\n' + '- '*60)

        print('Weights updates in the encoder:\n')
        if self.__hold_initialization is not None:
            print('\t- ' + 'Initial weights in the encoder will be kept for ' + str(self.__hold_initialization) + ' first epochs.')
        else:
            print('\t- ' + 'Initial weights in the encoder will change after first epoch.')
        if self.__hold_weights is not None:
            print('\t- ' + 'Weights in the encoder will change once every ' + str(self.__hold_weights) + ' epochs.')
        else:
            print('\t- ' + 'Weights in the encoder will change at every epoch.')
        print('\n' + '- '*60)

        print('Results reproducibility:\n')
        if self.__random_seed is not None:
            print('\t- ' + 'Reproducible neural network training will be assured using random seed: ' + str(self.__random_seed) + '.')
        else:
            print('\t- ' + 'Random seed not set, neural network training results will not be reproducible!')
        print('\n' + '- '*60)

# ------------------------------------------------------------------------------

    def print_weights_and_biases_init(self):
        """
        Prints initial weights and biases from all layers of the QoI-aware encoder-decoder.
        """

        for i in range(0,len(self.weights_and_biases_init)):
            if i%2==0: print('Layers ' + str(int(i/2) + 1) + ' -- ' + str(int(i/2) + 2) + ': ' + '- '*20)
            if i%2==0:
                print('\nWeight:')
            else:
                print('Bias:')
            print(self.weights_and_biases_init[i])
            print()

# ------------------------------------------------------------------------------

    def print_weights_and_biases_trained(self):
        """
        Prints trained weights and biases from all layers of the QoI-aware encoder-decoder.
        """

        if self.__trained:

            for i in range(0,len(self.weights_and_biases_trained)):
                if i%2==0: print('Layers ' + str(int(i/2) + 1) + ' -- ' + str(int(i/2) + 2) + ': ' + '- '*20)
                if i%2==0:
                    print('\nWeight:')
                else:
                    print('Bias:')
                print(self.weights_and_biases_trained[i])
                print()

        else:

            print('Model has not been trained yet!')

# ------------------------------------------------------------------------------

    def train(self):
        """
        Trains the QoI-aware encoder-decoder neural network model.

        After training, the optimized basis matrix for low-dimensional data projection can be obtained.
        """

        if self.__verbose: print('Starting model training...\n\n')

        if self.__random_seed is not None:
            tf.random.set_seed(self.__random_seed)

        bases_across_epochs = []
        training_losses_across_epochs = []
        validation_losses_across_epochs = []

        # Determine the first basis:
        basis_init = self.weights_and_biases_init[0]
        basis_init = basis_init / np.linalg.norm(basis_init, axis=0)
        bases_across_epochs.append(basis_init)

        if self.projection_independent_outputs is not None:
            decoder_outputs = cp.deepcopy(self.projection_independent_outputs)

            if self.projection_dependent_outputs is not None:
                current_projection_dependent_outputs = np.dot(self.projection_dependent_outputs, basis_init)
                decoder_outputs = np.hstack((decoder_outputs, current_projection_dependent_outputs))

        else:
            current_projection_dependent_outputs = np.dot(self.projection_dependent_outputs, basis_init)
            decoder_outputs = cp.deepcopy(current_projection_dependent_outputs)

        if self.projection_dependent_outputs is not None:
            if self.__transformed_projection_dependent_outputs == 'symlog':
                transformed_projection_dependent_outputs = preprocess.log_transform(current_projection_dependent_outputs, method='continuous-symlog', threshold=1.e-4)
                decoder_outputs = np.hstack((decoder_outputs, transformed_projection_dependent_outputs))
            elif self.__transformed_projection_dependent_outputs == 'signed-square-root':
                transformed_projection_dependent_outputs = current_projection_dependent_outputs + 10**(-4)
                transformed_projection_dependent_outputs = np.sign(transformed_projection_dependent_outputs) * np.sqrt(np.abs(transformed_projection_dependent_outputs))
                decoder_outputs = np.hstack((decoder_outputs, transformed_projection_dependent_outputs))

        # Normalize the dependent variables to match the output activation function range:
        if isinstance(self.__activation_decoder, str):
            if self.__activation_decoder == 'tanh':
                decoder_outputs_normalized, _, _ = preprocess.center_scale(decoder_outputs, scaling='-1to1')
                if self.__verbose: print('Decoder outputs are scaled to a -1 to 1 range.')
            elif self.__activation_decoder == 'sigmoid':
                decoder_outputs_normalized, _, _ = preprocess.center_scale(decoder_outputs, scaling='0to1')
                if self.__verbose: print('Decoder outputs are scaled to a 0 to 1 range.')
            elif self.__activation_decoder == 'linear':
                decoder_outputs_normalized = cp.deepcopy(decoder_outputs)
        elif isinstance(self.__activation_decoder, tuple):
            if self.__activation_decoder[-1] == 'tanh':
                decoder_outputs_normalized, _, _ = preprocess.center_scale(decoder_outputs, scaling='-1to1')
                if self.__verbose: print('Decoder outputs are scaled to a -1 to 1 range.')
            elif self.__activation_decoder[-1] == 'sigmoid':
                decoder_outputs_normalized, _, _ = preprocess.center_scale(decoder_outputs, scaling='0to1')
                if self.__verbose: print('Decoder outputs are scaled to a 0 to 1 range.')
            elif self.__activation_decoder[-1] == 'linear':
                decoder_outputs_normalized = cp.deepcopy(decoder_outputs)

        (n_observations_decoder_outputs, n_decoder_outputs) = np.shape(decoder_outputs)

        if self.n_total_outputs != n_decoder_outputs:
            raise ValueError("There is a mismatch between requested and actual number of decoder outputs! This is PCAfold's issue.")

        tic = time.perf_counter()

        n_count_epochs = 0

        if self.__validation_perc != 0:
            sample_random = preprocess.DataSampler(np.zeros((n_observations_decoder_outputs,)).astype(int), random_seed=self.__random_seed, verbose=False)
            (idx_train, idx_validation) = sample_random.random(100 - self.__validation_perc)
            validation_data = (self.__input_data[idx_validation,:], decoder_outputs_normalized[idx_validation,:])
            if self.__verbose: print('Using ' + str(self.__validation_perc) + '% of input data as validation data. Model will be trained on ' + str(100 - self.__validation_perc) + '% of input data.\n')
        else:
            sample_random = preprocess.DataSampler(np.zeros((n_observations_decoder_outputs,)).astype(int), random_seed=self.__random_seed, verbose=False)
            (idx_train, _) = sample_random.random(100)
            validation_data = None
            if self.__verbose: print('No validation data is used at model training. Model will be trained on 100% of input data.\n')

        for i_epoch in tqdm(self.__epochs_list):

            history = self.__qoi_aware_encoder_decoder.fit(self.__input_data[idx_train,:],
                                                           decoder_outputs_normalized[idx_train,:],
                                                           epochs=1,
                                                           batch_size=self.__batch_size,
                                                           shuffle=True,
                                                           validation_data=validation_data,
                                                           verbose=0)

            # Holding the initial weights constant for `hold_initialization` first epochs:
            if self.__hold_initialization is not None:
                if i_epoch < self.__hold_initialization:
                    weights_and_biases = self.__qoi_aware_encoder_decoder.get_weights()
                    weights_and_biases[0] = self.weights_and_biases_init[0]
                    self.__qoi_aware_encoder_decoder.set_weights(weights_and_biases)
                    n_count_epochs = 0

            # Change the weights only once every `hold_weights` epochs:
            if self.__hold_weights is not None:
                if self.__hold_initialization is None:
                    weights_and_biases = self.__qoi_aware_encoder_decoder.get_weights()
                    if (n_count_epochs % self.__hold_weights) == 0:
                        previous_weights = weights_and_biases[0]
                    weights_and_biases[0] = previous_weights
                    self.__qoi_aware_encoder_decoder.set_weights(weights_and_biases)
                    n_count_epochs += 1
                else:
                    if i_epoch >= self.__hold_initialization:
                        weights_and_biases = self.__qoi_aware_encoder_decoder.get_weights()
                        if (n_count_epochs % self.__hold_weights) == 0:
                            previous_weights = weights_and_biases[0]
                        weights_and_biases[0] = previous_weights
                        self.__qoi_aware_encoder_decoder.set_weights(weights_and_biases)
                        n_count_epochs += 1

            # Update the projection-dependent output variables - - - - - - - - -

            # Determine the current basis:
            basis_current = self.__qoi_aware_encoder_decoder.get_weights()[0]
            basis_current = basis_current / np.linalg.norm(basis_current, axis=0)
            bases_across_epochs.append(basis_current)

            # print(self.__qoi_aware_encoder_decoder.get_weights()[0])

            if self.projection_independent_outputs is not None:
                decoder_outputs = self.projection_independent_outputs

                if self.projection_dependent_outputs is not None:
                    current_projection_dependent_outputs = np.dot(self.projection_dependent_outputs, basis_current)
                    decoder_outputs = np.hstack((decoder_outputs, current_projection_dependent_outputs))

            else:
                current_projection_dependent_outputs = np.dot(self.projection_dependent_outputs, basis_current)
                decoder_outputs = current_projection_dependent_outputs

            if self.projection_dependent_outputs is not None:
                if self.__transformed_projection_dependent_outputs == 'symlog':
                    transformed_projection_dependent_outputs = preprocess.log_transform(current_projection_dependent_outputs, method='continuous-symlog', threshold=1.e-4)
                    decoder_outputs = np.hstack((decoder_outputs, transformed_projection_dependent_outputs))
                elif self.__transformed_projection_dependent_outputs == 'signed-square-root':
                    transformed_projection_dependent_outputs = current_projection_dependent_outputs + 10**(-4)
                    transformed_projection_dependent_outputs = np.sign(transformed_projection_dependent_outputs) * np.sqrt(np.abs(transformed_projection_dependent_outputs))
                    decoder_outputs = np.hstack((decoder_outputs, transformed_projection_dependent_outputs))

            # Normalize the dependent variables to match the output activation function range:
            if isinstance(self.__activation_decoder, str):
                if self.__activation_decoder == 'tanh':
                    decoder_outputs_normalized, _, _ = preprocess.center_scale(decoder_outputs, scaling='-1to1')
                elif self.__activation_decoder == 'sigmoid':
                    decoder_outputs_normalized, _, _ = preprocess.center_scale(decoder_outputs, scaling='0to1')
                elif self.__activation_decoder == 'linear':
                    decoder_outputs_normalized = cp.deepcopy(decoder_outputs)
            elif isinstance(self.__activation_decoder, tuple):
                if self.__activation_decoder[-1] == 'tanh':
                    decoder_outputs_normalized, _, _ = preprocess.center_scale(decoder_outputs, scaling='-1to1')
                elif self.__activation_decoder[-1] == 'sigmoid':
                    decoder_outputs_normalized, _, _ = preprocess.center_scale(decoder_outputs, scaling='0to1')
                elif self.__activation_decoder[-1] == 'linear':
                    decoder_outputs_normalized = cp.deepcopy(decoder_outputs)

            # Determine the new validation data:
            if self.__validation_perc != 0:
                validation_data = (self.__input_data[idx_validation,:], decoder_outputs_normalized[idx_validation,:])

            training_losses_across_epochs.append(history.history['loss'][0])
            if self.__validation_perc != 0: validation_losses_across_epochs.append(history.history['val_loss'][0])

        toc = time.perf_counter()
        if self.__verbose: print(f'Time it took: {(toc - tic)/60:0.1f} minutes.\n')

        self.__training_loss = training_losses_across_epochs
        self.__validation_loss = validation_losses_across_epochs
        self.__bases_across_epochs = bases_across_epochs
        self.__weights_and_biases_trained = self.__qoi_aware_encoder_decoder.get_weights()
        self.__trained = True
