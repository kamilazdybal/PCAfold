"""utilities.py: module for manifold optimization utilities."""

__author__ = "Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__copyright__ = "Copyright (c) 2020-2023, Kamila Zdybal, Elizabeth Armstrong, Alessandro Parente and James C. Sutherland"
__credits__ = ["Department of Chemical Engineering, University of Utah, Salt Lake City, Utah, USA", "Universite Libre de Bruxelles, Aero-Thermo-Mechanics Laboratory, Brussels, Belgium"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = ["Kamila Zdybal", "Elizabeth Armstrong"]
__email__ = ["kamilazdybal@gmail.com", "Elizabeth.Armstrong@chemeng.utah.edu", "James.Sutherland@chemeng.utah.edu"]
__status__ = "Production"

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import copy as cp
from termcolor import colored
import time
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import tensorflow as tf
from tensorflow.keras import layers, models, initializers
import tensorflow.compat.v1 as tf_v1
tf_v1.compat.v1.disable_eager_execution()
from PCAfold.styles import *
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis
from PCAfold import reconstruction

################################################################################
#
# Manifold optimization utilities
#
################################################################################

class QoIAwareProjection:
    """
    Enables computing QoI-aware encoder-decoder projections.

    The QoI-aware encoder-decoder is an autoencoder-like neural network that
    reconstructs important quantities of interest (QoIs) at the output of a decoder.
    The QoIs can be set to projection-independent variables (such as the original state variables)
    or projection-dependent variables, whose definition changes during neural network training.

    We introduce an intrusive modification to the neural network training process
    such that at each epoch, a low-dimensional basis matrix is computed from the current weights in the encoder.
    Any projection-dependent variables at the output get re-projected onto that basis.

    The rationale for performing dimensionality reduction with the QoI-aware strategy is
    that any poor topological behaviors on a low-dimensional projection will immediately increase the loss during training.
    These behaviors could be non-uniqueness in representing QoIs due to overlaps on a projection,
    or large gradients in QoIs caused by data compression in certain regions of a projection.
    Thus, the QoI-aware strategy naturally promotes improved projection topologies and can
    be useful in reduced-order modeling.

    An illustrative explanation of how the QoI-aware encoder-decoder works is presented in the figure below:

    .. image:: ../images/tutorial-qoi-aware-encoder-decoder.png
        :width: 700
        :align: center

    More information can be found in :cite:`Zdybal2023`.

    **Example:**

    .. code:: python

        from PCAfold import center_scale, QoIAwareProjection
        import numpy as np

        # Generate dummy dataset:
        X = np.random.rand(100,8)
        S = np.random.rand(100,8)

        # Request 2D QoI-aware encoder-decoder projection of the dataset:
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
                                       trainable_encoder_bias=True,
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

        (Model has been trained)


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

        	- Using 10% of input data as validation data
        	- Model will be trained on 90% of input data

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Hyperparameters:

        	- Batch size:		100
        	- # of epochs:		200
        	- Optimizer:		Adam
        	- Learning rate:	0.001
        	- Loss function:	MSE

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Weights initialization in the encoder:

        	- Glorot uniform

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Weights initialization in the decoder:

        	- Glorot uniform

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Weights updates in the encoder:

        	- Initial weights in the encoder will be kept for 10 first epochs
        	- Weights in the encoder will change once every 2 epochs

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Results reproducibility:

        	- Reproducible neural network training will be assured using random seed: 100

        = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        Training results:

        	- Minimum training loss:		0.0852246955037117
        	- Minimum training loss at epoch:	199

        	- Minimum validation loss:		0.06681100279092789
        	- Minimum validation loss at epoch:	182

        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    :param input_data:
        ``numpy.ndarray`` specifying the data set used as the input to the encoder-decoder. It should be of size ``(n_observations,n_variables)``.
    :param n_components:
        ``int`` specifying the dimensionality of the QoI-aware encoder-decoder projection. This is equal to the number of neurons in the bottleneck layer.
    :param projection_independent_outputs: (optional)
        ``numpy.ndarray`` specifying any projection-independent outputs at the decoder. It should be of size ``(n_observations,n_projection_independent_outputs)``.
    :param projection_dependent_outputs: (optional)
        ``numpy.ndarray`` specifying any projection-dependent outputs at the decoder. During training, ``projection_dependent_outputs`` is projected onto the current basis matrix and the decoder outputs are updated accordingly. It should be of size ``(n_observations,n_variables)``.
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
        ``numpy.ndarray`` specifying the custom initalization of the weights in the encoder. It should be of size ``(n_variables, n_components)``. If set to ``None``, weights in the encoder will be initialized using the Glorot uniform distribution.
    :param decoder_weights_init: (optional)
        ``tuple`` of ``numpy.ndarray`` specifying the custom initalization of the weights in the decoder. Each element in the tuple should have a shape that matches the architecture. If set to ``None``, weights in the encoder will be initialized using the Glorot uniform distribution.
    :param trainable_encoder_bias: (optional)
        ``bool`` specifying if the biases in the encoding layer should be trainable. Note, that for linear projections, bias does not change the projection quality and only acts to translate the projection along the low-dimensional coordinates.
    :param hold_initialization: (optional)
        ``int`` specifying the number of first epochs during which the initial weights in the encoder are held constant. If set to ``None``, weights in the encoder will change at the first epoch. This parameter can be used in conjunction with ``hold_weights``.
    :param hold_weights: (optional)
        ``int`` specifying how frequently the weights should be changed in the encoder. For example, if set to ``hold_weights=2``, the weights in the encoder will only be updated once every two epochs throught the whole training process. If set to ``None``, weights in the encoder will change at every epoch. This parameter can be used in conjunction with ``hold_initialization``.
    :param transformed_projection_dependent_outputs: (optional)
        ``str`` specifying if any nonlinear transformation of the projection-dependent outputs should be added at the decoder output. It can be ``'symlog'`` or ``'signed-square-root'``.
    :param transform_power: (optional)
        ``int`` or ``float`` as per ``preprocess.power_transform()``.
    :param transform_shift: (optional)
        ``int`` or ``float`` as per ``preprocess.power_transform()``.
    :param transform_sign_shift: (optional)
        ``int`` or ``float`` as per ``preprocess.power_transform()``.
    :param loss: (optional)
        ``str`` specifying the loss function. It can be ``'MAE'`` or ``'MSE'``.
    :param optimizer: (optional)
        ``str`` specifying the optimizer used during training. It can be ``'Adam'`` or ``'Nadam'``.
    :param batch_size: (optional)
        ``int`` specifying the batch size.
    :param n_epochs: (optional)
        ``int`` specifying the number of epochs.
    :param learning_rate: (optional)
        ``float`` specifying the learning rate passed to the optimizer.
    :param validation_perc: (optional)
        ``int`` specifying the percentage of the input data to be used as validation data during training. It should be a number larger than or equal to 0 and smaller than 100. Note, that if it is set above 0, not all of the input data will be used as training data. Note, that validation data does not impact model training!
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
                trainable_encoder_bias=True,
                hold_initialization=None,
                hold_weights=None,
                transformed_projection_dependent_outputs=None,
                transform_power=0.5,
                transform_shift=10**-4,
                transform_sign_shift=0.0,
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

        if not isinstance(n_components, int):
            raise ValueError("Parameter `n_components` has to be of type `int`.")

        if n_components < 1 or n_components >= n_input_variables:
            raise ValueError("Parameter `n_components` has to be larger than 0 and smaller than the dimensionality of the input data.")

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

            (n_projection_dependent_output_observations, n_output_variables) = np.shape(projection_dependent_outputs)

            if n_output_variables != n_input_variables:
                raise ValueError("Parameter `projection_dependent_outputs` has to have the same number of columns as `input_data` since it will be projected using the same basis matrix.")

            if n_projection_dependent_output_observations != n_input_observations:
                raise ValueError("Parameter `projection_dependent_outputs` has to have the same number of observations as parameter `input_data`.")

        else:
            n_projection_dependent_output_variables = 0

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

        self.__n_total_outputs = n_projection_independent_output_variables + n_projection_dependent_output_variables + n_transformed_projection_dependent_output_variables

        # Evaluate the architecture string:
        if len(decoder_interior_architecture)==0:
            architecture = str(n_input_variables) + '-' + str(n_components) + '-' + str(self.__n_total_outputs)
            neuron_count = [n_input_variables, n_components, self.__n_total_outputs]
        else:
            architecture = str(n_input_variables) + '-' + str(n_components) + '-' + '-'.join([str(i) for i in decoder_interior_architecture]) + '-' + str(self.__n_total_outputs)
            neuron_count = [n_input_variables, n_components] + [i for i in decoder_interior_architecture] + [self.__n_total_outputs]

        self.__neuron_count = neuron_count

        # Determine initialization of weights in the encoder
        if encoder_weights_init is not None:
            if not isinstance(encoder_weights_init, np.ndarray):
                raise ValueError("Parameter `encoder_weights_init` has to be of type `numpy.ndarray`.")
            # Shape of this array should be (n_input_variables, n_components) to match Kears:
            (n_encoder_weights_x, n_encoder_weights_y) = np.shape(encoder_weights_init)
            if not n_encoder_weights_x==n_input_variables or not n_encoder_weights_y==n_components:
                raise ValueError("Parameter `encoder_weights_init` has to have shape (n_input_variables, n_components).")
            encoder_kernel_initializer = tf.constant_initializer(encoder_weights_init)
        else:
            if random_seed is None:
                encoder_kernel_initializer = 'glorot_uniform'
            else:
                encoder_kernel_initializer = initializers.glorot_uniform(seed=random_seed)

        # Determine initialization of weights in the decoder:
        if decoder_weights_init is not None:
            if not isinstance(decoder_weights_init, tuple):
                raise ValueError("Parameter `decoder_weights_init` has to be of type `tuple`.")

            if len(decoder_weights_init) != len(decoder_interior_architecture)+1:
                raise ValueError("Parameter `decoder_weights_init` should have " + str(len(decoder_interior_architecture)+1) + " elements given the current network architecture.")

            for i, weight in enumerate(decoder_weights_init):
                if not isinstance(weight, np.ndarray):
                    raise ValueError("Elements of `decoder_weights_init` have to be of type `numpy.ndarray`.")

                (d1, d2) = np.shape(weight)
                if not d1==self.__neuron_count[1+i] or not d2==self.__neuron_count[i+2]:
                    raise ValueError("Shapes of elements in `decoder_weights_init` do not match the decoder architecture.")
            decoder_kernel_initializer = tuple([tf.constant_initializer(weight) for weight in decoder_weights_init])
        else:
            if random_seed is None:
                decoder_kernel_initializer = tuple(['glorot_uniform' for i in range(0,len(decoder_interior_architecture)+1)])
            else:
                decoder_kernel_initializer = tuple([initializers.glorot_uniform(seed=random_seed) for i in range(0,len(decoder_interior_architecture)+1)])

        if not isinstance(trainable_encoder_bias, bool):
            raise ValueError("Parameter `trainable_encoder_bias` has to be of type `bool`.")

        if hold_initialization is not None:
            if not isinstance(hold_initialization, int):
                raise ValueError("Parameter `hold_initialization` has to be of type `int`.")

        if hold_weights is not None:
            if not isinstance(hold_weights, int):
                raise ValueError("Parameter `hold_weights` has to be of type `int`.")

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
            model_optimizer = tf.optimizers.legacy.Adam(learning_rate)
        elif optimizer == 'Nadam':
            model_optimizer = tf.optimizers.legacy.Nadam(learning_rate)

        if not isinstance(batch_size, int):
            raise ValueError("Parameter `batch_size` has to be of type `int`.")

        if not isinstance(n_epochs, int):
            raise ValueError("Parameter `n_epochs` has to be of type `int`.")

        if not isinstance(learning_rate, float):
            raise ValueError("Parameter `learning_rate` has to be of type `float`.")

        if not isinstance(validation_perc, int):
            raise ValueError("Parameter `validation_perc` has to be of type `int`.")

        if (validation_perc < 0) or (validation_perc >= 100):
            raise ValueError("Parameter `validation_perc` has to be an integer between 0 and 100`.")

        # Set random seed for neural network training reproducibility:
        if random_seed is not None:
            if not isinstance(random_seed, int):
                raise ValueError("Parameter `random_seed` has to be of type `int`.")
            tf.random.set_seed(random_seed)
            tf.keras.utils.set_random_seed(random_seed)

        if not isinstance(verbose, bool):
            raise ValueError("Parameter `verbose` has to be a boolean.")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # Create an encoder-decoder neural network with a given architecture:
        qoi_aware_encoder_decoder = models.Sequential()
        qoi_aware_encoder_decoder.add(layers.Dense(n_components, input_dim=n_input_variables, use_bias=trainable_encoder_bias, activation='linear', kernel_initializer=encoder_kernel_initializer))
        for i, n_neurons in enumerate(decoder_interior_architecture):
            if isinstance(activation_decoder, str):
                qoi_aware_encoder_decoder.add(layers.Dense(n_neurons, activation=activation_decoder, kernel_initializer=decoder_kernel_initializer[i]))
            elif isinstance(activation_decoder, tuple):
                qoi_aware_encoder_decoder.add(layers.Dense(n_neurons, activation=activation_decoder[i], kernel_initializer=decoder_kernel_initializer[i]))
        if isinstance(activation_decoder, str):
            qoi_aware_encoder_decoder.add(layers.Dense(self.__n_total_outputs, activation=activation_decoder, kernel_initializer=decoder_kernel_initializer[-1]))
        elif isinstance(activation_decoder, tuple):
            qoi_aware_encoder_decoder.add(layers.Dense(self.__n_total_outputs, activation=activation_decoder[-1], kernel_initializer=decoder_kernel_initializer[-1]))

        # Compile the neural network model:
        qoi_aware_encoder_decoder.compile(model_optimizer, loss=model_loss)

        # Attributes coming from user inputs:
        self.__input_data = input_data
        self.__n_components = n_components
        self.__projection_independent_outputs = projection_independent_outputs
        self.__projection_dependent_outputs = projection_dependent_outputs
        self.__activation_decoder = activation_decoder
        self.__decoder_interior_architecture = decoder_interior_architecture
        self.__encoder_weights_init = encoder_weights_init
        self.__decoder_weights_init = decoder_weights_init
        self.__trainable_encoder_bias = trainable_encoder_bias
        self.__hold_initialization = hold_initialization
        self.__hold_weights = hold_weights
        self.__transformed_projection_dependent_outputs = transformed_projection_dependent_outputs
        self.__transform_power = transform_power
        self.__transform_shift = transform_shift
        self.__transform_sign_shift = transform_sign_shift
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
        self.__idx_min_training_loss = None
        self.__idx_min_validation_loss = None
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
            print('(Model has been trained)\n\n')
        else:
            print('(Model has not been trained yet)\n\n')

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
            print('\t- ' + 'Using ' + str(self.__validation_perc) + '% of input data as validation data')
        else:
            print('\t- ' + 'No validation data is used at model training')

        print('\t- ' + 'Model will be trained on ' + str(100 - self.__validation_perc) + '% of input data')

        print('\n' + '- '*60)

        print('Hyperparameters:\n')
        print('\t- ' + 'Batch size:\t\t' + str(self.__batch_size))
        print('\t- ' + '# of epochs:\t\t' + str(self.__n_epochs))
        print('\t- ' + 'Optimizer:\t\t' + self.__optimizer)
        print('\t- ' + 'Learning rate:\t' + str(self.__learning_rate))
        print('\t- ' + 'Loss function:\t' + self.__loss)
        print('\n' + '- '*60)

        print('Weights initialization in the encoder:\n')
        if self.__encoder_weights_init is None:
            print('\t- ' + 'Glorot uniform')
        else:
            print('\t- ' + 'User-provided custom initialization of the encoder')
        print('\n' + '- '*60)

        print('Weights initialization in the decoder:\n')
        if self.__decoder_weights_init is None:
            print('\t- ' + 'Glorot uniform')
        else:
            print('\t- ' + 'User-provided custom initialization of the decoder')
        print('\n' + '- '*60)

        print('Weights updates in the encoder:\n')
        if self.__hold_initialization is not None:
            print('\t- ' + 'Initial weights in the encoder will be kept for ' + str(self.__hold_initialization) + ' first epochs')
        else:
            print('\t- ' + 'Initial weights in the encoder will change after first epoch')
        if self.__hold_weights is not None:
            print('\t- ' + 'Weights in the encoder will change once every ' + str(self.__hold_weights) + ' epochs')
        else:
            print('\t- ' + 'Weights in the encoder will change at every epoch')
        print('\n' + '- '*60)

        print('Results reproducibility:\n')
        if self.__random_seed is not None:
            print('\t- ' + 'Reproducible neural network training will be assured using random seed: ' + str(self.__random_seed))
        else:
            print('\t- ' + 'Random seed not set, neural network training results will not be reproducible!')
        print('\n' + '= '*60)

        if self.__trained:

            idx_min_training_loss, = np.where(self.__training_loss==np.min(self.__training_loss))
            idx_min_training_loss = idx_min_training_loss[0]

            print('Training results:\n')
            print('\t- ' + 'Minimum training loss:\t\t' + str(np.min(self.__training_loss)))
            print('\t- ' + 'Minimum training loss at epoch:\t' + str(idx_min_training_loss+1))
            if self.__validation_perc != 0:
                idx_min_validation_loss, = np.where(self.__validation_loss==np.min(self.__validation_loss))
                idx_min_validation_loss = idx_min_validation_loss[0]
                print('\n\t- ' + 'Minimum validation loss:\t\t' + str(np.min(self.__validation_loss)))
                print('\t- ' + 'Minimum validation loss at epoch:\t' + str(idx_min_validation_loss+1))

            print('\n' + '- '*60)

# ------------------------------------------------------------------------------

    def train(self):
        """
        Trains the QoI-aware encoder-decoder neural network model.

        After training, the optimized basis matrix for low-dimensional data projection can be obtained.
        """

        if self.__verbose: print('Starting model training...\n\n')

        if self.__random_seed is not None:
            tf.random.set_seed(self.__random_seed)
            tf.keras.utils.set_random_seed(self.__random_seed)

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
                transformed_projection_dependent_outputs = preprocess.power_transform(current_projection_dependent_outputs, transform_power=self.__transform_power, transform_shift=self.__transform_shift, transform_sign_shift=self.__transform_sign_shift, invert=False)
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
                    transformed_projection_dependent_outputs = preprocess.power_transform(current_projection_dependent_outputs, transform_power=self.__transform_power, transform_shift=self.__transform_shift, transform_sign_shift=self.__transform_sign_shift, invert=False)
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

        idx_min_training_loss, = np.where(self.__training_loss==np.min(self.__training_loss))
        self.__idx_min_training_loss = idx_min_training_loss[0]

        if self.__validation_perc != 0:
            idx_min_validation_loss, = np.where(self.__validation_loss==np.min(self.__validation_loss))
            self.__idx_min_validation_loss = idx_min_validation_loss[0]

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

    def get_best_basis(self, method='min-training-loss'):
        """
        Returns the best low-dimensional basis according to the selected method.

        :param method: (optional)
            ``str`` specifying the method used to select the best basis. It should be ``'min-training-loss'``, ``'min-validation-loss'``, or ``'last-epoch'``.

        :return:
            - **best_basis** - ``numpy.ndarray`` specifying the best basis extracted from the ``bases_across_epochs`` attribute.
        """

        __methods = ['min-training-loss', 'min-validation-loss', 'last-epoch']

        if not isinstance(method, str):
            raise ValueError("Parameter `method` has to be of type `str`.")

        if method not in __methods:
            raise ValueError("Parameter `method` can only be 'min-training-loss', 'min-validation-loss', or 'last-epoch'.")

        if self.__trained:

            if method == 'min-training-loss':

                print('Minimum training loss:\t\t' + str(np.min(self.__training_loss)))
                print('Minimum training loss at epoch:\t' + str(self.__idx_min_training_loss+1))

                # We add one to the index, because the first basis correspond to the network intialization before training.
                # The length of the losses list is one less the length of the bases_across_epochs.
                best_basis = self.__bases_across_epochs[self.__idx_min_training_loss+1]

            elif method == 'min-validation-loss':

                if self.__validation_perc != 0:

                    print('Minimum validation loss:\t\t' + str(np.min(self.__validation_loss)))
                    print('Minimum validation loss at epoch:\t' + str(self.__idx_min_validation_loss+1))

                    # We add one to the index, because the first basis correspond to the network intialization before training.
                    # The length of the losses list is one less the length of the bases_across_epochs.
                    best_basis = self.__bases_across_epochs[self.__idx_min_validation_loss+1]

                else:

                    print('Validation loss not available.')

            elif method == 'last-epoch':

                print('Training loss at the last epoch:\t\t' + str(self.__training_loss[-1]))

                if self.__validation_perc != 0: print('Validation loss at the last epoch:\t\t' + str(self.__validation_loss[-1]))

                best_basis = self.__bases_across_epochs[-1]

        else:

            print('Model has not been trained yet!')

        return best_basis

# ------------------------------------------------------------------------------

    def plot_losses(self, markevery=100, figure_size=(15,5), save_filename=None):
        """
        Plots training and validation losses.

        :param markevery: (optional)
            ``int`` specifying how frequently the epoch number on the x-axis should be labelled.
        :param figure_size: (optional)
            ``tuple`` specifying figure size.
        :param save_filename: (optional)
            ``str`` specifying plot save location/filename. If set to ``None``
            plot will not be saved. You can also set a desired file extension,
            for instance ``.pdf``. If the file extension is not specified, the default
            is ``.png``.

        :return:
            - **plt** - ``matplotlib.pyplot`` plot handle.
        """

        if not isinstance(markevery, int):
            raise ValueError("Parameter `markevery` has to be of type `int`.")

        if not isinstance(figure_size, tuple):
            raise ValueError("Parameter `figure_size` has to be of type `tuple`.")

        if save_filename is not None:
            if not isinstance(save_filename, str):
                raise ValueError("Parameter `save_filename` has to be of type `str`.")

        if self.__trained:

            x_axis = [i for i in range(1,self.__n_epochs+1)]
            x_ticks = [1] + [i for i in range(1,self.__n_epochs) if i%markevery==0] + [self.__n_epochs]

            plt.figure(figsize=figure_size)
            plt.semilogy(x_axis, self.__training_loss, 'k', lw=3, label='Training loss')
            plt.scatter(self.__idx_min_training_loss+1, np.min(self.__training_loss), c='k', s=200, label='Min training loss', zorder=10)

            if self.__validation_perc != 0:
                plt.semilogy(x_axis, self.__validation_loss, 'r--', lw=2, label='Validation loss')
                plt.scatter(self.__idx_min_validation_loss+1, np.min(self.__validation_loss), c='r', s=100, label='Min validation loss', zorder=20)

            plt.xlabel('Epoch #', fontsize=font_labels)
            plt.xticks(x_ticks, rotation=90)
            plt.ylabel(self.__loss + ' loss', fontsize=font_labels)
            plt.legend(frameon=False, ncol=1, fontsize=font_legend)
            plt.grid(alpha=grid_opacity, zorder=1)

            if save_filename is not None: plt.savefig(save_filename, dpi=save_dpi, bbox_inches='tight')

            return plt

        else:

            print('Model has not been trained yet!')

# ------------------------------------------------------------------------------

def manifold_informed_forward_variable_addition(X, X_source, variable_names, scaling, bandwidth_values, target_variables=None, add_transformed_source=True, target_manifold_dimensionality=3, bootstrap_variables=None, penalty_function=None, power=1, vertical_shift=1, norm='max', integrate_to_peak=False, verbose=False):
    """
    Manifold-informed feature selection algorithm based on forward variable addition introduced in :cite:`Zdybal2022`. The goal of the algorithm is to
    select a meaningful subset of the original variables such that
    undesired behaviors on a PCA-derived manifold of a given dimensionality are minimized.
    The algorithm uses the cost function, :math:`\\mathcal{L}`, based on minimizing the area under the normalized variance derivatives curves, :math:`\\hat{\\mathcal{D}}(\\sigma)`,
    for the selected :math:`n_{dep}` dependent variables (as per ``cost_function_normalized_variance_derivative`` function).
    The algorithm can be bootstrapped in two ways:

    - Automatic bootstrap when ``bootstrap_variables=None``: the first best variable is selected automatically as the one that gives the lowest cost.

    - User-defined bootstrap when ``bootstrap_variables`` is set to a user-defined list of the bootstrap variables.

    The algorithm iterates, adding a new variable that exhibits the lowest cost at each iteration.
    The original variables in a data set get ordered according to their effect
    on the manifold topology. Assuming that the original data set is composed of :math:`Q` variables,
    the first output is a list of indices of the ordered
    original variables, :math:`\\mathbf{X} = [X_1, X_2, \\dots, X_Q]`. The second output is a list of indices of the selected
    subset of the original variables, :math:`\\mathbf{X}_S = [X_1, X_2, \\dots, X_n]`, that correspond to the minimum cost, :math:`\\mathcal{L}`.

    More information can be found in :cite:`Zdybal2022`.

    .. note::

        The algorithm can be very expensive (for large data sets) due to multiple computations of the normalized variance derivative.
        Try running it on multiple cores or on a sampled data set.

        In case the algorithm breaks when not being able to determine the peak
        location, try increasing the range in the ``bandwidth_values`` parameter.

    **Example:**

    .. code:: python

        from PCAfold import manifold_informed_forward_variable_addition as FVA
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)
        X_source = np.random.rand(100,10)

        # Define original variables to add to the optimization:
        target_variables = X[:,0:3]

        # Specify variables names
        variable_names = ['X_' + str(i) for i in range(0,10)]

        # Specify the bandwidth values to compute the optimization on:
        bandwidth_values = np.logspace(-4, 2, 50)

        # Run the subset selection algorithm:
        (ordered, selected, min_cost, costs) = FVA(X,
                                                   X_source,
                                                   variable_names,
                                                   scaling='auto',
                                                   bandwidth_values=bandwidth_values,
                                                   target_variables=target_variables,
                                                   add_transformed_source=True,
                                                   target_manifold_dimensionality=2,
                                                   bootstrap_variables=None,
                                                   penalty_function='peak',
                                                   norm='max',
                                                   integrate_to_peak=True,
                                                   verbose=True)

    :param X:
        ``numpy.ndarray`` specifying the original data set, :math:`\\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
    :param X_source:
        ``numpy.ndarray`` specifying the source terms, :math:`\\mathbf{S_X}`, corresponding to the state-space
        variables in :math:`\\mathbf{X}`. This parameter is applicable to data sets
        representing reactive flows. More information can be found in :cite:`Sutherland2009`. It should be of size ``(n_observations,n_variables)``.
    :param variable_names:
        ``list`` of ``str`` specifying variables names.
    :param scaling: (optional)
        ``str`` specifying the scaling methodology. It can be one of the following:
        ``'none'``, ``''``, ``'auto'``, ``'std'``, ``'pareto'``, ``'vast'``, ``'range'``, ``'0to1'``,
        ``'-1to1'``, ``'level'``, ``'max'``, ``'poisson'``, ``'vast_2'``, ``'vast_3'``, ``'vast_4'``.
    :param bandwidth_values:
        ``numpy.ndarray`` specifying the bandwidth values, :math:`\\sigma`, for :math:`\\hat{\\mathcal{D}}(\\sigma)` computation.
    :param target_variables: (optional)
        ``numpy.ndarray`` specifying the dependent variables that should be used in :math:`\\hat{\\mathcal{D}}(\\sigma)` computation. It should be of size ``(n_observations,n_target_variables)``.
    :param add_transformed_source: (optional)
        ``bool`` specifying if the PCA-transformed source terms of the state-space variables should be added in :math:`\\hat{\\mathcal{D}}(\\sigma)` computation, alongside the user-defined dependent variables.
    :param target_manifold_dimensionality: (optional)
        ``int`` specifying the target dimensionality of the PCA manifold.
    :param bootstrap_variables: (optional)
        ``list`` specifying the user-selected variables to bootstrap the algorithm with. If set to ``None``, automatic bootstrapping is performed.
    :param penalty_function: (optional)
        ``str`` specifying the weighting applied to each area.
        Set ``penalty_function='peak'`` to weight each area by the rightmost peak location, :math:`\\sigma_{peak, i}`, for the :math:`i^{th}` dependent variable.
        Set ``penalty_function='sigma'`` to weight each area continuously by the bandwidth.
        Set ``penalty_function='log-sigma-over-peak'`` to weight each area continuously by the :math:`\\log_{10}` -transformed bandwidth, normalized by the right most peak location, :math:`\\sigma_{peak, i}`.
        If ``penalty_function=None``, the area is not weighted.
    :param power: (optional)
        ``float`` or ``int`` specifying the power, :math:`r`. It can be used to control how much penalty should be applied to variance happening at the smallest length scales.
    :param vertical_shift: (optional)
        ``float`` or ``int`` specifying the vertical shift multiplier, :math:`b`. It can be used to control how much penalty should be applied to feature sizes.
    :param norm: (optional)
        ``str`` specifying the norm to apply for all areas :math:`A_i`. ``norm='average'`` uses an arithmetic average, ``norm='max'`` uses the :math:`L_{\\infty}` norm,
        ``norm='median'`` uses a median area, ``norm='cumulative'`` uses a cumulative area and ``norm='min'`` uses a minimum area.
    :param integrate_to_peak: (optional)
        ``bool`` specifying whether an individual area for the :math:`i^{th}` dependent variable should be computed only up the the rightmost peak location.
    :param verbose: (optional)
        ``bool`` for printing verbose details.

    :return:
        - **ordered_variables** - ``list`` specifying the indices of the ordered variables.
        - **selected_variables** - ``list`` specifying the indices of the selected variables that correspond to the minimum cost :math:`\\mathcal{L}`.
        - **optimized_cost** - ``float`` specifying the cost corresponding to the optimized subset.
        - **costs** - ``list`` specifying the costs, :math:`\\mathcal{L}`, from each iteration.
    """

    __penalty_functions = ['peak', 'sigma', 'log-sigma-over-peak']
    __norms = ['average', 'cumulative', 'max', 'median', 'min']

    if not isinstance(X, np.ndarray):
        raise ValueError("Parameter `X` has to be of type `numpy.ndarray`.")

    try:
        (n_observations, n_variables) = np.shape(X)
    except:
        raise ValueError("Parameter `X` has to have shape `(n_observations,n_variables)`.")

    if not isinstance(X_source, np.ndarray):
        raise ValueError("Parameter `X_source` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_source, n_variables_source) = np.shape(X_source)
    except:
        raise ValueError("Parameter `X_source` has to have shape `(n_observations,n_variables)`.")

    if n_variables_source != n_variables:
        raise ValueError("Parameter `X_source` has different number of variables than `X`.")

    if n_observations_source != n_observations:
        raise ValueError("Parameter `X_source` has different number of observations than `X`.")

    if not isinstance(variable_names, list):
        raise ValueError("Parameter `variable_names` has to be of type `list`.")

    if len(variable_names) != n_variables:
        raise ValueError("Parameter `variable_names` has different number of variables than `X`.")

    if not isinstance(scaling, str):
        raise ValueError("Parameter `scaling` has to be of type `str`.")

    if not isinstance(bandwidth_values, np.ndarray):
        raise ValueError("Parameter `bandwidth_values` has to be of type `numpy.ndarray`.")

    if target_variables is not None:
        if not isinstance(target_variables, np.ndarray):
            raise ValueError("Parameter `target_variables` has to be of type `numpy.ndarray`.")

        try:
            (n_d_hat_observations, n_target_variables) = np.shape(target_variables)
            target_variables_names = ['X' + str(i) for i in range(0,n_target_variables)]
        except:
            raise ValueError("Parameter `target_variables` has to have shape `(n_observations,n_target_variables)`.")

        if n_d_hat_observations != n_observations_source:
            raise ValueError("Parameter `target_variables` has different number of observations than `X_source`.")

    if not isinstance(add_transformed_source, bool):
        raise ValueError("Parameter `add_transformed_source` has to be of type `bool`.")

    if target_variables is None:
        if not add_transformed_source:
            raise ValueError("Either `target_variables` has to be specified or `add_transformed_source` has to be set to True.")

    if not isinstance(target_manifold_dimensionality, int):
        raise ValueError("Parameter `target_manifold_dimensionality` has to be of type `int`.")

    if bootstrap_variables is not None:
        if not isinstance(bootstrap_variables, list):
            raise ValueError("Parameter `bootstrap_variables` has to be of type `list`.")

    if penalty_function is not None:

        if not isinstance(penalty_function, str):
            raise ValueError("Parameter `penalty_function` has to be of type `str`.")

        if penalty_function not in __penalty_functions:
            raise ValueError("Parameter `penalty_function` has to be one of the following: 'peak', 'sigma', 'log-sigma-over-peak'.")

    if not isinstance(power, int) and not isinstance(power, float):
        raise ValueError("Parameter `power` has to be of type `float` or `int`.")

    if not isinstance(vertical_shift, int) and not isinstance(vertical_shift, float):
        raise ValueError("Parameter `vertical_shift` has to be of type `float` or `int`.")

    if not isinstance(norm, str):
        raise ValueError("Parameter `norm` has to be of type `str`.")

    if norm not in __norms:
        raise ValueError("Parameter `norm` has to be one of the following: 'average', 'cumulative', 'max', 'median', 'min'.")

    if not isinstance(integrate_to_peak, bool):
        raise ValueError("Parameter `integrate_to_peak` has to be of type `bool`.")

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be of type `bool`.")

    variables_indices = [i for i in range(0,n_variables)]

    costs = []

    # Automatic bootstrapping: -------------------------------------------------
    if bootstrap_variables is None:

        if verbose: print('Automatic bootstrapping...\n')

        bootstrap_cost_function = []

        bootstrap_tic = time.perf_counter()

        for i_variable in variables_indices:

            if verbose: print('\tCurrently checking variable:\t' + variable_names[i_variable])

            PCs = X[:,[i_variable]]
            PC_sources = X_source[:,[i_variable]]

            if target_variables is None:
                depvars = cp.deepcopy(PC_sources)
                depvar_names = ['SZ1']
            else:
                if add_transformed_source:
                    depvars = np.hstack((PC_sources, target_variables))
                    depvar_names = ['SZ1'] + target_variables_names
                else:
                    depvars = target_variables
                    depvar_names = target_variables_names

            bootstrap_variance_data = analysis.compute_normalized_variance(PCs, depvars, depvar_names=depvar_names, bandwidth_values=bandwidth_values)

            bootstrap_area = analysis.cost_function_normalized_variance_derivative(bootstrap_variance_data, penalty_function=penalty_function, power=power, vertical_shift=vertical_shift, norm=norm, integrate_to_peak=integrate_to_peak)
            if verbose: print('\tCost:\t%.4f' % bootstrap_area)
            bootstrap_cost_function.append(bootstrap_area)

        # Find a single best variable to bootstrap with:
        (best_bootstrap_variable_index, ) = np.where(np.array(bootstrap_cost_function)==np.min(bootstrap_cost_function))
        # This also handles cases where there are multiple minima with the same minimum value:
        best_bootstrap_variable_index = int(best_bootstrap_variable_index[0])

        costs.append(np.min(bootstrap_cost_function))

        bootstrap_variables = [best_bootstrap_variable_index]

        if verbose: print('\n\tVariable ' + variable_names[best_bootstrap_variable_index] + ' will be used as bootstrap.\n\tCost:\t%.4f' % np.min(bootstrap_cost_function) + '\n')

        bootstrap_toc = time.perf_counter()
        if verbose: print(f'Boostrapping time: {(bootstrap_toc - bootstrap_tic)/60:0.1f} minutes.' + '\n' + '-'*50)

    # Use user-defined bootstrapping: -----------------------------------------
    else:

        # Manifold dimensionality needs a fix here!
        if verbose: print('User-defined bootstrapping...\n')

        bootstrap_cost_function = []

        bootstrap_tic = time.perf_counter()

        if len(bootstrap_variables) < target_manifold_dimensionality:
            n_components = len(bootstrap_variables)
        else:
            n_components = cp.deepcopy(target_manifold_dimensionality)

        if verbose: print('\tUser-defined bootstrapping will be performed for a ' + str(n_components) + '-dimensional manifold.')

        bootstrap_pca = reduction.PCA(X[:,bootstrap_variables], scaling=scaling, n_components=n_components)
        PCs = bootstrap_pca.transform(X[:,bootstrap_variables])
        PC_sources = bootstrap_pca.transform(X_source[:,bootstrap_variables], nocenter=True)

        if target_variables is None:
            depvars = cp.deepcopy(PC_sources)
            depvar_names = ['SZ' + str(i) for i in range(0,n_components)]
        else:
            if add_transformed_source:
                depvars = np.hstack((PC_sources, target_variables))
                depvar_names = depvar_names = ['SZ' + str(i) for i in range(0,n_components)] + target_variables_names
            else:
                depvars = target_variables
                depvar_names = target_variables_names

        bootstrap_variance_data = analysis.compute_normalized_variance(PCs, depvars, depvar_names=depvar_names, bandwidth_values=bandwidth_values)

        bootstrap_area = analysis.cost_function_normalized_variance_derivative(bootstrap_variance_data, penalty_function=penalty_function, power=power, vertical_shift=vertical_shift, norm=norm, integrate_to_peak=integrate_to_peak)
        bootstrap_cost_function.append(bootstrap_area)
        costs.append(bootstrap_area)

        if verbose: print('\n\tVariable(s) ' + ', '.join([variable_names[i] for i in bootstrap_variables]) + ' will be used as bootstrap\n\tCost:\t%.4f' % np.min(bootstrap_area) + '\n')

        bootstrap_toc = time.perf_counter()
        if verbose: print(f'Boostrapping time: {(bootstrap_toc - bootstrap_tic)/60:0.1f} minutes.' + '\n' + '-'*50)

    # Iterate the algorithm starting from the bootstrap selection: -------------
    if verbose: print('Optimizing...\n')

    total_tic = time.perf_counter()

    ordered_variables = [i for i in bootstrap_variables]

    remaining_variables_list = [i for i in range(0,n_variables) if i not in bootstrap_variables]
    previous_area = np.min(bootstrap_cost_function)

    loop_counter = 0

    while len(remaining_variables_list) > 0:

        iteration_tic = time.perf_counter()

        loop_counter += 1

        if verbose:
            print('Iteration No.' + str(loop_counter))
            print('Currently adding variables from the following list: ')
            print([variable_names[i] for i in remaining_variables_list])

        current_cost_function = []

        for i_variable in remaining_variables_list:

            if len(ordered_variables) < target_manifold_dimensionality:
                n_components = len(ordered_variables) + 1
            else:
                n_components = cp.deepcopy(target_manifold_dimensionality)

            if verbose: print('\tCurrently added variable: ' + variable_names[i_variable])

            current_variables_list = ordered_variables + [i_variable]

            pca = reduction.PCA(X[:,current_variables_list], scaling=scaling, n_components=n_components)
            PCs = pca.transform(X[:,current_variables_list])
            PC_sources = pca.transform(X_source[:,current_variables_list], nocenter=True)

            if target_variables is None:
                depvars = cp.deepcopy(PC_sources)
                depvar_names = ['SZ' + str(i) for i in range(0,n_components)]
            else:
                if add_transformed_source:
                    depvars = np.hstack((PC_sources, target_variables))
                    depvar_names = depvar_names = ['SZ' + str(i) for i in range(0,n_components)] + target_variables_names
                else:
                    depvars = target_variables
                    depvar_names = target_variables_names

            current_variance_data = analysis.compute_normalized_variance(PCs, depvars, depvar_names=depvar_names, bandwidth_values=bandwidth_values)
            current_derivative, current_sigma, _ = analysis.normalized_variance_derivative(current_variance_data)

            current_area = analysis.cost_function_normalized_variance_derivative(current_variance_data, penalty_function=penalty_function, power=power, vertical_shift=vertical_shift, norm=norm, integrate_to_peak=integrate_to_peak)
            if verbose: print('\tCost:\t%.4f' % current_area)
            current_cost_function.append(current_area)

            if current_area <= previous_area:
                if verbose: print(colored('\tSAME OR BETTER', 'green'))
            else:
                if verbose: print(colored('\tWORSE', 'red'))

        min_area = np.min(current_cost_function)
        (best_variable_index, ) = np.where(np.array(current_cost_function)==min_area)
        # This also handles cases where there are multiple minima with the same minimum value:
        best_variable_index = int(best_variable_index[0])

        if verbose: print('\n\tVariable ' + variable_names[remaining_variables_list[best_variable_index]] + ' is added.\n\tCost:\t%.4f' % min_area + '\n')
        ordered_variables.append(remaining_variables_list[best_variable_index])
        remaining_variables_list = [i for i in range(0,n_variables) if i not in ordered_variables]
        if min_area <= previous_area:
            previous_area = min_area
        costs.append(min_area)

        iteration_toc = time.perf_counter()
        if verbose: print(f'\tIteration time: {(iteration_toc - iteration_tic)/60:0.1f} minutes.' + '\n' + '-'*50)

    # Compute the optimal subset where the cost is minimized: ------------------
    (min_cost_function_index, ) = np.where(costs==np.min(costs))
    # This also handles cases where there are multiple minima with the same minimum value:
    min_cost_function_index = int(min_cost_function_index[0])

    if min_cost_function_index+1 < target_manifold_dimensionality:
        selected_variables = list(np.array(ordered_variables)[0:target_manifold_dimensionality])
    else:
        selected_variables = list(np.array(ordered_variables)[0:min_cost_function_index+1])

    if verbose:

        print('Ordered variables:')
        print(', '.join([variable_names[i] for i in ordered_variables]))
        print(ordered_variables)
        print('Final cost: %.4f' % min_area)

        print('\nSelected variables:')
        print(', '.join([variable_names[i] for i in selected_variables]))
        print(selected_variables)
        print('Lowest cost: %.4f' % previous_area)

    total_toc = time.perf_counter()
    if verbose: print(f'\nOptimization time: {(total_toc - total_tic)/60:0.1f} minutes.' + '\n' + '-'*50)

    return ordered_variables, selected_variables, previous_area, costs

# ------------------------------------------------------------------------------

def manifold_informed_backward_variable_elimination(X, X_source, variable_names, scaling, bandwidth_values, target_variables=None, add_transformed_source=True, source_space=None, target_manifold_dimensionality=3, penalty_function=None, power=1, vertical_shift=1, norm='max', integrate_to_peak=False, verbose=False):
    """
    Manifold-informed feature selection algorithm based on backward variable elimination introduced in :cite:`Zdybal2022`. The goal of the algorithm is to
    select a meaningful subset of the original variables such that
    undesired behaviors on a PCA-derived manifold of a given dimensionality are minimized.
    The algorithm uses the cost function, :math:`\\mathcal{L}`, based on minimizing the area under the normalized variance derivatives curves, :math:`\\hat{\\mathcal{D}}(\\sigma)`,
    for the selected :math:`n_{dep}` dependent variables (as per ``cost_function_normalized_variance_derivative`` function).

    The algorithm iterates, removing another variable that has an effect of decreasing the cost the most at each iteration.
    The original variables in a data set get ordered according to their effect
    on the manifold topology. Assuming that the original data set is composed of :math:`Q` variables,
    the first output is a list of indices of the ordered
    original variables, :math:`\\mathbf{X} = [X_1, X_2, \\dots, X_Q]`. The second output is a list of indices of the selected
    subset of the original variables, :math:`\\mathbf{X}_S = [X_1, X_2, \\dots, X_n]`, that correspond to the minimum cost, :math:`\\mathcal{L}`.

    More information can be found in :cite:`Zdybal2022`.

    .. note::

        The algorithm can be very expensive (for large data sets) due to multiple computations of the normalized variance derivative.
        Try running it on multiple cores or on a sampled data set.

        In case the algorithm breaks when not being able to determine the peak
        location, try increasing the range in the ``bandwidth_values`` parameter.

    **Example:**

    .. code:: python

        from PCAfold import manifold_informed_backward_variable_elimination as BVE
        import numpy as np

        # Generate dummy data set:
        X = np.random.rand(100,10)
        X_source = np.random.rand(100,10)

        # Define original variables to add to the optimization:
        target_variables = X[:,0:3]

        # Specify variables names
        variable_names = ['X_' + str(i) for i in range(0,10)]

        # Specify the bandwidth values to compute the optimization on:
        bandwidth_values = np.logspace(-4, 2, 50)

        # Run the subset selection algorithm:
        (ordered, selected, min_cost, costs) = BVE(X,
                                                   X_source,
                                                   variable_names,
                                                   scaling='auto',
                                                   bandwidth_values=bandwidth_values,
                                                   target_variables=target_variables,
                                                   add_transformed_source=True,
                                                   target_manifold_dimensionality=2,
                                                   penalty_function='peak',
                                                   norm='max',
                                                   integrate_to_peak=True,
                                                   verbose=True)

    :param X:
        ``numpy.ndarray`` specifying the original data set, :math:`\\mathbf{X}`. It should be of size ``(n_observations,n_variables)``.
    :param X_source:
        ``numpy.ndarray`` specifying the source terms, :math:`\\mathbf{S_X}`, corresponding to the state-space
        variables in :math:`\\mathbf{X}`. This parameter is applicable to data sets
        representing reactive flows. More information can be found in :cite:`Sutherland2009`. It should be of size ``(n_observations,n_variables)``.
    :param variable_names:
        ``list`` of ``str`` specifying variables names. Order of names in the ``variable_names`` list should match the order of variables (columns) in ``X``.
    :param scaling: (optional)
        ``str`` specifying the scaling methodology. It can be one of the following:
        ``'none'``, ``''``, ``'auto'``, ``'std'``, ``'pareto'``, ``'vast'``, ``'range'``, ``'0to1'``,
        ``'-1to1'``, ``'level'``, ``'max'``, ``'poisson'``, ``'vast_2'``, ``'vast_3'``, ``'vast_4'``.
    :param bandwidth_values:
        ``numpy.ndarray`` specifying the bandwidth values, :math:`\\sigma`, for :math:`\\hat{\\mathcal{D}}(\\sigma)` computation.
    :param target_variables: (optional)
        ``numpy.ndarray`` specifying the dependent variables that should be used in :math:`\\hat{\\mathcal{D}}(\\sigma)` computation. It should be of size ``(n_observations,n_target_variables)``.
    :param add_transformed_source: (optional)
        ``bool`` specifying if the PCA-transformed source terms of the state-space variables should be added in :math:`\\hat{\\mathcal{D}}(\\sigma)` computation, alongside the user-defined dependent variables.
    :param source_space: (optional)
        ``str`` specifying the space to which the PC source terms should be transformed before computing the cost. It can be one of the following: ``symlog``, ``continuous-symlog``, ``original-and-symlog``, ``original-and-continuous-symlog``. If set to ``None``, PC source terms are kept in their original PCA-space.
    :param target_manifold_dimensionality: (optional)
        ``int`` specifying the target dimensionality of the PCA manifold.
    :param penalty_function: (optional)
        ``str`` specifying the weighting applied to each area.
        Set ``penalty_function='peak'`` to weight each area by the rightmost peak location, :math:`\\sigma_{peak, i}`, for the :math:`i^{th}` dependent variable.
        Set ``penalty_function='sigma'`` to weight each area continuously by the bandwidth.
        Set ``penalty_function='log-sigma-over-peak'`` to weight each area continuously by the :math:`\\log_{10}` -transformed bandwidth, normalized by the right most peak location, :math:`\\sigma_{peak, i}`.
        If ``penalty_function=None``, the area is not weighted.
    :param power: (optional)
        ``float`` or ``int`` specifying the power, :math:`r`. It can be used to control how much penalty should be applied to variance happening at the smallest length scales.
    :param vertical_shift: (optional)
        ``float`` or ``int`` specifying the vertical shift multiplier, :math:`b`. It can be used to control how much penalty should be applied to feature sizes.
    :param norm: (optional)
        ``str`` specifying the norm to apply for all areas :math:`A_i`. ``norm='average'`` uses an arithmetic average, ``norm='max'`` uses the :math:`L_{\\infty}` norm,
        ``norm='median'`` uses a median area, ``norm='cumulative'`` uses a cumulative area and ``norm='min'`` uses a minimum area.
    :param integrate_to_peak: (optional)
        ``bool`` specifying whether an individual area for the :math:`i^{th}` dependent variable should be computed only up the the rightmost peak location.
    :param verbose: (optional)
        ``bool`` for printing verbose details.

    :return:
        - **ordered_variables** - ``list`` specifying the indices of the ordered variables.
        - **selected_variables** - ``list`` specifying the indices of the selected variables that correspond to the minimum cost :math:`\\mathcal{L}`.
        - **optimized_cost** - ``float`` specifying the cost corresponding to the optimized subset.
        - **costs** - ``list`` specifying the costs, :math:`\\mathcal{L}`, from each iteration.
    """

    __penalty_functions = ['peak', 'sigma', 'log-sigma-over-peak']
    __norms = ['average', 'cumulative', 'max', 'median', 'min']
    __source_spaces = ['symlog', 'continuous-symlog', 'original-and-symlog', 'original-and-continuous-symlog']

    if not isinstance(X, np.ndarray):
        raise ValueError("Parameter `X` has to be of type `numpy.ndarray`.")

    try:
        (n_observations, n_variables) = np.shape(X)
    except:
        raise ValueError("Parameter `X` has to have shape `(n_observations,n_variables)`.")

    if not isinstance(X_source, np.ndarray):
        raise ValueError("Parameter `X_source` has to be of type `numpy.ndarray`.")

    try:
        (n_observations_source, n_variables_source) = np.shape(X_source)
    except:
        raise ValueError("Parameter `X_source` has to have shape `(n_observations,n_variables)`.")

    if n_variables_source != n_variables:
        raise ValueError("Parameter `X_source` has different number of variables than `X`.")

    # TODO: In the future, we might want to allow different number of observations, there is no reason why they should be equal.
    if n_observations_source != n_observations:
        raise ValueError("Parameter `X_source` has different number of observations than `X`.")

    if not isinstance(variable_names, list):
        raise ValueError("Parameter `variable_names` has to be of type `list`.")

    if len(variable_names) != n_variables:
        raise ValueError("Parameter `variable_names` has different number of variables than `X`.")

    if not isinstance(scaling, str):
        raise ValueError("Parameter `scaling` has to be of type `str`.")

    if not isinstance(bandwidth_values, np.ndarray):
        raise ValueError("Parameter `bandwidth_values` has to be of type `numpy.ndarray`.")

    if target_variables is not None:
        if not isinstance(target_variables, np.ndarray):
            raise ValueError("Parameter `target_variables` has to be of type `numpy.ndarray`.")
        try:
            (n_d_hat_observations, n_target_variables) = np.shape(target_variables)
            target_variables_names = ['X' + str(i) for i in range(0,n_target_variables)]
        except:
            raise ValueError("Parameter `target_variables` has to have shape `(n_observations,n_target_variables)`.")

        if n_d_hat_observations != n_observations_source:
            raise ValueError("Parameter `target_variables` has different number of observations than `X_source`.")

    if not isinstance(add_transformed_source, bool):
        raise ValueError("Parameter `add_transformed_source` has to be of type `bool`.")

    if target_variables is None:
        if not add_transformed_source:
            raise ValueError("Either `target_variables` has to be specified or `add_transformed_source` has to be set to True.")

    if source_space is not None:
        if not isinstance(source_space, str):
            raise ValueError("Parameter `source_space` has to be of type `str`.")
        if source_space.lower() not in __source_spaces:
            raise ValueError("Parameter `source_space` has to be one of the following: symlog`, `continuous-symlog`.")

    if not isinstance(target_manifold_dimensionality, int):
        raise ValueError("Parameter `target_manifold_dimensionality` has to be of type `int`.")

    if penalty_function is not None:
        if not isinstance(penalty_function, str):
            raise ValueError("Parameter `penalty_function` has to be of type `str`.")
        if penalty_function not in __penalty_functions:
            raise ValueError("Parameter `penalty_function` has to be one of the following: 'peak', 'sigma', 'log-sigma-over-peak'.")

    if not isinstance(power, int) and not isinstance(power, float):
        raise ValueError("Parameter `power` has to be of type `float` or `int`.")

    if not isinstance(vertical_shift, int) and not isinstance(vertical_shift, float):
        raise ValueError("Parameter `vertical_shift` has to be of type `float` or `int`.")

    if not isinstance(norm, str):
        raise ValueError("Parameter `norm` has to be of type `str`.")

    if norm not in __norms:
        raise ValueError("Parameter `norm` has to be one of the following: 'average', 'cumulative', 'max', 'median', 'min'.")

    if not isinstance(integrate_to_peak, bool):
        raise ValueError("Parameter `integrate_to_peak` has to be of type `bool`.")

    if not isinstance(verbose, bool):
        raise ValueError("Parameter `verbose` has to be of type `bool`.")

    costs = []

    if verbose: print('Optimizing...\n')

    if verbose:
        if add_transformed_source is not None:
            if source_space is not None:
                print('PC source terms will be assessed in the ' + source_space + ' space.\n')

    total_tic = time.perf_counter()

    remaining_variables_list = [i for i in range(0,n_variables)]

    ordered_variables = []

    loop_counter = 0

    # Iterate the algorithm: ---------------------------------------------------
    while len(remaining_variables_list) > target_manifold_dimensionality:

        iteration_tic = time.perf_counter()

        loop_counter += 1

        if verbose:
            print('Iteration No.' + str(loop_counter))
            print('Currently eliminating variable from the following list: ')
            print([variable_names[i] for i in remaining_variables_list])

        current_cost_function = []

        for i_variable in remaining_variables_list:

            if verbose: print('\tCurrently eliminated variable: ' + variable_names[i_variable])

            # Consider a subset with all variables but the currently eliminated one:
            current_variables_list = [i for i in remaining_variables_list if i != i_variable]

            if verbose:
                print('\tRunning PCA for a subset:')
                print('\t' + ', '.join([variable_names[i] for i in current_variables_list]))

            pca = reduction.PCA(X[:,current_variables_list], scaling=scaling, n_components=target_manifold_dimensionality)
            PCs = pca.transform(X[:,current_variables_list])
            (PCs, _, _) = preprocess.center_scale(PCs, '-1to1')

            if add_transformed_source:
                PC_sources = pca.transform(X_source[:,current_variables_list], nocenter=True)
                if source_space is not None:
                    if source_space == 'original-and-symlog':
                        transformed_PC_sources = preprocess.log_transform(PC_sources, method='symlog', threshold=1.e-4)
                    elif source_space == 'original-and-continuous-symlog':
                        transformed_PC_sources = preprocess.log_transform(PC_sources, method='continuous-symlog', threshold=1.e-4)
                    else:
                        transformed_PC_sources = preprocess.log_transform(PC_sources, method=source_space, threshold=1.e-4)

            if target_variables is None:
                if source_space == 'original-and-symlog' or source_space == 'original-and-continuous-symlog':
                    depvars = np.hstack((PC_sources, transformed_PC_sources))
                    depvar_names = ['SZ' + str(i) for i in range(0,target_manifold_dimensionality)] + ['symlog-SZ' + str(i) for i in range(0,target_manifold_dimensionality)]
                elif source_space == 'symlog' or source_space == 'continuous-symlog':
                    depvars = cp.deepcopy(transformed_PC_sources)
                    depvar_names = ['symlog-SZ' + str(i) for i in range(0,target_manifold_dimensionality)]
                else:
                    depvars = cp.deepcopy(PC_sources)
                    depvar_names = ['SZ' + str(i) for i in range(0,target_manifold_dimensionality)]
            else:
                if add_transformed_source:
                    if source_space == 'original-and-symlog' or source_space == 'original-and-continuous-symlog':
                        depvars = np.hstack((PC_sources, transformed_PC_sources, target_variables))
                        depvar_names = ['SZ' + str(i) for i in range(0,target_manifold_dimensionality)] + ['symlog-SZ' + str(i) for i in range(0,target_manifold_dimensionality)] + target_variables_names
                    elif source_space == 'symlog' or source_space == 'continuous-symlog':
                        depvars = np.hstack((transformed_PC_sources, target_variables))
                        depvar_names = ['symlog-SZ' + str(i) for i in range(0,target_manifold_dimensionality)] + target_variables_names
                    else:
                        depvars = np.hstack((PC_sources, target_variables))
                        depvar_names = ['SZ' + str(i) for i in range(0,target_manifold_dimensionality)] + target_variables_names
                else:
                    depvars = cp.deepcopy(target_variables)
                    depvar_names = cp.deepcopy(target_variables_names)

            current_variance_data = analysis.compute_normalized_variance(PCs, depvars, depvar_names=depvar_names, scale_unit_box = False, bandwidth_values=bandwidth_values)
            current_area = analysis.cost_function_normalized_variance_derivative(current_variance_data, penalty_function=penalty_function, power=power, vertical_shift=vertical_shift, norm=norm, integrate_to_peak=integrate_to_peak)
            if verbose: print('\tCost:\t%.4f' % current_area)
            current_cost_function.append(current_area)

            # Starting from the second iteration, we can make a comparison with the previous iteration's results:
            if loop_counter > 1:
                if current_area <= previous_area:
                    if verbose: print(colored('\tSAME OR BETTER', 'green'))
                else:
                    if verbose: print(colored('\tWORSE', 'red'))

        min_area = np.min(current_cost_function)
        costs.append(min_area)

        # Search for the variable whose removal will decrease the cost the most:
        (worst_variable_index, ) = np.where(np.array(current_cost_function)==min_area)
        # This also handles cases where there are multiple minima with the same minimum value:
        worst_variable_index = int(worst_variable_index[0])

        if verbose: print('\n\tVariable ' + variable_names[remaining_variables_list[worst_variable_index]] + ' is removed.\n\tCost:\t%.4f' % min_area + '\n')

        # Append removed variable in the ascending order, this list is later flipped to have variables ordered from most to least important:
        ordered_variables.append(remaining_variables_list[worst_variable_index])

        # Create a new list of variables to loop over at the next iteration:
        remaining_variables_list = [i for i in range(0,n_variables) if i not in ordered_variables]

        if loop_counter > 1:
            if min_area <= previous_area:
                previous_area = min_area
        else:
            previous_area = min_area

        iteration_toc = time.perf_counter()
        if verbose: print(f'\tIteration time: {(iteration_toc - iteration_tic)/60:0.1f} minutes.' + '\n' + '-'*50)

    # Compute the optimal subset where the overal cost from all iterations is minimized: ------------------

    # One last time remove the worst variable:
    del current_cost_function[worst_variable_index]

    for i in remaining_variables_list:
        ordered_variables.append(i)

    for i in range(0,len(remaining_variables_list)):
        costs.append(current_cost_function[i])

    # Invert lists to have variables ordered from most to least important:
    ordered_variables = ordered_variables[::-1]
    costs = costs[::-1]

    (min_cost_function_index, ) = np.where(costs==np.min(costs))
    # This also handles cases where there are multiple minima with the same minimum value:
    min_cost_function_index = int(min_cost_function_index[0])

    selected_variables = list(np.array(ordered_variables)[0:min_cost_function_index])

    optimized_cost = costs[min_cost_function_index]

    if verbose:

        print('Ordered variables:')
        print(', '.join([variable_names[i] for i in ordered_variables]))
        print(ordered_variables)
        print('Final cost: %.4f' % min_area)

        print('\nSelected variables:')
        print(', '.join([variable_names[i] for i in selected_variables]))
        print(selected_variables)
        print('Lowest cost: %.4f' % optimized_cost)

    total_toc = time.perf_counter()
    if verbose: print(f'\nOptimization time: {(total_toc - total_tic)/60:0.1f} minutes.' + '\n' + '-'*50)

    return ordered_variables, selected_variables, optimized_cost, costs

# ------------------------------------------------------------------------------

class QoIAwareProjectionPOUnet:
    """
    This is analogous to ``QoIAwareProjection`` but uses ``PartitionOfUnityNetwork`` as the decoder.

    More information can be found in :cite:`Armstrong2023`.

    **Example:**

    .. code:: python

        from PCAfold import init_uniform_partitions, PCA, QoIAwareProjectionPOUnet
        import numpy as np
        import tensorflow as tf

        # generate dummy data set:
        ivars = np.random.rand(100,3)

        # initialize a projection (e.g., using PCA)
        pca = PCA(ivars, scaling='none', n_components=2)
        ivar_proj = pca.transform(ivars)

        # initialize the QoIAwareProjectionPOUnet parameters
        net = QoIAwareProjectionPOUnet(pca.A[:,:2], **init_uniform_partitions([5,7], ivar_proj), basis_type='linear')

        # function for defining the training dependent variables (can include a projection)
        dvar = np.vstack((ivars[:,0] + ivars[:,1], 2.*ivars[:,0] + 3.*ivars[:,1], 3.*ivars[:,0] + 5.*ivars[:,1])).T
        def dvar_func(proj_weights):
            temp = tf.Variable(np.expand_dims(dvar, axis=2), name='eval_qoi', dtype=net._reconstruction._dtype)
            temp = net.tf_projection(temp, nobias=True)
            return temp

        # build the training graph with provided training data
        net.build_training_graph(ivars, dvar_func)

        # train the projection
        net.train(1000)

        # compute new projected variables
        net.projection(ivars)

        # evaluate the encoder-decoder
        net(ivars)

        # Save the data to a file
        net.write_data_to_file('filename.pkl')

        # reload projection data from file
        net2 = QoIAwareProjectionPOUnet.load_from_file('filename.pkl')

    :param projection_weights:
        array of the projection matrix weights
    :param partition_centers:
        array size (number of partitions) x (number of ivar inputs) for partition locations
    :param partition_shapes:
        array size (number of partitions) x (number of ivar inputs) for partition shapes influencing the RBF widths
    :param basis_type:
        string (``'constant'``, ``'linear'``, or ``'quadratic'``) for the degree of polynomial basis
    :param projection_biases:
        (optional, default None) array of the biases (offsets) corresponding to the projection weights, if ``None`` the projections are offset by zeros
    :param basis_coeffs:
        (optional, default ``None``) if the array of polynomial basis coefficients is known, it may be provided here,
        otherwise it will be initialized with ``build_training_graph`` and trained with ``train``
    :param dtype:
        (optional, default ``'float64'``) string specifying either float type ``'float64'`` or ``'float32'``

    **Attributes:**

    - **projection_weights** - (read only) array of the current projection weights
    - **projection_biases** - (read only) array of the projection biases
    - **reconstruction_model** - (read only) the current POUnet decoder
    - **partition_centers** - (read only) array of the current partition centers
    - **partition_shapes** - (read only) array of the current partition shape parameters
    - **basis_type** - (read only) string relaying the basis degree
    - **basis_coeffs** - (read only) array of the current basis coefficients
    - **proj_ivar_center** - (read only) array of the centering parameters used in the POUnet for the projected ivar inputs
    - **proj_ivar_scale** - (read only) array of the scaling parameters used in the POUnet for the projected ivar inputs
    - **dtype** - (read only) string relaying the data type (``'float64'`` or ``'float32'``)
    - **training_archive** - (read only) dictionary of the errors and POUnet states archived during training
    - **iterations** - (read only) array of the iterations archived during training
    """

    def __init__(self,
                 projection_weights,
                 partition_centers,
                 partition_shapes,
                 basis_type,
                 projection_biases=None,
                 basis_coeffs=None,
                 dtype='float64',
                 **kwargs
                ):
        self._projection_weights = projection_weights.copy()
        self._nstate = projection_weights.shape[0]
        self._nproj = projection_weights.shape[1]
        self._projection_biases = projection_biases.copy() if projection_biases is not None else np.zeros(partition_centers.shape[1])
        if self._projection_biases.shape[0] != self._nproj:
            raise ValueError("projection_biases size",self._projection_biases.shape[0],'must match dimensionality of projection weights',self._nproj)
        self._reconstruction = reconstruction.PartitionOfUnityNetwork(
                                                        partition_centers=partition_centers,
                                                        partition_shapes=partition_shapes,
                                                        basis_type=basis_type,
                                                        ivar_center=None,
                                                        ivar_scale=None,
                                                        basis_coeffs=basis_coeffs,
                                                        transform_power=1.,
                                                        transform_shift=0.,
                                                        transform_sign_shift=0.,
                                                        dtype=dtype
                                                       )
        if self._nproj != self._reconstruction._nd:
            raise ValueError("projection_weights dimensionality",self._nproj,"does not match reconstruction model input dimensionality",self._reconstruction._nd)
        self._sess = tf_v1.Session()
        self._lr = tf_v1.Variable(1.e-3, name='lr', dtype=self._reconstruction._dtype)
        self._proj_ivar_center = kwargs['ivar_center'] if 'ivar_center' in kwargs else self._reconstruction.ivar_center
        self._proj_inv_ivar_scale = 1./kwargs['ivar_scale'] if 'ivar_scale' in kwargs else 1./self._reconstruction.ivar_scale
        self._t_proj_ivar_center = tf_v1.Variable(np.expand_dims(self._proj_ivar_center, axis=2), name='proj_centers', dtype=self._reconstruction._dtype)
        self._t_proj_inv_ivar_scale = tf_v1.Variable(np.expand_dims(self._proj_inv_ivar_scale, axis=2), name='proj_scales', dtype=self._reconstruction._dtype)
        self._t_projection_biases = tf_v1.constant(self._projection_biases, name='biases', dtype=self._reconstruction._dtype)
        self._t_projection_weights = tf_v1.Variable(self._projection_weights, name='weights', dtype=self._reconstruction._dtype)
        self._sess.run(tf_v1.global_variables_initializer())
        self._built_graph = False

    @tf_v1.function
    def tf_projection(self, y, nobias=False):
        """
        version of ``projection`` using tensorflow operations and Tensors
        """
        const = 0. if nobias else self._t_projection_biases
        return tf_v1.reduce_sum(y * self._t_projection_weights, axis=1) + const

    def projection(self, ivars, nobias=False):
        """
        Projects the independent variable inputs using the current projection weights and biases

        :param ivars:
            array of independent variable query points
        :param nobias:
            (optional, default False) whether or not to apply the projection bias. Analogous to ``nocenter`` in the PCA ``transform`` function.

        :return:
            array of the projected independent variable query points
        """
        if ivars.shape[1] != self._nstate:
            raise ValueError("Dimensionality of inputs",y.shape[1],'does not match expected dimensionality',self._nstate)
        tf_y = tf_v1.Variable(np.expand_dims(ivars, axis=2), name='eval_pts', dtype=self._reconstruction._dtype)
        self._sess.run(tf_v1.variables_initializer([tf_y]))
        return self._sess.run(self.tf_projection(tf_y, nobias))

    def update_lr(self, lr):
        """
        update the learning rate for training

        :param lr:
            float for the learning rate
        """
        print('updating lr:', lr)
        self._sess.run(self._lr.assign(lr))

    def update_l2reg(self, l2reg):
        """
        update the least-squares regularization for training

        :param l2reg:
            float for the least-squares regularization
        """
        print('updating l2reg:', l2reg)
        self._sess.run(self._reconstruction._l2reg.assign(l2reg))

    def build_training_graph(self, ivars, dvars_function, error_type='abs', constrain_positivity=False, first_trainable_idx=0):
        """
        Construct the graph used during training (including defining the training errors) with the provided training data

        :param ivars:
            array of independent variables for training
        :param dvars_function:
            function (using tensorflow operations) for defining the dependent variable(s) for training.
            This must take a single argument of the projection weights which, if used, will be evaluated with the weights as they are updated
        :param error_type:
            (optional, default ``'abs'``) the type of training error: relative ``'rel'`` or absolute ``'abs'``
        :param constrain_positivity:
            (optional, default False) when True, it penalizes the training error with :math:`f - |f|` for dependent variables :math:`f`.
            This can be useful for defining projected source term dependent variables, for example.
        :param first_trainable_idx:
            (optional, default 0) This separates the trainable projection weights (with index greater than or equal to ``first_trainable_idx``) from the nontrainable projection weights.
        """
        if ivars.shape[1] != self._nstate:
            raise ValueError("Dimensionality of ivars",ivars.shape[1],'does not match expected dimensionality',self._nstate)
        if error_type != 'rel' and error_type != 'abs':
            raise ValueError("Supported error_type include rel and abs.")

        if first_trainable_idx<0 or first_trainable_idx>self._nproj-1:
            raise ValueError("first_trainable_idx must be greater than 0 and less than",self._nproj)
        self._t_train_weights = tf_v1.Variable(self._projection_weights[:,first_trainable_idx:], name='train', dtype=self._reconstruction._dtype)
        if first_trainable_idx != 0:
            self._t_nontrain_weights = tf_v1.constant(self._projection_weights[:,:first_trainable_idx], name='nontrain', dtype=self._reconstruction._dtype)
            self._t_projection_weights = tf_v1.squeeze(tf_v1.stack([self._t_nontrain_weights, self._t_train_weights], axis=1))
        else:
            self._t_projection_weights = self._t_train_weights

        self._t_yt = tf_v1.Variable(np.expand_dims(ivars, axis=2), name='eval_pts', dtype=self._reconstruction._dtype)
        self._t_xyt_prescale = tf_v1.expand_dims(self.tf_projection(self._t_yt), axis=2)

        self._xytmin = tf_v1.reduce_min(self._t_xyt_prescale, axis=0, keepdims=True)
        self._xytmax = tf_v1.reduce_max(self._t_xyt_prescale, axis=0, keepdims=True)
        self._t_proj_ivar_center = self._xytmin
        self._t_proj_inv_ivar_scale = 1./(self._xytmax - self._xytmin)
        self._t_xyt = (self._t_xyt_prescale - self._t_proj_ivar_center) * self._t_proj_inv_ivar_scale

        self._t_ft = dvars_function(self._t_projection_weights)
        self._reconstruction.build_training_graph(self._t_xyt, self._t_ft, error_type, constrain_positivity, istensor=True)

        self._opt = tf_v1.train.AdamOptimizer(learning_rate=self._lr).minimize(self._reconstruction._train_err, var_list=[self._t_train_weights, self._reconstruction._t_xp, self._reconstruction._t_sp])

        self._sess.run(tf_v1.global_variables_initializer())
        self._built_graph = True
        self._reconstruction._training_archive['data'] = list([self.__getstate__()])

    def train(self, iterations, archive_rate=100, use_best_archive_sse=True, verbose=False):
        """
        Performs training using a least-squares gradient descent block coordinate descent strategy.
        This alternates between updating the partition and projection parameters with gradient descent and updating the basis coefficients with least-squares.

        :param iterations:
            integer for number of training iterations to perform
        :param archive_rate:
            (optional, default 100) the rate at which the errors and parameters are archived during training. These can be accessed with the ``training_archive`` attribute
        :param use_best_archive_sse:
            (optional, default True) when True will set the POUnet parameters to those with the lowest error observed during training,
            otherwise the parameters from the last iteration are used
        :param verbose:
            (optional, default False) when True will print progress
        """
        if not self._built_graph:
            raise ValueError("Need to call build_training_graph before train.")
        if use_best_archive_sse and archive_rate>iterations:
            raise ValueError("Cannot archive the best parameters with archive_rate", archive_rate,'over',iterations,'iterations.')

        if verbose:
            print('-' * 60)
            print(f'  {"iteration":>10} | {"mean sqr":>10} | {"% max":>10}  | {"sum sqr":>10}')
            print('-' * 60)

        if use_best_archive_sse:
            best_trainable_weights = self._sess.run(self._t_train_weights).copy()
            best_error = self._sess.run(self._reconstruction._t_ssre).copy()
            best_centers = self._sess.run(self._reconstruction._t_xp).copy()
            best_shapes = self._sess.run(self._reconstruction._t_sp).copy()
            best_coeffs = self._sess.run(self._reconstruction._t_basis_coeffs).copy()

        for i in range(iterations):
            self._sess.run(self._opt) # update partitions & projection
            self._sess.run(self._reconstruction._lstsq) # update basis coefficients

            if not (i + 1) % archive_rate:
                msre, infre, ssre = self._sess.run(self._reconstruction._t_msre), self._sess.run(self._reconstruction._t_infre), self._sess.run(self._reconstruction._t_ssre)
                if verbose:
                    print(f'  {i + 1:10} | {msre:10.2e} | {100. * infre:10.2f}% | {ssre:10.2e}')
                self._reconstruction._iterations.append(self._reconstruction._iterations[-1] + archive_rate)
                self._reconstruction._training_archive['mse'].append(msre)
                self._reconstruction._training_archive['inf'].append(infre)
                self._reconstruction._training_archive['sse'].append(ssre)
                self._reconstruction._training_archive['data'].append(self.__getstate__())

                if use_best_archive_sse:
                    if ssre < best_error:
                        if verbose:
                            print('resetting best error')
                        best_error = ssre.copy()
                        best_trainable_weights = self._sess.run(self._t_train_weights).copy()
                        best_centers = self._sess.run(self._reconstruction._t_xp).copy()
                        best_shapes = self._sess.run(self._reconstruction._t_sp).copy()
                        best_coeffs = self._sess.run(self._reconstruction._t_basis_coeffs).copy()
        if use_best_archive_sse:
            self._sess.run(self._t_train_weights.assign(best_trainable_weights))
            self._sess.run(self._reconstruction._t_xp.assign(best_centers))
            self._sess.run(self._reconstruction._t_sp.assign(best_shapes))
            self._sess.run(self._reconstruction._t_basis_coeffs.assign(best_coeffs))

    def __call__(self, xeval):
        """
        evaluate the encoder-decoder

        :param xeval:
            array of independent variable query points

        :return:
            array of predictions
        """
        if xeval.shape[1] != self._nstate:
                raise ValueError("Dimensionality of inputs",xeval.shape[1],'does not match expected dimensionality',self._nstate)
        xeval_tf = tf_v1.Variable(np.expand_dims(xeval, axis=2), name='eval_pts', dtype=self._reconstruction._dtype)
        self._sess.run(tf_v1.variables_initializer([xeval_tf]))
        t_proj = tf_v1.expand_dims(self.tf_projection(xeval_tf), axis=2)
        t_proj_scale = (t_proj - self._t_proj_ivar_center) * self._t_proj_inv_ivar_scale

        pred = self._reconstruction.tf_call(t_proj_scale)
        return self._sess.run(pred)

    def __getstate__(self):
        """dictionary of current encoder-decoder parameters"""
        return dict(projection_weights=self.projection_weights,
                    projection_biases=self.projection_biases,
                    partition_centers=self.partition_centers,
                    partition_shapes=self.partition_shapes,
                    basis_type=self.basis_type,
                    ivar_center=self.proj_ivar_center,
                    ivar_scale=self.proj_ivar_scale,
                    basis_coeffs=self.basis_coeffs,
                    dtype=self.dtype
                   )

    def write_data_to_file(self, filename):
        """
        Save class data to a specified file using pickle. This does not include the archived data from training,
        which can be separately accessed with training_archive and saved outside of ``QoIAwareProjectionPOUnet``.

        :param filename:
            string
        """
        with open(filename, 'wb') as file_output:
            pickle.dump(self.__getstate__(), file_output)

    @classmethod
    def load_data_from_file(cls, filename):
        """
        Load data from a specified ``filename`` with pickle (following ``write_data_to_file``)

        :param filename:
            string

        :return:
            dictionary of the encoder-decoder data
        """
        with open(filename, 'rb') as file_input:
            pickled_data = pickle.load(file_input)
        return pickled_data

    @classmethod
    def load_from_file(cls, filename):
        """Load class from a specified ``filename`` with pickle (following ``write_data_to_file``)

        :param filename:
            string

        :return:
            ``QoIAwareProjectionPOUnet``
        """
        return cls(**cls.load_data_from_file(filename))

    @property
    def projection_weights(self):
        return self._sess.run(self._t_projection_weights)

    @property
    def projection_biases(self):
        return self._sess.run(self._t_projection_biases)

    @property
    def iterations(self):
        return self._reconstruction.iterations

    @property
    def training_archive(self):
        return self._reconstruction.training_archive

    @property
    def basis_type(self):
        return self._reconstruction.basis_type

    @property
    def dtype(self):
        return self._reconstruction.dtype

    @property
    def partition_centers(self):
        return self._sess.run(self._reconstruction._t_xp)

    @property
    def partition_shapes(self):
        return self._sess.run(self._reconstruction._t_sp)

    @property
    def basis_coeffs(self):
        return self._sess.run(self._reconstruction._t_basis_coeffs)

    @property
    def proj_ivar_center(self):
        return self._sess.run(self._t_proj_ivar_center)[:,:,0]

    @property
    def proj_ivar_scale(self):
        return 1./self._sess.run(self._t_proj_inv_ivar_scale)[:,:,0]

    @property
    def reconstruction_model(self):
        ivar_center = self.proj_ivar_center
        ivar_scale = self.proj_ivar_scale
        return reconstruction.PartitionOfUnityNetwork(
                                        partition_centers=self.partition_centers,
                                        partition_shapes=self.partition_shapes,
                                        basis_type=self.basis_type,
                                        ivar_center=ivar_center,
                                        ivar_scale=ivar_scale,
                                        basis_coeffs=self.basis_coeffs,
                                        transform_power=1.,
                                        transform_shift=0.,
                                        transform_sign_shift=0.,
                                        dtype=self.dtype
                                        )
