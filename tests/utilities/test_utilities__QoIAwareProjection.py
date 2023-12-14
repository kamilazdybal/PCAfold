import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import analysis
from PCAfold import QoIAwareProjection
from tensorflow import optimizers
from tensorflow.keras import initializers

class Utilities(unittest.TestCase):

    def test_utilities__QoIAwareProjection__allowed_class_init(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2

        try:
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data)
        except:
            self.assertTrue(False)

        try:
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Nadam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            activation_decoder='sigmoid',
                                            decoder_interior_architecture=(4,5,6),
                                            encoder_weights_init=None,
                                            decoder_weights_init=None,
                                            hold_initialization=4,
                                            hold_weights=2,
                                            transformed_projection_dependent_outputs='symlog',
                                            loss='MAE',
                                            batch_size=10,
                                            n_epochs=20,
                                            validation_perc=0,
                                            random_seed=100,
                                            verbose=False)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__training(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2

        try:
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Nadam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            activation_decoder='sigmoid',
                                            decoder_interior_architecture=(4,5),
                                            encoder_weights_init=None,
                                            decoder_weights_init=None,
                                            hold_initialization=4,
                                            hold_weights=2,
                                            transformed_projection_dependent_outputs='symlog',
                                            loss='MAE',
                                            batch_size=10,
                                            n_epochs=6,
                                            validation_perc=0,
                                            random_seed=100,
                                            verbose=False)

            projection.train()

        except:

            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__not_allowed_class_init(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data, n_components, optimizers.legacy.Adam(0.001))

        smaller_output_data = np.random.rand(70,4)

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data, n_components, optimizers.legacy.Adam(0.001), projection_independent_outputs=smaller_output_data)

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data, n_components, optimizers.legacy.Adam(0.001), projection_dependent_outputs=smaller_output_data)

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data, n_components, optimizers.legacy.Adam(0.001), projection_dependent_outputs=smaller_output_data, transformed_projection_dependent_outputs='symlog')


        # Mismatch between input_data and projection_dependent_outputs:

        input_data = np.random.rand(100,6)
        output_data = np.random.rand(100,3)
        n_components = 2

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data, n_components, optimizers.legacy.Adam(0.001), projection_dependent_outputs=output_data)

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data, n_components, optimizers.legacy.Adam(0.001), projection_dependent_outputs=output_data, transformed_projection_dependent_outputs='symlog')


# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__architecture_inconsistent_with_custom_activation_functions(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            activation_decoder=('tanh', 'tanh', 'tanh'),
                                            decoder_interior_architecture=(4,))

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            activation_decoder=('tanh',),
                                            decoder_interior_architecture=(4,))

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            activation_decoder=('tanh',),
                                            decoder_interior_architecture=(4,5,6))

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            activation_decoder=('tanh', 'tanh', 'tanh', 'tanh', 'tanh'),
                                            decoder_interior_architecture=(4,5,6))

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            activation_decoder=('tanh', 'sigmoid'),
                                            decoder_interior_architecture=(4,5,6))

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__invalid_string(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            activation_decoder=('tanh', 'name', 'tanh'),
                                            decoder_interior_architecture=(4,5))

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            activation_decoder=('name', 'tanh'),
                                            decoder_interior_architecture=(4,))

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            activation_decoder=('tanh', 'name'),
                                            decoder_interior_architecture=(4,))

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            activation_decoder='name',
                                            decoder_interior_architecture=(4,5))

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            loss='name')

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            'name',
                                            projection_independent_outputs=output_data)

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_dependent_outputs=output_data,
                                            transformed_projection_dependent_outputs='name')

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__wrong_init_type(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection([],
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data)

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            [],
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data)

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=[],
                                            projection_dependent_outputs=output_data)

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            activation_decoder=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            decoder_interior_architecture=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            encoder_weights_init=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            decoder_weights_init=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            hold_initialization=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            hold_weights=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            transformed_projection_dependent_outputs=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            loss=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            'none',
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,)

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            batch_size=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            n_epochs=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            validation_perc=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            random_seed=[])

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            projection_dependent_outputs=output_data,
                                            verbose=[])

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__wrong_encoder_weights_init_shape(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            encoder_weights_init=np.ones((3,2)))

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            encoder_weights_init=np.ones((4,3)))

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            encoder_weights_init=np.ones((10,10)))

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__wrong_decoder_weights_init_shape(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            decoder_weights_init=(np.ones((3,2))),)

        with self.assertRaises(ValueError):
            projection = QoIAwareProjection(input_data,
                                            n_components,
                                            optimizers.legacy.Adam(0.001),
                                            projection_independent_outputs=output_data,
                                            decoder_interior_architecture=(5,6),
                                            decoder_weights_init=(np.ones((2,5)),np.ones((5,6)),np.ones((6,6))),)

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__check_custom_weights_initialization(self):

        input_data = np.random.rand(100,6)
        output_data = np.random.rand(100,4)
        n_components = 2

        projection = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        decoder_interior_architecture=(5,6),
                                        encoder_weights_init=np.ones((6,2)),
                                        decoder_weights_init=(2*np.ones((2,5)),3*np.ones((5,6)),4*np.ones((6,4))),)

        weights_and_biases_init = projection.weights_and_biases_init

        self.assertTrue(np.allclose(weights_and_biases_init[0], np.ones((6,2))))
        self.assertTrue(np.allclose(weights_and_biases_init[2], 2*np.ones((2,5))))
        self.assertTrue(np.allclose(weights_and_biases_init[4], 3*np.ones((5,6))))
        self.assertTrue(np.allclose(weights_and_biases_init[6], 4*np.ones((6,4))))

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__access_attributes(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2

        try:
            projection = QoIAwareProjection(input_data, n_components, optimizers.legacy.Adam(0.001), projection_independent_outputs=output_data)
            X = projection.input_data
            X = projection.n_components
            X = projection.projection_independent_outputs
            X = projection.projection_dependent_outputs
            X = projection.architecture
            X = projection.n_total_outputs
            X = projection.qoi_aware_encoder_decoder
            X = projection.weights_and_biases_init
            X = projection.weights_and_biases_trained
            X = projection.weights_and_biases_best
            X = projection.training_loss
            X = projection.validation_loss
            X = projection.best_epoch_counter
            X = projection.best_training_loss
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__either_projection_dependent_or_independent(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2

        try:
            projection = QoIAwareProjection(input_data, n_components, optimizers.legacy.Adam(0.001), projection_independent_outputs=output_data)
            projection = QoIAwareProjection(input_data, n_components, optimizers.legacy.Adam(0.001), projection_dependent_outputs=output_data)
        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__attributes_not_available_before_training(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2

        projection = QoIAwareProjection(input_data, n_components, optimizers.legacy.Adam(0.001), projection_independent_outputs=output_data)

        X = projection.training_loss
        self.assertTrue(X is None)

        X = projection.validation_loss
        self.assertTrue(X is None)

        X = projection.weights_and_biases_best
        self.assertTrue(X is None)

        X = projection.weights_and_biases_trained
        self.assertTrue(X is None)

        X = projection.best_epoch_counter
        self.assertTrue(X is None)

        X = projection.best_training_loss
        self.assertTrue(X is None)

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__attributes_available_after_training(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2

        projection = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        n_epochs=2,)
        projection.train()


        X = projection.training_loss
        self.assertTrue(X is not None)
        self.assertTrue(isinstance(X, list))

        X = projection.validation_loss
        self.assertTrue(X is not None)
        self.assertTrue(isinstance(X, list))

        X = projection.weights_and_biases_best
        self.assertTrue(X is not None)
        self.assertTrue(isinstance(X, list))

        X = projection.weights_and_biases_trained
        self.assertTrue(X is not None)
        self.assertTrue(isinstance(X, list))

        X = projection.best_epoch_counter
        self.assertTrue(X is not None)
        self.assertTrue(isinstance(X, int))

        X = projection.best_training_loss
        self.assertTrue(X is not None)
        self.assertTrue(isinstance(X, float))

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__number_of_losses_equal_to_n_epochs_plus_one(self):

        input_data = np.random.rand(100,4)
        output_data = np.random.rand(100,4)
        n_components = 2
        n_epochs = 5

        projection = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        n_epochs=n_epochs)
        projection.train()

        X = projection.training_loss
        self.assertTrue(len(X)==n_epochs+1)

        X = projection.validation_loss
        self.assertTrue(len(X)==n_epochs+1)

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__check_architecture(self):

        input_data = np.random.rand(100,7)
        output_data = np.random.rand(100,7)
        n_components = 3

        projection = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data[:,0:5],
                                        projection_dependent_outputs=output_data,
                                        decoder_interior_architecture=())

        self.assertTrue(projection.architecture=='7-3-8')

        projection = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data[:,0:5],
                                        projection_dependent_outputs=output_data,
                                        decoder_interior_architecture=(4,5,6))

        self.assertTrue(projection.architecture=='7-3-4-5-6-8')

        projection = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data[:,0:5],
                                        projection_dependent_outputs=output_data,
                                        transformed_projection_dependent_outputs='symlog',
                                        decoder_interior_architecture=(4,5,6))

        self.assertTrue(projection.architecture=='7-3-4-5-6-11')

        projection = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_dependent_outputs=output_data,
                                        decoder_interior_architecture=(6,))

        self.assertTrue(projection.architecture=='7-3-6-3')

        projection = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_dependent_outputs=output_data,
                                        decoder_interior_architecture=(9,),
                                        transformed_projection_dependent_outputs='symlog')

        self.assertTrue(projection.architecture=='7-3-9-6')

        projection = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data[:,0:5],
                                        decoder_interior_architecture=(11,),
                                        transformed_projection_dependent_outputs='symlog')

        self.assertTrue(projection.architecture=='7-3-11-5')

# ------------------------------------------------------------------------------

    # We cannot test this with the current implementation.

    # def test_utilities__QoIAwareProjection__holding_weights(self):
    #
    #     input_data = np.random.rand(100,7)
    #     output_data = np.random.rand(100,7)
    #     n_components = 2
    #
    #     projection = QoIAwareProjection(input_data,
    #                                     n_components,
    #                                     optimizers.legacy.Adam(0.001),
    #                                     projection_independent_outputs=output_data,
    #                                     projection_dependent_outputs=output_data,
    #                                     hold_initialization=5,
    #                                     n_epochs=10)
    #
    #     projection.train()
    #
    #     bases_across_epochs = projection.bases_across_epochs
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[1]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[2]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[3]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[4]))
    #
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[5]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[6]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[7]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[8]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[9]))
    #
    #     projection = QoIAwareProjection(input_data,
    #                                     n_components,
    #                                     optimizers.legacy.Adam(0.001),
    #                                     projection_independent_outputs=output_data,
    #                                     projection_dependent_outputs=output_data,
    #                                     hold_initialization=5,
    #                                     hold_weights=2,
    #                                     n_epochs=10)
    #
    #     projection.train()
    #
    #     bases_across_epochs = projection.bases_across_epochs
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[1]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[2]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[3]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[4]))
    #
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[5]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[6]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[7]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[8]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[9]))
    #
    #     projection = QoIAwareProjection(input_data,
    #                                     n_components,
    #                                     optimizers.legacy.Adam(0.001),
    #                                     projection_independent_outputs=output_data,
    #                                     projection_dependent_outputs=output_data,
    #                                     hold_initialization=5,
    #                                     hold_weights=2,
    #                                     n_epochs=10)
    #
    #     projection.train()
    #
    #     bases_across_epochs = projection.bases_across_epochs
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[1]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[2]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[3]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[4]))
    #
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[5]))
    #     self.assertTrue(np.allclose(bases_across_epochs[5], bases_across_epochs[6]))
    #
    #     self.assertTrue(not np.allclose(bases_across_epochs[6], bases_across_epochs[7]))
    #     self.assertTrue(np.allclose(bases_across_epochs[7], bases_across_epochs[8]))
    #
    #     self.assertTrue(not np.allclose(bases_across_epochs[8], bases_across_epochs[9]))
    #
    #     projection = QoIAwareProjection(input_data,
    #                                     n_components,
    #                                     optimizers.legacy.Adam(0.001),
    #                                     projection_independent_outputs=output_data,
    #                                     projection_dependent_outputs=output_data,
    #                                     hold_weights=2,
    #                                     n_epochs=10)
    #
    #     projection.train()
    #
    #     bases_across_epochs = projection.bases_across_epochs
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[1]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[1], bases_across_epochs[2]))
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[2], bases_across_epochs[3]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[3], bases_across_epochs[4]))
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[4], bases_across_epochs[5]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[5], bases_across_epochs[6]))
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[6], bases_across_epochs[7]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[7], bases_across_epochs[8]))
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[8], bases_across_epochs[9]))
    #
    #     # Check that holding weights works even if we initialize the encoder and decoder weights:
    #
    #     projection = QoIAwareProjection(input_data,
    #                                     n_components,
    #                                     optimizers.legacy.Adam(0.001),
    #                                     projection_independent_outputs=output_data,
    #                                     projection_dependent_outputs=output_data,
    #                                     encoder_weights_init=np.random.random((7,2)),
    #                                     hold_initialization=5,
    #                                     n_epochs=10)
    #
    #     projection.train()
    #
    #     bases_across_epochs = projection.bases_across_epochs
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[1]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[2]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[3]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[4]))
    #
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[5]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[6]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[7]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[8]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[9]))
    #
    #     projection = QoIAwareProjection(input_data,
    #                                     n_components,
    #                                     optimizers.legacy.Adam(0.001),
    #                                     projection_independent_outputs=output_data,
    #                                     projection_dependent_outputs=output_data,
    #                                     encoder_weights_init=np.random.random((7,2)),
    #                                     hold_initialization=5,
    #                                     hold_weights=2,
    #                                     n_epochs=10)
    #
    #     projection.train()
    #
    #     bases_across_epochs = projection.bases_across_epochs
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[1]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[2]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[3]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[4]))
    #
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[5]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[6]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[7]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[8]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[9]))
    #
    #     projection = QoIAwareProjection(input_data,
    #                                     n_components,
    #                                     optimizers.legacy.Adam(0.001),
    #                                     projection_independent_outputs=output_data,
    #                                     projection_dependent_outputs=output_data,
    #                                     encoder_weights_init=np.random.random((7,2)),
    #                                     hold_initialization=5,
    #                                     hold_weights=2,
    #                                     n_epochs=10)
    #
    #     projection.train()
    #
    #     bases_across_epochs = projection.bases_across_epochs
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[1]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[2]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[3]))
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[4]))
    #
    #     self.assertTrue(not np.allclose(bases_across_epochs[0], bases_across_epochs[5]))
    #     self.assertTrue(np.allclose(bases_across_epochs[5], bases_across_epochs[6]))
    #
    #     self.assertTrue(not np.allclose(bases_across_epochs[6], bases_across_epochs[7]))
    #     self.assertTrue(np.allclose(bases_across_epochs[7], bases_across_epochs[8]))
    #
    #     self.assertTrue(not np.allclose(bases_across_epochs[8], bases_across_epochs[9]))
    #
    #     projection = QoIAwareProjection(input_data,
    #                                     n_components,
    #                                     optimizers.legacy.Adam(0.001),
    #                                     projection_independent_outputs=output_data,
    #                                     projection_dependent_outputs=output_data,
    #                                     encoder_weights_init=np.random.random((7,2)),
    #                                     hold_weights=2,
    #                                     n_epochs=10)
    #
    #     projection.train()
    #
    #     bases_across_epochs = projection.bases_across_epochs
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[0], bases_across_epochs[1]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[1], bases_across_epochs[2]))
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[2], bases_across_epochs[3]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[3], bases_across_epochs[4]))
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[4], bases_across_epochs[5]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[5], bases_across_epochs[6]))
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[6], bases_across_epochs[7]))
    #     self.assertTrue(not np.allclose(bases_across_epochs[7], bases_across_epochs[8]))
    #
    #     self.assertTrue(np.allclose(bases_across_epochs[8], bases_across_epochs[9]))

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__reproducible_network_initialization(self):

        input_data = np.random.rand(100,7)
        output_data = np.random.rand(100,5)
        n_components = 3

        # Initialize two QoIAwareProjection objects with the same random seed:

        projection_1 = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        decoder_interior_architecture=(6,6),
                                        random_seed=100)

        projection_2 = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        decoder_interior_architecture=(6,6),
                                        random_seed=100)

        self.assertTrue(np.allclose(projection_1.weights_and_biases_init[0], projection_2.weights_and_biases_init[0]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_init[1], projection_2.weights_and_biases_init[1]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_init[2], projection_2.weights_and_biases_init[2]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_init[3], projection_2.weights_and_biases_init[3]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_init[4], projection_2.weights_and_biases_init[4]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_init[5], projection_2.weights_and_biases_init[5]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_init[6], projection_2.weights_and_biases_init[6]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_init[7], projection_2.weights_and_biases_init[7]))

        # Initialize two QoIAwareProjection objects with different random seeds:

        projection_1 = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        decoder_interior_architecture=(6,6),
                                        random_seed=100)

        projection_2 = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        decoder_interior_architecture=(6,6),
                                        random_seed=200)

        self.assertTrue(not np.allclose(projection_1.weights_and_biases_init[0], projection_2.weights_and_biases_init[0]))
        self.assertTrue(not np.allclose(projection_1.weights_and_biases_init[2], projection_2.weights_and_biases_init[2]))
        self.assertTrue(not np.allclose(projection_1.weights_and_biases_init[4], projection_2.weights_and_biases_init[4]))
        self.assertTrue(not np.allclose(projection_1.weights_and_biases_init[6], projection_2.weights_and_biases_init[6]))

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__reproducible_network_training(self):

        input_data = np.random.rand(100,7)
        output_data = np.random.rand(100,5)
        n_components = 3

        # Initialize two QoIAwareProjection objects with the same random seed:

        projection_1 = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        decoder_interior_architecture=(6,6),
                                        n_epochs=20,
                                        random_seed=100)

        projection_1.train()

        projection_2 = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        decoder_interior_architecture=(6,6),
                                        n_epochs=20,
                                        random_seed=100)

        projection_2.train()

        self.assertTrue(np.allclose(projection_1.weights_and_biases_trained[0], projection_2.weights_and_biases_trained[0]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_trained[1], projection_2.weights_and_biases_trained[1]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_trained[2], projection_2.weights_and_biases_trained[2]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_trained[3], projection_2.weights_and_biases_trained[3]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_trained[4], projection_2.weights_and_biases_trained[4]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_trained[5], projection_2.weights_and_biases_trained[5]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_trained[6], projection_2.weights_and_biases_trained[6]))
        self.assertTrue(np.allclose(projection_1.weights_and_biases_trained[7], projection_2.weights_and_biases_trained[7]))

        # Initialize two QoIAwareProjection objects with different random seeds:

        projection_1 = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        decoder_interior_architecture=(6,6),
                                        n_epochs=20,
                                        random_seed=100)

        projection_1.train()

        projection_2 = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        decoder_interior_architecture=(6,6),
                                        n_epochs=20,
                                        random_seed=200)

        projection_2.train()

        self.assertTrue(not np.allclose(projection_1.weights_and_biases_trained[0], projection_2.weights_and_biases_trained[0]))
        self.assertTrue(not np.allclose(projection_1.weights_and_biases_trained[1], projection_2.weights_and_biases_trained[1]))
        self.assertTrue(not np.allclose(projection_1.weights_and_biases_trained[2], projection_2.weights_and_biases_trained[2]))
        self.assertTrue(not np.allclose(projection_1.weights_and_biases_trained[3], projection_2.weights_and_biases_trained[3]))
        self.assertTrue(not np.allclose(projection_1.weights_and_biases_trained[4], projection_2.weights_and_biases_trained[4]))
        self.assertTrue(not np.allclose(projection_1.weights_and_biases_trained[5], projection_2.weights_and_biases_trained[5]))
        self.assertTrue(not np.allclose(projection_1.weights_and_biases_trained[6], projection_2.weights_and_biases_trained[6]))
        self.assertTrue(not np.allclose(projection_1.weights_and_biases_trained[7], projection_2.weights_and_biases_trained[7]))

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__trainable_encoder_bias(self):

        input_data = np.random.rand(100,7)
        output_data = np.random.rand(100,5)
        n_components = 3

        # No bias in the encoder:
        projection_1 = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        decoder_interior_architecture=(6,6),
                                        trainable_encoder_bias=False,
                                        n_epochs=20,
                                        random_seed=100)

        self.assertTrue(projection_1.weights_and_biases_init[0].shape==(7,3))
        self.assertTrue(projection_1.weights_and_biases_init[1].shape==(3,6))
        self.assertTrue(projection_1.weights_and_biases_init[2].shape==(6,))
        self.assertTrue(projection_1.weights_and_biases_init[3].shape==(6,6))
        self.assertTrue(projection_1.weights_and_biases_init[4].shape==(6,))
        self.assertTrue(projection_1.weights_and_biases_init[5].shape==(6,5))
        self.assertTrue(projection_1.weights_and_biases_init[6].shape==(5,))

        # With bias in the encoder:
        projection_1 = QoIAwareProjection(input_data,
                                        n_components,
                                        optimizers.legacy.Adam(0.001),
                                        projection_independent_outputs=output_data,
                                        decoder_interior_architecture=(6,6),
                                        trainable_encoder_bias=True,
                                        n_epochs=20,
                                        random_seed=100)

        self.assertTrue(projection_1.weights_and_biases_init[0].shape==(7,3))
        self.assertTrue(projection_1.weights_and_biases_init[1].shape==(3,))
        self.assertTrue(projection_1.weights_and_biases_init[2].shape==(3,6))
        self.assertTrue(projection_1.weights_and_biases_init[3].shape==(6,))
        self.assertTrue(projection_1.weights_and_biases_init[4].shape==(6,6))
        self.assertTrue(projection_1.weights_and_biases_init[5].shape==(6,))
        self.assertTrue(projection_1.weights_and_biases_init[6].shape==(6,5))
        self.assertTrue(projection_1.weights_and_biases_init[7].shape==(5,))

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__optimizers(self):


        # Generate dummy dataset:
        x = np.linspace(0,10,100)[:,None]
        y = np.logspace(1,2,100)[:,None]
        z = np.sqrt(np.linspace(10,100,100)[:,None])
        X = np.hstack((x, y, z))
        S = np.hstack((x**2, x, y**2))
        n_components = 1

        (input_data, centers, scales) = preprocess.center_scale(X, scaling='0to1')

        try:
            projection = QoIAwareProjection(input_data,
                                           n_components,
                                           optimizers.legacy.Adam(0.001),
                                           projection_independent_outputs=input_data[:,0:2],
                                           projection_dependent_outputs=S,
                                           activation_decoder=('tanh', 'tanh', 'tanh'),
                                           decoder_interior_architecture=(5,8),
                                           transformed_projection_dependent_outputs='signed-square-root',
                                           batch_size=100,
                                           n_epochs=10,
                                           random_seed=100)

            projection = QoIAwareProjection(input_data,
                                           n_components,
                                           optimizers.legacy.Nadam(0.001),
                                           projection_independent_outputs=input_data[:,0:2],
                                           projection_dependent_outputs=S,
                                           activation_decoder=('tanh', 'tanh', 'tanh'),
                                           decoder_interior_architecture=(5,8),
                                           transformed_projection_dependent_outputs='signed-square-root',
                                           batch_size=100,
                                           n_epochs=10,
                                           random_seed=100)

            projection = QoIAwareProjection(input_data,
                                           n_components,
                                           optimizers.legacy.RMSprop(0.001),
                                           projection_independent_outputs=input_data[:,0:2],
                                           projection_dependent_outputs=S,
                                           activation_decoder=('tanh', 'tanh', 'tanh'),
                                           decoder_interior_architecture=(5,8),
                                           transformed_projection_dependent_outputs='signed-square-root',
                                           batch_size=100,
                                           n_epochs=10,
                                           random_seed=100)

            projection = QoIAwareProjection(input_data,
                                           n_components,
                                           optimizers.legacy.Adadelta(0.001),
                                           projection_independent_outputs=input_data[:,0:2],
                                           projection_dependent_outputs=S,
                                           activation_decoder=('tanh', 'tanh', 'tanh'),
                                           decoder_interior_architecture=(5,8),
                                           transformed_projection_dependent_outputs='signed-square-root',
                                           batch_size=100,
                                           n_epochs=10,
                                           random_seed=100)

            projection = QoIAwareProjection(input_data,
                                           n_components,
                                           optimizers.legacy.Adagrad(0.001),
                                           projection_independent_outputs=input_data[:,0:2],
                                           projection_dependent_outputs=S,
                                           activation_decoder=('tanh', 'tanh', 'tanh'),
                                           decoder_interior_architecture=(5,8),
                                           transformed_projection_dependent_outputs='signed-square-root',
                                           batch_size=100,
                                           n_epochs=10,
                                           random_seed=100)

            projection = QoIAwareProjection(input_data,
                                           n_components,
                                           optimizers.legacy.Adamax(0.001),
                                           projection_independent_outputs=input_data[:,0:2],
                                           projection_dependent_outputs=S,
                                           activation_decoder=('tanh', 'tanh', 'tanh'),
                                           decoder_interior_architecture=(5,8),
                                           transformed_projection_dependent_outputs='signed-square-root',
                                           batch_size=100,
                                           n_epochs=10,
                                           random_seed=100)

            projection = QoIAwareProjection(input_data,
                                           n_components,
                                           optimizers.legacy.SGD(0.001),
                                           projection_independent_outputs=input_data[:,0:2],
                                           projection_dependent_outputs=S,
                                           activation_decoder=('tanh', 'tanh', 'tanh'),
                                           decoder_interior_architecture=(5,8),
                                           transformed_projection_dependent_outputs='signed-square-root',
                                           batch_size=100,
                                           n_epochs=10,
                                           random_seed=100)

        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__best_trained_network(self):

        # Generate dummy dataset:
        x = np.linspace(0,10,100)[:,None]
        y = np.logspace(1,2,100)[:,None]
        z = np.sqrt(np.linspace(10,100,100)[:,None])
        X = np.hstack((x, y, z))
        S = np.hstack((x**2, x, y**2))
        n_components = 1
        n_epochs = 10

        (input_data, centers, scales) = preprocess.center_scale(X, scaling='0to1')

        try:
            projection = QoIAwareProjection(input_data,
                                           n_components,
                                           optimizers.legacy.Adam(0.001),
                                           projection_independent_outputs=input_data[:,0:2],
                                           projection_dependent_outputs=S,
                                           activation_decoder=('tanh', 'tanh', 'tanh'),
                                           decoder_interior_architecture=(5,8),
                                           transformed_projection_dependent_outputs='signed-square-root',
                                           batch_size=100,
                                           n_epochs=n_epochs,
                                           random_seed=100)

            projection.train()

            best_epoch_counter = projection.best_epoch_counter
            min_loss_idx, = np.where(projection.training_loss==np.min(projection.training_loss))
            self.assertTrue(best_epoch_counter==min_loss_idx[0])
            self.assertTrue(projection.best_training_loss==np.min(projection.training_loss))

            if best_epoch_counter==n_epochs:
                self.assertTrue(np.array_equal(projection.weights_and_biases_trained[0], projection.weights_and_biases_best[0]))

        except:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_utilities__QoIAwareProjection__check_various_initializations_of_weights(self):

        import tensorflow as tf
        from keras.models import Model
        from keras.layers import Input, Lambda, Dense, concatenate
        from PCAfold import center_scale


        random_seed = 200
        X = np.random.rand(100,5)
        S = np.random.rand(100,5)

        tf.random.set_seed(random_seed)
        tf.keras.utils.set_random_seed(random_seed)

        encoder_kernel_initializer = tf.keras.initializers.RandomNormal(seed=random_seed)
        decoder_kernel_initializer = tf.keras.initializers.RandomUniform(seed=random_seed)

        # Create an encoding-decoding model manually
        input_layer = Input(shape=(5,))
        encoder = Dense(2, activation='linear', kernel_initializer=encoder_kernel_initializer)(input_layer)
        decoder_1 = Dense(5, activation='tanh', kernel_initializer=decoder_kernel_initializer)(encoder)
        decoder_2 = Dense(10, activation='tanh', kernel_initializer=decoder_kernel_initializer)(decoder_1)
        decoder_3 = Dense(10, activation='tanh', kernel_initializer=decoder_kernel_initializer)(decoder_2)
        output_layer = Dense(2, activation='tanh', kernel_initializer=decoder_kernel_initializer)(decoder_3)
        manual_model = Model(input_layer, output_layer)

        # Create an encoding-decoding model using PCAfold:
        qoi_aware = QoIAwareProjection(X,
                                       n_components=2,
                                       optimizer=optimizers.legacy.Adam(learning_rate=0.001),
                                       projection_independent_outputs=None,
                                       projection_dependent_outputs=S,
                                       activation_decoder=('tanh', 'tanh', 'tanh', 'tanh'),
                                       decoder_interior_architecture=(5,10,10),
                                       encoder_kernel_initializer=encoder_kernel_initializer,
                                       decoder_kernel_initializer=decoder_kernel_initializer,
                                       random_seed=random_seed)

        for i in range(0,10):
            self.assertTrue(np.array_equal(qoi_aware.weights_and_biases_init[i], manual_model.get_weights()[i]))

        encoder_kernel_initializer = tf.keras.initializers.LecunNormal(seed=random_seed)
        decoder_kernel_initializer = tf.keras.initializers.HeUniform(seed=random_seed)

        # Create an encoding-decoding model manually
        input_layer = Input(shape=(5,))
        encoder = Dense(2, activation='linear', kernel_initializer=encoder_kernel_initializer)(input_layer)
        decoder_1 = Dense(5, activation='tanh', kernel_initializer=decoder_kernel_initializer)(encoder)
        decoder_2 = Dense(10, activation='tanh', kernel_initializer=decoder_kernel_initializer)(decoder_1)
        decoder_3 = Dense(10, activation='tanh', kernel_initializer=decoder_kernel_initializer)(decoder_2)
        output_layer = Dense(2, activation='tanh', kernel_initializer=decoder_kernel_initializer)(decoder_3)
        manual_model = Model(input_layer, output_layer)

        # Create an encoding-decoding model using PCAfold:
        qoi_aware = QoIAwareProjection(X,
                                       n_components=2,
                                       optimizer=optimizers.legacy.Adam(learning_rate=0.001),
                                       projection_independent_outputs=None,
                                       projection_dependent_outputs=S,
                                       activation_decoder=('tanh', 'tanh', 'tanh', 'tanh'),
                                       decoder_interior_architecture=(5,10,10),
                                       encoder_kernel_initializer=encoder_kernel_initializer,
                                       decoder_kernel_initializer=decoder_kernel_initializer,
                                       random_seed=random_seed)

        for i in range(0,10):
            self.assertTrue(np.array_equal(qoi_aware.weights_and_biases_init[i], manual_model.get_weights()[i]))

        encoder_kernel_initializer = tf.keras.initializers.GlorotUniform(seed=random_seed)
        decoder_kernel_initializer = tf.keras.initializers.GlorotUniform(seed=random_seed)

        # Create an encoding-decoding model manually
        input_layer = Input(shape=(5,))
        encoder = Dense(2, activation='linear', kernel_initializer=encoder_kernel_initializer)(input_layer)
        decoder_1 = Dense(5, activation='tanh', kernel_initializer=decoder_kernel_initializer)(encoder)
        decoder_2 = Dense(10, activation='tanh', kernel_initializer=decoder_kernel_initializer)(decoder_1)
        decoder_3 = Dense(10, activation='tanh', kernel_initializer=decoder_kernel_initializer)(decoder_2)
        output_layer = Dense(2, activation='tanh', kernel_initializer=decoder_kernel_initializer)(decoder_3)
        manual_model = Model(input_layer, output_layer)

        # Create an encoding-decoding model using PCAfold:
        qoi_aware = QoIAwareProjection(X,
                                       n_components=2,
                                       optimizer=optimizers.legacy.Adam(learning_rate=0.001),
                                       projection_independent_outputs=None,
                                       projection_dependent_outputs=S,
                                       activation_decoder=('tanh', 'tanh', 'tanh', 'tanh'),
                                       decoder_interior_architecture=(5,10,10),
                                       encoder_kernel_initializer=encoder_kernel_initializer,
                                       decoder_kernel_initializer=decoder_kernel_initializer,
                                       random_seed=random_seed)

        for i in range(0,10):
            self.assertTrue(np.array_equal(qoi_aware.weights_and_biases_init[i], manual_model.get_weights()[i]))
