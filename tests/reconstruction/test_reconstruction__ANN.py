import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import reconstruction
from tensorflow import optimizers

class Reconstruction(unittest.TestCase):

    def test_reconstruction__ANN__allowed_class_init(self):

        input_data = np.random.rand(100,8)
        output_data = np.random.rand(100,3)

        try:
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001))
        except Exception:
            self.assertTrue(False)

        try:
            ann_model = reconstruction.ANN(input_data,
                            output_data,
                            optimizers.legacy.Adam(0.001),
                            interior_architecture=(),
                            activation_functions='tanh')
        except Exception:
            self.assertTrue(False)

        try:
            ann_model = reconstruction.ANN(input_data,
                            output_data,
                            optimizers.legacy.Adam(0.001),
                            interior_architecture=(10,),
                            activation_functions='tanh')
        except Exception:
            self.assertTrue(False)

        try:
            ann_model = reconstruction.ANN(input_data,
                            output_data,
                            optimizers.legacy.Adam(0.001),
                            interior_architecture=(5,4),
                            activation_functions='tanh')
        except Exception:
            self.assertTrue(False)

        try:
            ann_model = reconstruction.ANN(input_data,
                            output_data,
                            optimizers.legacy.Adam(0.001),
                            interior_architecture=(5,4),
                            activation_functions=('tanh', 'tanh', 'linear'),
                            kernel_initializer='glorot_uniform',
                            bias_initializer='zeros',
                            loss='MSE',
                            batch_size=100,
                            n_epochs=1000,
                            validation_perc=10,
                            random_seed=100)
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__not_allowed_class_init(self):

        input_data = np.random.rand(100,8)
        output_data = np.random.rand(100,3)
        smaller_output_data = np.random.rand(70,3)

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN([], output_data, optimizers.legacy.Adam(0.001))

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, [], optimizers.legacy.Adam(0.001))

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, smaller_output_data, optimizers.legacy.Adam(0.001))

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), activation_functions=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), kernel_initializer=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), bias_initializer=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), loss=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, 'none')

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), batch_size=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), n_epochs=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), validation_perc=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), random_seed=[])

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__access_attributes(self):

        input_data = np.random.rand(100,8)
        output_data = np.random.rand(100,3)

        try:
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001))
            X = ann_model.input_data
            X = ann_model.output_data
            X = ann_model.architecture
            X = ann_model.ann_model
            X = ann_model.weights_and_biases_init
            X = ann_model.weights_and_biases_trained
            X = ann_model.training_loss
            X = ann_model.validation_loss
        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__attributes_not_available_before_training(self):

        input_data = np.random.rand(100,6)
        output_data = np.random.rand(100,4)

        ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001))

        X = ann_model.weights_and_biases_trained
        self.assertTrue(X is None)

        X = ann_model.training_loss
        self.assertTrue(X is None)

        X = ann_model.validation_loss
        self.assertTrue(X is None)

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__attributes_available_after_training(self):

        input_data = np.random.rand(100,6)
        output_data = np.random.rand(100,4)

        ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), n_epochs=5)

        ann_model.train()

        X = ann_model.weights_and_biases_trained
        self.assertTrue(X is not None)
        self.assertTrue(isinstance(X, list))

        X = ann_model.training_loss
        self.assertTrue(X is not None)
        self.assertTrue(isinstance(X, list))

        X = ann_model.validation_loss
        self.assertTrue(X is not None)
        self.assertTrue(isinstance(X, list))

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__number_of_losses_equal_to_n_epochs(self):

        input_data = np.random.rand(100,6)
        output_data = np.random.rand(100,4)
        n_epochs = 5

        ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), n_epochs=n_epochs)

        ann_model.train()

        X = ann_model.training_loss
        self.assertTrue(len(X)==n_epochs)

        X = ann_model.validation_loss
        self.assertTrue(len(X)==n_epochs)

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__check_architecture(self):

        input_data = np.random.rand(100,7)
        output_data = np.random.rand(100,5)

        ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=())

        self.assertTrue(ann_model.architecture=='7-5')

        ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(4,5,6))

        self.assertTrue(ann_model.architecture=='7-4-5-6-5')

        ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(6,))

        self.assertTrue(ann_model.architecture=='7-6-5')

        ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(9,))

        self.assertTrue(ann_model.architecture=='7-9-5')

        ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(11,))

        self.assertTrue(ann_model.architecture=='7-11-5')

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__reproducible_network_initialization(self):

        input_data = np.random.rand(100,7)
        output_data = np.random.rand(100,5)

        # Initialize two ANN objects with the same random seed:

        network_1 = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(8,8,8), random_seed=100)

        network_2 = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(8,8,8), random_seed=100)

        self.assertTrue(np.allclose(network_1.weights_and_biases_init[0], network_2.weights_and_biases_init[0]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_init[1], network_2.weights_and_biases_init[1]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_init[2], network_2.weights_and_biases_init[2]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_init[3], network_2.weights_and_biases_init[3]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_init[4], network_2.weights_and_biases_init[4]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_init[5], network_2.weights_and_biases_init[5]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_init[6], network_2.weights_and_biases_init[6]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_init[7], network_2.weights_and_biases_init[7]))

        # Initialize two ANN objects with different random seeds:

        network_1 = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(8,8,8), random_seed=100)

        network_2 = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(8,8,8), random_seed=200)

        self.assertTrue(not np.allclose(network_1.weights_and_biases_init[0], network_2.weights_and_biases_init[0]))
        self.assertTrue(not np.allclose(network_1.weights_and_biases_init[2], network_2.weights_and_biases_init[2]))
        self.assertTrue(not np.allclose(network_1.weights_and_biases_init[4], network_2.weights_and_biases_init[4]))
        self.assertTrue(not np.allclose(network_1.weights_and_biases_init[6], network_2.weights_and_biases_init[6]))

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__reproducible_network_training(self):

        input_data = np.random.rand(100,7)
        output_data = np.random.rand(100,5)

        # Initialize two ANN objects with the same random seed:

        network_1 = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(8,8,8), n_epochs=10, random_seed=100)

        network_1.train()

        network_2 = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(8,8,8), n_epochs=10, random_seed=100)

        network_2.train()

        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[0], network_2.weights_and_biases_trained[0]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[1], network_2.weights_and_biases_trained[1]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[2], network_2.weights_and_biases_trained[2]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[3], network_2.weights_and_biases_trained[3]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[4], network_2.weights_and_biases_trained[4]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[5], network_2.weights_and_biases_trained[5]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[6], network_2.weights_and_biases_trained[6]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[7], network_2.weights_and_biases_trained[7]))

        # Initialize two ANN objects with different random seeds:

        network_1 = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(8,8,8,8), n_epochs=10, random_seed=100)

        network_1.train()

        network_2 = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(8,8,8,8), n_epochs=10, random_seed=200)

        network_2.train()

        self.assertTrue(not np.allclose(network_1.weights_and_biases_trained[0], network_2.weights_and_biases_trained[0]))
        self.assertTrue(not np.allclose(network_1.weights_and_biases_trained[1], network_2.weights_and_biases_trained[1]))
        self.assertTrue(not np.allclose(network_1.weights_and_biases_trained[2], network_2.weights_and_biases_trained[2]))
        self.assertTrue(not np.allclose(network_1.weights_and_biases_trained[3], network_2.weights_and_biases_trained[3]))
        self.assertTrue(not np.allclose(network_1.weights_and_biases_trained[4], network_2.weights_and_biases_trained[4]))
        self.assertTrue(not np.allclose(network_1.weights_and_biases_trained[5], network_2.weights_and_biases_trained[5]))
        self.assertTrue(not np.allclose(network_1.weights_and_biases_trained[6], network_2.weights_and_biases_trained[6]))
        self.assertTrue(not np.allclose(network_1.weights_and_biases_trained[7], network_2.weights_and_biases_trained[7]))

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__reproducible_network_training_using_mini_batches(self):

        input_data = np.random.rand(100,7)
        output_data = np.random.rand(100,5)

        # Initialize two ANN objects with the same random seed:

        network_1 = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(8,8,8), batch_size=20, n_epochs=10, random_seed=100)

        network_1.train()

        network_2 = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001), interior_architecture=(8,8,8), batch_size=20, n_epochs=10, random_seed=100)

        network_2.train()

        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[0], network_2.weights_and_biases_trained[0]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[1], network_2.weights_and_biases_trained[1]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[2], network_2.weights_and_biases_trained[2]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[3], network_2.weights_and_biases_trained[3]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[4], network_2.weights_and_biases_trained[4]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[5], network_2.weights_and_biases_trained[5]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[6], network_2.weights_and_biases_trained[6]))
        self.assertTrue(np.allclose(network_1.weights_and_biases_trained[7], network_2.weights_and_biases_trained[7]))

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__optimizers(self):

        input_data = np.random.rand(100,8)
        output_data = np.random.rand(100,3)

        try:

            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adam(0.001))
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Nadam(0.001))
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.RMSprop(0.001))
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.SGD(0.001))
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adamax(0.001))
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adadelta(0.001))
            ann_model = reconstruction.ANN(input_data, output_data, optimizers.legacy.Adagrad(0.001))

        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__kernel_initializers(self):

        from tensorflow.keras import initializers

        input_data = np.random.rand(100,6)
        output_data = np.random.rand(100,4)

        try:
            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            kernel_initializer=initializers.RandomUniform(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            kernel_initializer=initializers.RandomNormal(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            kernel_initializer=initializers.LecunUniform(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            kernel_initializer=initializers.LecunNormal(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            kernel_initializer=initializers.GlorotUniform(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            kernel_initializer=initializers.GlorotNormal(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            kernel_initializer=initializers.HeUniform(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            kernel_initializer=initializers.HeNormal(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            kernel_initializer=initializers.Ones())

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            kernel_initializer=initializers.Zeros())

        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__bias_initializers(self):

        from tensorflow.keras import initializers

        input_data = np.random.rand(100,6)
        output_data = np.random.rand(100,4)

        try:

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            bias_initializer=initializers.RandomUniform(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            bias_initializer=initializers.RandomNormal(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            bias_initializer=initializers.LecunUniform(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            bias_initializer=initializers.LecunNormal(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            bias_initializer=initializers.GlorotUniform(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            bias_initializer=initializers.GlorotNormal(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            bias_initializer=initializers.HeUniform(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            bias_initializer=initializers.HeNormal(seed=100))

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            bias_initializer=initializers.Ones())

            ann_model = reconstruction.ANN(input_data,
                                            output_data,
                                            optimizers.legacy.Adam(0.001),
                                            n_epochs=5,
                                            bias_initializer=initializers.Zeros())

        except Exception:
            self.assertTrue(False)

# ------------------------------------------------------------------------------
