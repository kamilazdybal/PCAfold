import unittest
import numpy as np
from PCAfold import preprocess
from PCAfold import reduction
from PCAfold import reconstruction

class Reconstruction(unittest.TestCase):

    def test_reconstruction__ANN__allowed_class_init(self):

        input_data = np.random.rand(100,8)
        output_data = np.random.rand(100,3)

        try:
            ann_model = reconstruction.ANN(input_data,
                            output_data)
        except Exception:
            self.assertTrue(False)

        try:
            ann_model = reconstruction.ANN(input_data,
                            output_data,
                            interior_architecture=(),
                            activation_functions='tanh')
        except Exception:
            self.assertTrue(False)

        try:
            ann_model = reconstruction.ANN(input_data,
                            output_data,
                            interior_architecture=(10,),
                            activation_functions='tanh')
        except Exception:
            self.assertTrue(False)

        try:
            ann_model = reconstruction.ANN(input_data,
                            output_data,
                            interior_architecture=(5,4),
                            activation_functions='tanh')
        except Exception:
            self.assertTrue(False)

        try:
            ann_model = reconstruction.ANN(input_data,
                            output_data,
                            interior_architecture=(5,4),
                            activation_functions=('tanh', 'tanh', 'linear'),
                            weights_init='glorot_uniform',
                            biases_init='zeros',
                            loss='MSE',
                            optimizer='Adam',
                            batch_size=100,
                            n_epochs=1000,
                            learning_rate=0.001,
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
            ann_model = reconstruction.ANN([], output_data)

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, [])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, smaller_output_data)

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, interior_architecture=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, activation_functions=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, weights_init=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, biases_init=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, loss=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, optimizer=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, batch_size=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, n_epochs=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, learning_rate=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, validation_perc=[])

        with self.assertRaises(ValueError):
            ann_model = reconstruction.ANN(input_data, output_data, random_seed=[])

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__access_attributes(self):

        input_data = np.random.rand(100,8)
        output_data = np.random.rand(100,3)

        try:
            ann_model = reconstruction.ANN(input_data, output_data)
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

        ann_model = reconstruction.ANN(input_data, output_data)

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

        ann_model = reconstruction.ANN(input_data, output_data, n_epochs=5)

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

        ann_model = reconstruction.ANN(input_data, output_data, n_epochs=n_epochs)

        ann_model.train()

        X = ann_model.training_loss
        self.assertTrue(len(X)==n_epochs)

        X = ann_model.validation_loss
        self.assertTrue(len(X)==n_epochs)

# ------------------------------------------------------------------------------

    def test_reconstruction__ANN__check_architecture(self):

        input_data = np.random.rand(100,7)
        output_data = np.random.rand(100,5)

        ann_model = reconstruction.ANN(input_data, output_data, interior_architecture=())

        self.assertTrue(ann_model.architecture=='7-5')

        ann_model = reconstruction.ANN(input_data, output_data, interior_architecture=(4,5,6))

        self.assertTrue(ann_model.architecture=='7-4-5-6-5')

        ann_model = reconstruction.ANN(input_data, output_data, interior_architecture=(6,))

        self.assertTrue(ann_model.architecture=='7-6-5')

        ann_model = reconstruction.ANN(input_data, output_data, interior_architecture=(9,))

        self.assertTrue(ann_model.architecture=='7-9-5')

        ann_model = reconstruction.ANN(input_data, output_data, interior_architecture=(11,))

        self.assertTrue(ann_model.architecture=='7-11-5')

# ------------------------------------------------------------------------------
