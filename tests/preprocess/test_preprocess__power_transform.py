import unittest
import numpy as np
from PCAfold import preprocess

class Preprocess(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Preprocess, self).__init__(*args, **kwargs)
        self.X = np.random.rand(100,20) + 1
        self.zerotol = 1.e-15

    def test_preprocess__power_transform__sqrt_transform(self):
        Xt = preprocess.power_transform(self.X, 0.5)
        self.assertTrue(np.max(np.abs(Xt - np.sqrt(self.X)))<self.zerotol)
        Xo = preprocess.power_transform(Xt, 0.5, invert=True)
        self.assertTrue(np.max(np.abs(Xo - self.X))<self.zerotol)
        self.assertTrue(np.max(np.abs(Xo - Xt*Xt))<self.zerotol)


    def test_preprocess__power_transform__no_transform(self):
        self.assertTrue(np.max(np.abs(preprocess.power_transform(self.X, 1.) - self.X))<self.zerotol)

    
    def test_preprocess__power_transform__shift_transform(self):
        self.assertTrue(np.max(np.abs(preprocess.power_transform(self.X, 1., transform_shift=0.2) - self.X - 0.2))<self.zerotol)

    def test_preprocess__power_transform__signshift_transform(self):
        self.assertTrue(np.max(np.abs(preprocess.power_transform(self.X, 1., transform_shift=0.2, transform_sign_shift=0.3) - self.X - 0.5))<self.zerotol)

    def test_preprocess__power_transform__full_transform(self):
        Xt = preprocess.power_transform(self.X, 2., transform_shift=0.2, transform_sign_shift=0.3)
        Xt_manual = (self.X + 0.2)*(self.X + 0.2) + 0.3
        self.assertTrue(np.max(np.abs(Xt - Xt_manual))<self.zerotol)

        Xo = self.X - 1.5
        Xt = preprocess.power_transform(Xo, 2., transform_shift=0.2, transform_sign_shift=0.3)
        Xt_manual_wrong = (Xo + 0.2)*(Xo + 0.2) + 0.3
        self.assertTrue(np.max(np.abs(Xt - Xt_manual_wrong))>self.zerotol)
        Xt_manual = (Xo + 0.2)*(Xo + 0.2) * np.sign(Xo+0.2) + 0.3 * np.sign(Xo+0.2)
        self.assertTrue(np.max(np.abs(Xt - Xt_manual))<self.zerotol)

