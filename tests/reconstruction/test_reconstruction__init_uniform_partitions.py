import unittest
import numpy as np
from PCAfold import init_uniform_partitions


class Reconstruction(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Reconstruction, self).__init__(*args, **kwargs)
        ivar1 = np.linspace(0,1,20)
        self._ivar1 = ivar1[np.argwhere((ivar1<0.4)|(ivar1>0.6))[:,0]]
        self._npx = 5
        self._npy = 2
        self._npz = 3

    def test_reconstruction__init_uniform_partitions__outputvars(self):
        ivars = np.array([self._ivar1]).T
        init_data = init_uniform_partitions([5], ivars, verbose=False)
        vars = list(init_data.keys())
        for k in ['partition_centers', 'partition_shapes', 'ivar_center', 'ivar_scale']:
            self.assertTrue(k in vars)

    def test_reconstruction__init_uniform_partitions__1D(self):
        ivars = np.array([self._ivar1]).T
        init_data = init_uniform_partitions([self._npx], ivars, verbose=False)
        self.assertTrue(init_data['partition_centers'].shape==(4,1))
        self.assertTrue(init_data['ivar_center']==np.array([[0.]]))
        self.assertTrue(init_data['ivar_scale']==np.array([[1.]]))
        centers = np.ones_like(init_data['partition_centers'])
        centers[:,0] = np.array([0.1,0.3,0.7,0.9])
        shapes = 10. * np.ones_like(centers)
        zerotol = 1.e-15
        self.assertTrue(np.max(np.abs(init_data['partition_centers']-centers))<zerotol)
        self.assertTrue(np.max(np.abs(init_data['partition_shapes']-shapes))<zerotol)

        center = 0.5
        scale = 2.
        ivars_cs = ivars * scale + center
        init_data_cs = init_uniform_partitions([self._npx], ivars_cs, verbose=False)
        self.assertTrue(init_data_cs['ivar_center']==np.array([[center]]))
        self.assertTrue(init_data_cs['ivar_scale']==np.array([[scale]]))
        self.assertTrue(np.max(np.abs(init_data_cs['partition_centers']-centers))<zerotol)
        self.assertTrue(np.max(np.abs(init_data_cs['partition_shapes']-shapes))<zerotol)

    def test_reconstruction__init_uniform_partitions__2D(self):
        ivars = np.meshgrid(self._ivar1, self._ivar1)
        ivars = np.vstack([b.ravel() for b in ivars]).T
        init_data = init_uniform_partitions([self._npx, self._npy], ivars, verbose=False)
        self.assertTrue(init_data['partition_centers'].shape==(8,2))

        centers = np.ones_like(init_data['partition_centers'])
        centers1 = np.array([0.1,0.3,0.7,0.9])
        centers[:4,0] = centers1
        centers[4:,0] = centers1
        centers[:4,1] = 0.25
        centers[4:,1] = 0.75
        shapes = 10. * np.ones_like(centers)
        shapes[:,1] = 4.
        zerotol = 1.e-15
        self.assertTrue(np.max(np.abs(init_data['partition_centers']-centers))<zerotol)
        self.assertTrue(np.max(np.abs(init_data['partition_shapes']-shapes))<zerotol)

        c1 = 0.5
        c2 = 0.2
        s1 = 2.
        s2 = 3.
        ivars_cs = ivars.copy()
        ivars_cs[:,0] = ivars_cs[:,0] * s1 + c1
        ivars_cs[:,1] = ivars_cs[:,1] * s2 + c2
        init_data_cs = init_uniform_partitions([self._npx, self._npy], ivars_cs, verbose=False)
        self.assertTrue(init_data_cs['ivar_center'][0,0]==c1)
        self.assertTrue(init_data_cs['ivar_center'][0,1]==c2)
        self.assertTrue(init_data_cs['ivar_scale'][0,0]==s1)
        self.assertTrue(init_data_cs['ivar_scale'][0,1]==s2)
        self.assertTrue(np.max(np.abs(init_data_cs['partition_centers']-centers))<zerotol)
        self.assertTrue(np.max(np.abs(init_data_cs['partition_shapes']-shapes))<zerotol)

    def test_reconstruction__init_uniform_partitions__3D(self):
        ivars = np.meshgrid(self._ivar1, self._ivar1, self._ivar1)
        ivars = np.vstack([b.ravel() for b in ivars]).T
        init_data = init_uniform_partitions([self._npx, self._npy, self._npz], ivars, verbose=False)
        self.assertTrue(init_data['partition_centers'].shape==(24,3))
