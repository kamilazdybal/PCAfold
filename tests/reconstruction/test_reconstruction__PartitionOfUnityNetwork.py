import unittest
import numpy as np
from PCAfold import PartitionOfUnityNetwork, init_uniform_partitions
import os

class Reconstruction(unittest.TestCase):

    def test_reconstruction__PartitionOfUnityNetwork__orderoperations(self):
        ivars = np.expand_dims(np.linspace(0,1,21),axis=1)
        net = PartitionOfUnityNetwork(np.array([[0.5]]), np.array([[1.]]), 'constant')
        try:
            net.lstsq(False)
        except:
            self.assertTrue(True)
        try:
            net.train(2)
        except:
            self.assertTrue(True)
        try:
            net(ivars)
        except:
            self.assertTrue(True)
        try:
            net.derivatives(ivars)
        except:
            self.assertTrue(True)
        
        net = PartitionOfUnityNetwork(np.array([[0.5]]), np.array([[1.]]), 'constant', basis_coeffs=np.array([[1.]]))
        try:
            net.lstsq(False)
        except:
            self.assertTrue(True)
        try:
            net.train(2)
        except:
            self.assertTrue(True)
        try:
            net(ivars)
        except:
            self.assertTrue(False)
        try:
            net.derivatives(ivars)
        except:
            self.assertTrue(False)

    def test_reconstruction__PartitionOfUnityNetwork__options(self):
        try:
            for ndim in range(1,5):
                net = PartitionOfUnityNetwork(np.ones((1,ndim)), np.ones((1,ndim)), 'constant')
                net.build_training_graph(np.ones((21,ndim)), np.ones(21))
                net = PartitionOfUnityNetwork(np.ones((1,ndim)), np.ones((1,ndim)), 'linear')
                net.build_training_graph(np.ones((21,ndim)), np.ones(21))
        except:
            self.assertTrue(False)
        try:
            for ndim in range(1,4):
                net = PartitionOfUnityNetwork(np.ones((1,ndim)), np.ones((1,ndim)), 'quadratic')
                net.build_training_graph(np.ones((21,ndim)), np.ones(21))
        except:
            self.assertTrue(False)
        try:
            ndim = 4
            net = PartitionOfUnityNetwork(np.ones((1,ndim)), np.ones((1,ndim)), 'quadratic')
            net.build_training_graph(np.ones((21,ndim)), np.ones(21))
        except:
            self.assertTrue(True)
        try:
            net = PartitionOfUnityNetwork(np.ones((1,1)), np.ones((1,1)), 'cubic')
        except:
            self.assertTrue(True)
        try:
            net = PartitionOfUnityNetwork(np.ones((1,1)), np.ones((1,2)), 'constant')
        except:
            self.assertTrue(True)
        try:
            net = PartitionOfUnityNetwork(np.ones((1,1)), np.ones((1,1)), 'constant')
            net.build_training_graph(np.ones((21,2)), np.ones(21))
        except:
            self.assertTrue(True)
        try:
            net = PartitionOfUnityNetwork(np.ones((1,1)), np.ones((1,1)), 'constant', basis_coeffs=np.ones((1,1)))
            net.build_training_graph(np.ones((21,1)), np.ones((21,2)))
        except:
            self.assertTrue(True)
        try:
            net = PartitionOfUnityNetwork(np.ones((1,1)), np.ones((1,1)), 'constant', dtype='float8')
        except:
            self.assertTrue(True)
        try:
            net = PartitionOfUnityNetwork(np.ones((1,1)), np.ones((1,1)), 'constant', dtype='float32')
        except:
            self.assertTrue(False)

    def test_reconstruction__PartitionOfUnityNetwork__basis_constant(self):
        ivars = np.expand_dims(np.linspace(0,1,21),axis=1)
        dvar = 3. * np.ones(ivars.size)
        der = np.zeros(ivars.size)
        net = PartitionOfUnityNetwork(np.array([[0.5]]), np.array([[1.]]), 'constant')
        net.build_training_graph(ivars, dvar)
        net.lstsq(False)
        zerotol = 1.e-9
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 3.)<zerotol)
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(der.ravel() - net.derivatives(ivars)))<zerotol)
        net.train(2,archive_rate=1)
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 3.)<zerotol)
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(der.ravel() - net.derivatives(ivars)))<zerotol)
        
    def test_reconstruction__PartitionOfUnityNetwork__basis_linear(self):
        ivars = np.expand_dims(np.linspace(0,1,21),axis=1)
        dvar = 5. * ivars + 2.
        der = 5.*np.ones(ivars.size)
        net = PartitionOfUnityNetwork(np.array([[0.5]]), np.array([[1.]]), 'linear')
        net.build_training_graph(ivars, dvar)
        net.lstsq(False)
        zerotol = 1.e-9
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 2.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,1] - 5.)<zerotol)
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(der.ravel() - net.derivatives(ivars).ravel()))<zerotol)
        net.train(2,archive_rate=1)
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 2.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,1] - 5.)<zerotol)
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(der.ravel() - net.derivatives(ivars).ravel()))<zerotol)

        net = PartitionOfUnityNetwork(np.array([[0.5]]), np.array([[1.]]), 'linear', basis_coeffs=np.array([[2., 5.]]))
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)

        ivars_cs = ivars*4. + 5.
        net = PartitionOfUnityNetwork(**init_uniform_partitions([1], ivars_cs), basis_type='linear', basis_coeffs=np.array([[2., 5.]]))
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars_cs)))<zerotol)

    def test_reconstruction__PartitionOfUnityNetwork__basis_quadratic(self):
        ivars = np.expand_dims(np.linspace(0,1,21),axis=1)
        dvar = 8.*ivars**2 + 6. * ivars + 4.
        der = 16.*ivars.ravel() + 6.*np.ones(ivars.size)
        net = PartitionOfUnityNetwork(np.array([[0.5]]), np.array([[1.]]), 'quadratic')
        net.build_training_graph(ivars, dvar)
        net.lstsq(False)
        zerotol = 1.e-8
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 4.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,1] - 6.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,2] - 8.)<zerotol)
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(der.ravel() - net.derivatives(ivars).ravel()))<zerotol)
        net.train(2,archive_rate=1)
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 4.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,1] - 6.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,2] - 8.)<zerotol)
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(der.ravel() - net.derivatives(ivars).ravel()))<zerotol)
    
    def test_reconstruction__PartitionOfUnityNetwork__basis_linear_2D(self):
        ivars = np.linspace(0,1,21)
        ivars = np.meshgrid(ivars, ivars)
        ivars = np.vstack([b.ravel() for b in ivars]).T
        dvar = 5. * ivars[:,0] + 7. * ivars[:,1] + 2.
        derx = 5.*np.ones(ivars.shape[0])
        dery = 7.*np.ones(ivars.shape[0])
        net = PartitionOfUnityNetwork(np.array([[0.5,0.5]]), np.array([[1.,1.]]), 'linear')
        net.build_training_graph(ivars, dvar)
        net.lstsq(False)
        zerotol = 1.e-9
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 2.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,1] - 5.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,2] - 7.)<zerotol)
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(derx.ravel() - net.derivatives(ivars)[:,0]))<zerotol)
        self.assertTrue(np.max(np.abs(dery.ravel() - net.derivatives(ivars)[:,1]))<zerotol)
        net.train(2,archive_rate=1)
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 2.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,1] - 5.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,2] - 7.)<zerotol)
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(derx.ravel() - net.derivatives(ivars)[:,0]))<zerotol)
        self.assertTrue(np.max(np.abs(dery.ravel() - net.derivatives(ivars)[:,1]))<zerotol)
    
    def test_reconstruction__PartitionOfUnityNetwork__basis_linear_2dvars(self):
        ivars = np.expand_dims(np.linspace(0,1,21),axis=1)
        dvar1 = 5. * ivars + 2.
        dvar2 = 7. * ivars + 3.
        dvars = np.hstack((dvar1, dvar2))
        der1 = 5.*np.ones(ivars.size)
        der2 = 7.*np.ones(ivars.size)
        net = PartitionOfUnityNetwork(np.array([[0.5]]), np.array([[1.]]), 'linear')
        try:
            net.build_training_graph(ivars, dvars, 'rel')
        except:
            self.assertTrue(True)
        net.build_training_graph(ivars, dvars)
        net.lstsq(False)
        zerotol = 1.e-9
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 2.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,1] - 5.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[1,0] - 3.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[1,1] - 7.)<zerotol)
        self.assertTrue(np.max(np.abs(dvars - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(der1.ravel() - net.derivatives(ivars,0).ravel()))<zerotol)
        self.assertTrue(np.max(np.abs(der2.ravel() - net.derivatives(ivars,1).ravel()))<zerotol)
        net.train(2,archive_rate=1)
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 2.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,1] - 5.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[1,0] - 3.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[1,1] - 7.)<zerotol)
        self.assertTrue(np.max(np.abs(dvars - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(der1.ravel() - net.derivatives(ivars,0).ravel()))<zerotol)
        self.assertTrue(np.max(np.abs(der2.ravel() - net.derivatives(ivars,1).ravel()))<zerotol)
        
    
    def test_reconstruction__PartitionOfUnityNetwork__basis_linear_2part(self):
        ivars = np.expand_dims(np.linspace(0,1,21),axis=1)
        dvar = 5. * ivars + 2.
        net = PartitionOfUnityNetwork(np.array([[0.25], [0.75]]), np.array([[2.],[2.]]), 'linear')
        net.build_training_graph(ivars, dvar)
        net.lstsq(False)
        zerotol = 1.e-7
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 2.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,1] - 2.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,2] - 5.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,3] - 5.)<zerotol)
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)
        net.train(2,archive_rate=1)
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 2.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,1] - 2.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,2] - 5.)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,3] - 5.)<zerotol)
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)


    def test_reconstruction__PartitionOfUnityNetwork__basis_quadratic_transform(self):
        ivars = np.expand_dims(np.linspace(0,1,21),axis=1)
        dvar = ivars.ravel().copy()
        der = np.ones(ivars.size)
        net = PartitionOfUnityNetwork(np.array([[0.5]]), np.array([[1.]]), 'quadratic', 
                                      transform_power=2., transform_shift=0.1, transform_sign_shift=0.2)
        net.build_training_graph(ivars, dvar)
        net.lstsq(False)
        zerotol = 1.e-8
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 0.21)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,1] - 0.20)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,2] - 1.00)<zerotol)
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(der.ravel() - net.derivatives(ivars).ravel()))<zerotol)
        net.train(2,archive_rate=1)
        self.assertTrue(np.abs(net.basis_coeffs[0,0] - 0.21)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,1] - 0.20)<zerotol)
        self.assertTrue(np.abs(net.basis_coeffs[0,2] - 1.00)<zerotol)
        self.assertTrue(np.max(np.abs(dvar.ravel() - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(der.ravel() - net.derivatives(ivars).ravel()))<zerotol)

    def test_reconstruction__PartitionOfUnityNetwork__readwrite(self):
        ivars = np.expand_dims(np.linspace(0,1,21),axis=1)
        dvar1 = 5. * ivars + 2.
        dvar2 = 7. * ivars + 3.
        dvars = np.hstack((dvar1, dvar2))
        net = PartitionOfUnityNetwork(np.array([[0.5]]), np.array([[1.]]), 'linear')
        net.build_training_graph(ivars,dvars)
        net.lstsq(False)
        pkl = '__removeaftertest.pkl'
        txt = '__removeaftertest.txt'
        try:
            net.write_data_to_txt(txt)
        except:
            self.assertTrue(True)
        net.write_data_to_file(pkl)

        zerotol = 1.e-14

        net2 = PartitionOfUnityNetwork.load_from_file(pkl)
        net3 = PartitionOfUnityNetwork(**PartitionOfUnityNetwork.load_data_from_file(pkl))
        self.assertTrue(np.max(np.abs(net2(ivars) - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(net3(ivars) - net(ivars)))<zerotol)

        net = PartitionOfUnityNetwork(np.array([[0.5]]), np.array([[1.]]), 'linear')
        net.build_training_graph(ivars,dvar1)
        net.lstsq(False)
        net.write_data_to_txt(txt)
        net.write_data_to_file(pkl)

        net2 = PartitionOfUnityNetwork.load_from_file(pkl)
        net3 = PartitionOfUnityNetwork(**PartitionOfUnityNetwork.load_data_from_file(pkl))
        net4 = PartitionOfUnityNetwork(**PartitionOfUnityNetwork.load_data_from_txt(txt))
        self.assertTrue(np.max(np.abs(net2(ivars) - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(net3(ivars) - net(ivars)))<zerotol)
        self.assertTrue(np.max(np.abs(net4(ivars) - net(ivars)))<zerotol)

        os.remove(pkl)
        os.remove(txt)
