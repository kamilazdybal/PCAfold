import unittest
import numpy as np
from PCAfold import QoIAwareProjectionPOUnet, init_uniform_partitions
import os
import tensorflow.compat.v1 as tf


class Utilities(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Utilities, self).__init__(*args, **kwargs)
        X,Y = np.meshgrid(np.linspace(0,1,21), np.linspace(0,1,11))
        self._ivar_orig = np.vstack((X.ravel(), Y.ravel())).T
        theta = 0.25*np.pi
        self._rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        self._ivar_rot = self._ivar_orig.dot(self._rot)

        theta = -theta
        self._unrot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        self._ivar_unrot = self._ivar_rot.dot(self._unrot)

    def test_utilities__QoIAwareProjectionPOUnet__orderoperations(self):
        ivars = np.expand_dims(np.linspace(0,1,21),axis=1)
        net = QoIAwareProjectionPOUnet(np.ones((1,1)), np.array([[0.5]]), np.array([[1.]]), 'constant')
        try:
            net.train(2)
        except:
            self.assertTrue(True)
        try:
            net(ivars)
        except:
            self.assertTrue(True)

        net = QoIAwareProjectionPOUnet(np.ones((1,1)), np.array([[0.5]]), np.array([[1.]]), 'constant', basis_coeffs=np.array([[1.]]))
        try:
            net.train(2)
        except:
            self.assertTrue(True)
        try:
            net(ivars)
        except:
            self.assertTrue(False)

        try:
            net = QoIAwareProjectionPOUnet(np.ones((2,2)), np.array([[0.5]]), np.array([[1.]]), 'constant')
        except:
            self.assertTrue(True)
        try:
            net = QoIAwareProjectionPOUnet(np.ones((1,1)), np.array([[0.5]]), np.array([[1.]]), 'constant', projection_biases=np.ones(2))
        except:
            self.assertTrue(True)

    def test_utilities__QoIAwareProjectionPOUnet__projection(self):
        ### only rotation
        net = QoIAwareProjectionPOUnet(self._unrot, **init_uniform_partitions([5,7], self._ivar_unrot), basis_type='linear')
        proj = net.projection(self._ivar_rot)
        zerotol = 1.e-15
        self.assertTrue(np.max(np.abs(proj - self._ivar_orig))<zerotol)
        self.assertTrue(np.max(np.abs(proj - self._ivar_unrot))<zerotol)

        ### with projection
        unrot2 = self._unrot[:,:1]
        ivar_unrot = self._ivar_rot.dot(unrot2)
        net = QoIAwareProjectionPOUnet(unrot2, **init_uniform_partitions([5], ivar_unrot), basis_type='linear')
        proj = net.projection(self._ivar_rot)
        self.assertTrue(np.max(np.abs(proj.ravel() - self._ivar_orig[:,0]))<zerotol)
        self.assertTrue(np.max(np.abs(proj - ivar_unrot))<zerotol)

        ### with bias
        ivar_unrot = self._ivar_rot.dot(unrot2) + 0.2
        net = QoIAwareProjectionPOUnet(unrot2, **init_uniform_partitions([5], ivar_unrot), basis_type='linear', projection_biases=np.array([0.2]))
        proj = net.projection(self._ivar_rot)
        self.assertTrue(np.max(np.abs(proj - ivar_unrot))<zerotol)

    def test_utilities__QoIAwareProjectionPOUnet__reconstruction(self):
        ### only rotation
        net = QoIAwareProjectionPOUnet(self._unrot, **init_uniform_partitions([5,7], self._ivar_unrot), basis_type='linear')
        dvar = np.vstack((self._ivar_rot[:,0] + self._ivar_rot[:,1], 2.*self._ivar_rot[:,0] + 3.*self._ivar_rot[:,1])).T
        def dvar_func(proj_weights):
            temp = tf.Variable(np.expand_dims(dvar, axis=2), name='eval_qoi', dtype=net._reconstruction._dtype)
            temp = net.tf_projection(temp, nobias=True)
            return temp
        net.build_training_graph(self._ivar_rot, dvar_func)
        net.train(10,archive_rate=1)
        pred = net(self._ivar_rot)
        recon = net.reconstruction_model
        proj = net.projection(self._ivar_rot)
        pred_recon = recon(proj)

        zerotol = 1.e-16
        self.assertTrue(np.max(np.abs(recon.ivar_center - net.proj_ivar_center))<zerotol)
        self.assertTrue(np.max(np.abs(recon.ivar_scale - net.proj_ivar_scale))<zerotol)
        self.assertTrue(np.max(np.abs(pred - pred_recon))<zerotol)

        ### with projection
        net = QoIAwareProjectionPOUnet(self._unrot[:,:1], **init_uniform_partitions([5], self._ivar_unrot[:,:1]), basis_type='linear')
        net.build_training_graph(self._ivar_rot, dvar_func)
        net.train(10,archive_rate=1)
        pred = net(self._ivar_rot)
        recon = net.reconstruction_model
        proj = net.projection(self._ivar_rot)
        pred_recon = recon(proj)

        self.assertTrue(np.max(np.abs(recon.ivar_center - net.proj_ivar_center))<zerotol)
        self.assertTrue(np.max(np.abs(recon.ivar_scale - net.proj_ivar_scale))<zerotol)
        self.assertTrue(np.max(np.abs(pred - pred_recon))<zerotol)

        ### with projection input
        net = QoIAwareProjectionPOUnet(self._unrot[:,:1], **init_uniform_partitions([5], self._ivar_unrot[:,:1]), basis_type='linear')
        def dvar_func(proj_weights):
            temp = tf.Variable(np.expand_dims(dvar, axis=2), name='eval_qoi', dtype=net._reconstruction._dtype)
            return tf.reduce_sum(temp * proj_weights, axis=1)
        net.build_training_graph(self._ivar_rot, dvar_func)
        net.train(10,archive_rate=1)
        pred2 = net(self._ivar_rot)
        self.assertTrue(np.max(np.abs(pred - pred2))<zerotol)

        ### single dvar
        net = QoIAwareProjectionPOUnet(self._unrot, **init_uniform_partitions([5,7], self._ivar_unrot), basis_type='linear')
        def dvar_func(proj_weights):
            return tf.Variable(self._ivar_rot[:,0], name='eval_qoi', dtype=net._reconstruction._dtype)
        net.build_training_graph(self._ivar_rot, dvar_func)
        net.train(10,archive_rate=1)
        pred = net(self._ivar_rot)
        recon = net.reconstruction_model
        proj = net.projection(self._ivar_rot)
        pred_recon = recon(proj)
        self.assertTrue(np.max(np.abs(recon.ivar_center - net.proj_ivar_center))<zerotol)
        self.assertTrue(np.max(np.abs(recon.ivar_scale - net.proj_ivar_scale))<zerotol)
        self.assertTrue(np.max(np.abs(pred - pred_recon))<zerotol)


    def test_utilities__QoIAwareProjectionPOUnet__trainweights(self):
        net = QoIAwareProjectionPOUnet(self._unrot, **init_uniform_partitions([5,7], self._ivar_unrot), basis_type='linear')
        dvar = np.vstack((self._ivar_rot[:,0] + self._ivar_rot[:,1], 2.*self._ivar_rot[:,0] + 3.*self._ivar_rot[:,1])).T
        def dvar_func(proj_weights):
            temp = tf.Variable(np.expand_dims(dvar, axis=2), name='eval_qoi', dtype=net._reconstruction._dtype)
            temp = net.tf_projection(temp, nobias=True)
            return temp
        try:
            net.build_training_graph(self._ivar_rot, dvar_func, first_trainable_idx=2) # idx outside bounds
        except:
            self.assertTrue(True)
        zerotol = 1.e-16
        ### only 1 variable weight
        net.build_training_graph(self._ivar_rot, dvar_func, first_trainable_idx=1)
        net.train(10,archive_rate=1)
        self.assertTrue(np.max(np.abs(self._unrot[:,0] - net.projection_weights[:,0]))<zerotol)
        self.assertTrue(np.max(np.abs(self._unrot[:,1] - net.projection_weights[:,1]))>zerotol)

        ### all variable weight
        net = QoIAwareProjectionPOUnet(self._unrot, **init_uniform_partitions([5,7], self._ivar_unrot), basis_type='linear')
        net.build_training_graph(self._ivar_rot, dvar_func, first_trainable_idx=0)
        net.train(10,archive_rate=1)
        self.assertTrue(np.max(np.abs(self._unrot[:,0] - net.projection_weights[:,0]))>zerotol)
        self.assertTrue(np.max(np.abs(self._unrot[:,1] - net.projection_weights[:,1]))>zerotol)

    def test_utilities__QoIAwareProjectionPOUnet__readwrite(self):
        pkl = '__removeaftertest2.pkl'
        net = QoIAwareProjectionPOUnet(self._unrot, **init_uniform_partitions([5,7], self._ivar_unrot), basis_type='linear')
        dvar = np.vstack((self._ivar_rot[:,0] + self._ivar_rot[:,1], 2.*self._ivar_rot[:,0] + 3.*self._ivar_rot[:,1])).T
        def dvar_func(proj_weights):
            temp = tf.Variable(np.expand_dims(dvar, axis=2), name='eval_qoi', dtype=net._reconstruction._dtype)
            temp = net.tf_projection(temp, nobias=True)
            return temp
        net.build_training_graph(self._ivar_rot, dvar_func)
        net.train(10,archive_rate=1)
        net.write_data_to_file(pkl)

        zerotol = 1.e-16
        net2 = QoIAwareProjectionPOUnet.load_from_file(pkl)
        net3 = QoIAwareProjectionPOUnet(**QoIAwareProjectionPOUnet.load_data_from_file(pkl))
        self.assertTrue(np.max(np.abs(net2(self._ivar_rot) - net(self._ivar_rot)))<zerotol)
        self.assertTrue(np.max(np.abs(net3(self._ivar_rot) - net(self._ivar_rot)))<zerotol)

        os.remove(pkl)
