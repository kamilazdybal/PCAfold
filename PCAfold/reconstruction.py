"""reconstruction.py: module for reconstruction of QoIs from manifolds."""

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
from PCAfold.preprocess import center_scale
import pickle
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()


_ndim_write = 'ndim'
_npartition_write = 'npartition'
_nbasis_write = 'nbasis'
_floattype_write = 'float_type'
_tpower_write = 'transform_power'
_tshift_write = 'transform_shift'
_tsignshift_write = 'transform_sign_shift'
_ivarcenter_write = 'ivar_center'
_ivarscale_write = 'ivar_scale'
_pcenters_write = 'centers'
_pshapes_write = 'shapes'
_coeffs_write = 'coeffs'


def init_uniform_partitions(list_npartitions, ivars, width_factor=0.5, verbose=False):
    """
    Computes parameters for initializing partition locations near training data with uniform spacing in each dimension.

    **Example:**

    .. code:: python

        from PCAfold import init_uniform_partitions
        import numpy as np

        # Generate dummy data set:
        ivars = np.random.rand(100,2)

        # compute partition parameters for an initial 5x7 grid:
        init_data = init_uniform_partitions([5, 7], ivars)

    :param list_npartitions:
        list of integers specifying the number of partitions to try initializing in each dimension. Only partitions near the provided ivars are kept.
    :param ivars:
        array of independent variables used for determining which partitions to keep
    :param width_factor:
        (optional, default 0.5) the factor multiplying the spacing between partitions for initializing the partitions' RBF widths
    :param verbose:
        (optional, default False) when True, prints the number of partitions retained compared to the initial grid

    :return:
        a dictionary of partition parameters to be used in initializing a ``PartitionOfUnityNetwork``
    """
    ndim = len(list_npartitions)
    if ndim != ivars.shape[1]:
        raise ValueError("length of partition list must match dimensionality of ivars.")
    ivars_cs, center, scale = center_scale(ivars, '0to1')

    init_parameters = {}
    init_parameters['ivar_center'] = np.array([center])
    init_parameters['ivar_scale'] = np.array([scale])

    boundaries = []
    widths = np.zeros(ndim)
    for i in range(ndim):
        full_boundaries = np.linspace(0,1,list_npartitions[i]+1)
        boundaries.append(full_boundaries[:-1])
        widths[i] = full_boundaries[1] - full_boundaries[0]
    boundaries = np.vstack([b.ravel() for b in np.meshgrid(*boundaries)]).T

    partition_centers = []
    for i in range(boundaries.shape[0]):
        subivar = ivars_cs.copy()
        for j in range(boundaries.shape[1]):
            window = np.argwhere((subivar[:, j]>=boundaries[i,j])&(subivar[:, j]<=boundaries[i,j]+widths[j]))[:, 0]
            subivar = subivar[window,:]
            if len(window) == 0:
                break
            else:
                if j == ndim-1:
                    partition_centers.append(boundaries[i,:]+0.5*widths)
    partition_centers = np.array(partition_centers)
    init_parameters['partition_centers'] = partition_centers
    if verbose:
        print('kept', partition_centers.shape[0], 'partitions out of',boundaries.shape[0])

    partition_shapes = np.zeros_like(partition_centers)
    for i in range(ndim):
        partition_shapes[:,i] = width_factor * widths[i]
    init_parameters['partition_shapes'] = 1./partition_shapes

    return init_parameters

class PartitionOfUnityNetwork:
    """
    A class for reconstruction (regression) of QoIs using POUnets.

    The POUnets are constructed with a single-layer network of normalized radial basis functions (RBFs) whose neurons each own and weight a polynomial basis.
    For independent variable inputs :math:`\\vec{x}` of dimensionality :math:`d`, the :math:`i^{\\text{th}}` partition or neuron is computed as

    .. math::

        \\Phi_i(\\vec{x};\\vec{h}_i,K_i) = \\phi^{{\\rm RBF}}_i(\\vec{x};\\vec{h}_i,K_i)/\\sum_j \\phi^{{\\rm RBF}}_j(\\vec{x};\\vec{h}_i,K_i)

    where

    .. math::

        \\phi_i^{{\\rm RBF}}(\\vec{x};\\vec{h}_i,K_i) = \\exp\\left(-(\\vec{x}-\\vec{h}_i)^\\mathsf{T}K_i(\\vec{x}-\\vec{h}_i)\\right) \\nonumber

    with vector :math:`\\vec{h}_i` and diagonal matrix :math:`K_i` defining the :math:`d` center and :math:`d` shape parameters, respectively, for training.

    The final output of a POUnet is then obtained through

    .. math::

        g(\\vec{x};\\vec{h},K,c) = \\sum_{i=1}^{p}\\left(\\Phi_i(\\vec{x};\\vec{h}_i,K_i)\\sum_{k=1}^{b}c_{i,k}m_k(\\vec{x})\\right)

    where the polynomial basis is represented as a sum of :math:`b` Taylor monomials,
    with the :math:`k^{\\text{th}}` monomial written as :math:`m_k(\\vec{x})`,
    that are multiplied by trainable basis coefficients :math:`c`.
    The number of basis monomials is determined by the ``basis_type`` for the polynomial.
    For example, in two-dimensional space, a quadratic polynomial basis contains :math:`b=6`
    monomial functions :math:`\\{1, x_1, x_2, x_1^2, x_2^2, x_1x_2\\}`.
    The combination of the partitions and polynomial basis functions creates localized polynomial fits for a QoI.

    More information can be found in :cite:`Armstrong2022`.

    The ``PartitionOfUnityNetwork`` class also provides a nonlinear transformation for the dependent variable(s) during training,
    which can be beneficial if the variable changes over orders of magnitude, for example.
    The equation for the transformation of variable :math:`f` is

    .. math::

        (|f + s_1|)^\\alpha \\text{sign}(f + s_1) + s_2 \\text{sign}(f + s_1)

    where :math:`\\alpha` is the ``transform_power``, :math:`s_1` is the ``transform_shift``, and :math:`s_2` is the ``transform_sign_shift``.

    **Example:**

    .. code:: python

        from PCAfold import init_uniform_partitions, PartitionOfUnityNetwork
        import numpy as np

        # Generate dummy data set:
        ivars = np.random.rand(100,2)
        dvars = 2.*ivars[:,0] + 3.*ivars[:,1]

        # Initialize the POUnet parameters
        net = PartitionOfUnityNetwork(**init_uniform_partitions([5,7], ivars), basis_type='linear')

        # Build the training graph with provided training data
        net.build_training_graph(ivars, dvars)

        # (optional) update the learning rate (default is 1.e-3)
        net.update_lr(1.e-4)

        # (optional) update the least-squares regularization (default is 1.e-10)
        net.update_l2reg(1.e-10)

        # Train the POUnet
        net.train(1000)

        # Evaluate the POUnet
        pred = net(ivars)

        # Evaluate the POUnet derivatives
        der = net.derivatives(ivars)

        # Save the POUnet to a file
        net.write_data_to_file('filename.pkl')

        # Load a POUnet from file
        net2 = PartitionOfUnityNetwork.load_from_file('filename.pkl')

        # Evaluate the loaded POUnet (without needing to call build_training_graph)
        pred2 = net2(ivars)

    :param partition_centers:
        array size (number of partitions) x (number of ivar inputs) for partition locations
    :param partition_shapes:
        array size (number of partitions) x (number of ivar inputs) for partition shapes influencing the RBF widths
    :param basis_type:
        string (``'constant'``, ``'linear'``, or ``'quadratic'``) for the degree of polynomial basis
    :param ivar_center:
        (optional, default ``None``) array for centering the ivar inputs before evaluating the POUnet, if ``None`` centers with zeros
    :param ivar_scale:
        (optional, default ``None``) array for scaling the ivar inputs before evaluating the POUnet, if ``None`` scales with ones
    :param basis_coeffs:
        (optional, default ``None``) if the array of polynomial basis coefficients is known, it may be provided here, 
        otherwise it will be initialized with ``build_training_graph`` and trained with ``train``
    :param transform_power:
        (optional, default 1.) the power parameter used in the transformation equation during training
    :param transform_shift:
        (optional, default 0.) the shift parameter used in the transformation equation during training
    :param transform_sign_shift:
        (optional, default 0.) the signed shift parameter used in the transformation equation during training
    :param dtype:
        (optional, default ``'float64'``) string specifying either float type ``'float64'`` or ``'float32'``

    **Attributes:**

    - **partition_centers** - (read only) array of the current partition centers
    - **partition_shapes** - (read only) array of the current partition shape parameters
    - **basis_type** - (read only) string relaying the basis degree
    - **basis_coeffs** - (read only) array of the current basis coefficients
    - **ivar_center** - (read only) array of the centering parameters for the ivar inputs
    - **ivar_scale** - (read only) array of the scaling parameters for the ivar inputs
    - **dtype** - (read only) string relaying the data type (``'float64'`` or ``'float32'``)
    - **training_archive** - (read only) dictionary of the errors and POUnet states archived during training
    - **iterations** - (read only) array of the iterations archived during training
    """

    def __init__(self, 
                 partition_centers,
                 partition_shapes,
                 basis_type,
                 ivar_center=None,
                 ivar_scale=None,
                 basis_coeffs=None,
                 transform_power=1.,
                 transform_shift=0.,
                 transform_sign_shift=0.,
                 dtype='float64'
                ):
        self._sess = tf.Session()
        if partition_centers.shape != partition_shapes.shape:
            raise ValueError("Size of partition centers and shapes must match")
        self._partition_centers = partition_centers.copy()
        self._partition_shapes = partition_shapes.copy()
        if basis_type not in ['constant','linear','quadratic']:
            raise ValueError("Supported basis_type includes constant, linear, or quadratic")
        self._basis_type = basis_type
        self._np = self._partition_centers.shape[0]
        self._nd = self._partition_centers.shape[1]

        if dtype != 'float64' and dtype != 'float32':
            raise ValueError("Only float32 and float64 dtype supported")
        self._dtype_str = dtype
        self._dtype = tf.float64 if self._dtype_str == 'float64' else tf.float32

        l2reg_f = 1.e-10 if self._dtype_str == 'float64' else 1.e-6
        self._l2reg = tf.Variable(l2reg_f, name='l2reg', dtype=self._dtype)
        self._lr = tf.Variable(1.e-3, name='lr', dtype=self._dtype)

        self._ivar_center = ivar_center if ivar_center is not None else np.zeros((1,partition_centers.shape[1]))
        self._inv_ivar_scale = 1./ivar_scale if ivar_scale is not None else np.ones((1,partition_centers.shape[1]))
        if len(self._ivar_center.shape) == 1:
            self._ivar_center = np.array([self._ivar_center])
        if len(self._inv_ivar_scale.shape) == 1:
            self._inv_ivar_scale = np.array([self._inv_ivar_scale])
        if self._ivar_center.shape[1] != partition_centers.shape[1]:
            raise ValueError("ivar_center dimensionality", self._ivar_center.shape[1],"does not match partition parameters",partition_centers.shape[1])
        if self._inv_ivar_scale.shape[1] != partition_centers.shape[1]:
            raise ValueError("ivar_scale dimensionality", self._inv_ivar_scale.shape[1],"does not match partition parameters",partition_centers.shape[1])

        self._transform_power = transform_power
        self._transform_shift = transform_shift
        self._transform_sign_shift = transform_sign_shift

        if basis_coeffs is not None:
            self._basis_coeffs = basis_coeffs.copy()
            self._isready = True
            self._t_basis_coeffs = tf.Variable(self._basis_coeffs, name='basis_coeffs', dtype=self._dtype)
        else:
            self._basis_coeffs = None
            self._isready = False

        self._t_ivar_center = tf.constant(np.expand_dims(self._ivar_center, axis=2), name='centers', dtype=self._dtype)
        self._t_inv_ivar_scale = tf.constant(np.expand_dims(self._inv_ivar_scale, axis=2), name='scales', dtype=self._dtype)
        self._t_xp = tf.Variable(self._partition_centers, name='partition_centers', dtype=self._dtype)
        self._t_sp = tf.Variable(self._partition_shapes, name='partition_scales', dtype=self._dtype)
        self._sess.run(tf.global_variables_initializer())
        self._built_graph = False

    @classmethod
    def load_data_from_file(cls, filename):
        """
        Load data from a specified ``filename`` with pickle (following ``write_data_to_file``)

        :param filename:
            string

        :return:
            dictionary of the POUnet data
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
            ``PartitionOfUnityNetwork``
        """
        return cls(**cls.load_data_from_file(filename))

    @classmethod
    def load_data_from_txt(cls, filename, verbose=False):
        """
        Load data from a specified txt ``filename`` (following ``write_data_to_txt``)

        :param filename:
            string
        :param verbose:
            (optional, default False) print out the data as it is read

        :return:
            dictionary of the POUnet data
        """
        out_data = {}
        with open(filename) as file:
            content = file.readlines()

            if content[1].strip('\n')!=_ndim_write:
                raise ValueError('inconsistent read',content[1].strip('\n'),'and expected',_ndim_write)
            ndim = int(content[2])
            if verbose:
                print(content[1].strip('\n'), ndim)

            if content[3].strip('\n')!=_npartition_write:
                raise ValueError('inconsistent read',content[3].strip('\n'),'and expected',_npartition_write)
            npart = int(content[4])
            if verbose:
                print(content[3].strip('\n'), npart)
            
            if content[5].strip('\n')!=_nbasis_write:
                raise ValueError('inconsistent read',content[5].strip('\n'),'and expected',_nbasis_write)
            nbasis = int(content[6])
            if verbose:
                print(content[5].strip('\n'), nbasis)
            if nbasis==0:
                out_data['basis_type'] = 'constant'
            elif nbasis==1:
                out_data['basis_type'] = 'linear'
            elif nbasis==2:
                out_data['basis_type'] = 'quadratic'

            if content[7].strip('\n')!=_floattype_write:
                raise ValueError('inconsistent read',content[7].strip('\n'),'and expected',_floattype_write)
            float_str = content[8].strip('\n')
            out_data['dtype'] = float_str
            if verbose:
                print(content[7].strip('\n'), float_str)

            if content[9].strip('\n')!=_tpower_write:
                raise ValueError('inconsistent read',content[9].strip('\n'),'and expected',_tpower_write)
            transform_power = float(content[10])
            out_data['transform_power'] = transform_power
            if verbose:
                print(content[9].strip('\n'), transform_power)

            if content[11].strip('\n')!=_tshift_write:
                raise ValueError('inconsistent read',content[11].strip('\n'),'and expected',_tshift_write)
            transform_shift = float(content[12])
            out_data['transform_shift'] = transform_shift
            if verbose:
                print(content[11].strip('\n'), transform_shift)

            if content[13].strip('\n')!=_tsignshift_write:
                raise ValueError('inconsistent read',content[13].strip('\n'),'and expected',_tsignshift_write)
            transform_sign_shift = float(content[14])
            out_data['transform_sign_shift'] = transform_sign_shift
            if verbose:
                print(content[13].strip('\n'), transform_sign_shift)

            if content[15].strip('\n')!=_ivarcenter_write:
                raise ValueError('inconsistent read',content[15].strip('\n'),'and expected',_ivarcenter_write)
            if verbose:
                print(content[15].strip('\n'))
            istart = 16
            ivar_centers = np.zeros((1,ndim))
            for i in range(ndim):
                ivar_centers[0,i] = float(content[istart])
                istart += 1
            if verbose:
                print(ivar_centers)
            out_data['ivar_center'] = ivar_centers

            if content[istart].strip('\n')!=_ivarscale_write:
                raise ValueError('inconsistent read',content[istart].strip('\n'),'and expected',_ivarscale_write)
            ivar_scale = np.zeros((1,ndim))
            if verbose:
                print(content[istart].strip('\n'))
            istart += 1
            for i in range(ndim):
                ivar_scale[0,i] = float(content[istart])
                istart += 1
            if verbose:
                print(ivar_scale)
            out_data['ivar_scale'] = ivar_scale

            
            if content[istart].strip('\n')!=_pcenters_write:
                raise ValueError('inconsistent read',content[istart].strip('\n'),'and expected',_pcenters_write)
            ncoef = int(ndim*npart)
            centers = np.zeros(ncoef)
            if verbose:
                print(content[istart].strip('\n'))
            istart += 1
            for i in range(ncoef):
                centers[i] = float(content[istart])
                istart += 1
            centers = centers.reshape(npart,ndim)
            if verbose:
                print(centers)
            out_data['partition_centers'] = centers

            if content[istart].strip('\n')!=_pshapes_write:
                raise ValueError('inconsistent read',content[istart].strip('\n'),'and expected',_pshapes_write)
            shapes = np.zeros(ncoef)
            if verbose:
                print(content[istart].strip('\n'))
            istart += 1
            for i in range(ncoef):
                shapes[i] = float(content[istart])
                istart += 1
            shapes = shapes.reshape(npart,ndim)
            if verbose:
                print(shapes)
            out_data['partition_shapes'] = shapes

            if content[istart].strip('\n')!=_coeffs_write:
                raise ValueError('inconsistent read',content[istart].strip('\n'),'and expected',_coeffs_write)
            ncoef = nbasis * ndim + 1
            if nbasis>1:
                ncoef += ndim*(ndim+1)*0.5-ndim
            ncoef *= npart
            ncoef = int(ncoef)
            coeffs = np.zeros(ncoef)
            if verbose:
                print(content[istart].strip('\n'), ncoef)
            istart += 1
            for i in range(ncoef):
                coeffs[i] = float(content[istart])
                istart += 1
            coeffs = coeffs.reshape(npart,ncoef//npart).T
            coeffs = coeffs.ravel()
            if verbose:
                print(coeffs)
            out_data['basis_coeffs'] = np.array([coeffs])
        return out_data

    @tf.function
    def tf_transform(self, x):
        inter = x + self._transform_shift
        return tf.math.pow(tf.cast(tf.abs(inter), dtype=self._dtype), self._transform_power) * tf.math.sign(inter) + self._transform_sign_shift*tf.math.sign(inter)

    @tf.function
    def tf_untransform(self, x):
        o = self._transform_sign_shift*tf.math.sign(x)
        inter = tf.math.sign(x-o) * tf.math.pow(tf.cast(tf.abs(x-o), dtype=self._dtype), 1./self._transform_power)
        return inter - self._transform_shift

    @tf.function
    def tf_partitions_prenorm(self, x):
        # evaluate non-normalized partitions
        self._t_xmxp_scaled = (x - tf.transpose(self._t_xp)) * tf.transpose(self._t_sp)
        return tf.math.exp(-tf.reduce_sum(self._t_xmxp_scaled * self._t_xmxp_scaled, axis=1))

    @tf.function
    def tf_partitions(self, x):
        # evaluate normalized partitions
        self._t_nnp = self.tf_partitions_prenorm(x)
        return tf.transpose(tf.transpose(self._t_nnp) / tf.reduce_sum(self._t_nnp, axis=1))

    @tf.function
    def tf_predict(self, x, t_p):
        # evaluate basis
        for ib in range(self._t_basis_coeffs.shape[0]):
            if self._basis_type == 'constant':
                t_basis = self._t_basis_coeffs[ib, :self._np]
            elif self._basis_type == 'linear':
                t_basis = self._t_basis_coeffs[ib, :self._np]
                for i in range(self._nd):
                    t_basis += self._t_basis_coeffs[ib, self._np*(i+1):self._np*(i+2)] * x[:, i:i+1, 0]
            elif self._basis_type == 'quadratic':
                if self._nd == 1:
                    t_basis = self._t_basis_coeffs[ib, :self._np] + \
                            self._t_basis_coeffs[ib, self._np:self._np*2] * x[:, :1, 0] + \
                            self._t_basis_coeffs[ib, self._np*2:self._np*3] * x[:, :1, 0] * x[:, :1, 0]
                elif self._nd == 2:
                    t_basis = self._t_basis_coeffs[ib, :self._np] + \
                                    self._t_basis_coeffs[ib, self._np:self._np*2] * x[:, :1, 0] + \
                                    self._t_basis_coeffs[ib, self._np*2:self._np*3] * x[:, 1:2, 0] + \
                                    self._t_basis_coeffs[ib, self._np*3:self._np*4] * x[:, :1, 0] * x[:, :1, 0] + \
                                    self._t_basis_coeffs[ib, self._np*4:self._np*5] * x[:, 1:2, 0] * x[:, 1:2, 0] 
                    t_basis += self._t_basis_coeffs[ib, self._np*5:self._np*6] * x[:, :1, 0] * x[:, 1:2, 0] # crossterm
                elif self._nd == 3:
                    t_basis = self._t_basis_coeffs[ib, :self._np] + \
                                    self._t_basis_coeffs[ib, self._np:self._np*2] * x[:, :1, 0] + \
                                    self._t_basis_coeffs[ib, self._np*2:self._np*3] * x[:, 1:2, 0] + \
                                    self._t_basis_coeffs[ib, self._np*3:self._np*4] * x[:, 2:3, 0] + \
                                    self._t_basis_coeffs[ib, self._np*4:self._np*5] * x[:, :1, 0] * x[:, :1, 0] + \
                                    self._t_basis_coeffs[ib, self._np*5:self._np*6] * x[:, 1:2, 0] * x[:, 1:2, 0] + \
                                    self._t_basis_coeffs[ib, self._np*6:self._np*7] * x[:, 2:3, 0] * x[:, 2:3, 0]
                    t_basis += self._t_basis_coeffs[ib, self._np*7:self._np*8] * x[:, :1, 0] * x[:, 1:2, 0] # crossterm
                    t_basis += self._t_basis_coeffs[ib, self._np*8:self._np*9] * x[:, :1, 0] * x[:, 2:3, 0] # crossterm
                    t_basis += self._t_basis_coeffs[ib, self._np*9:self._np*10] * x[:, 2:3, 0] * x[:, 1:2, 0] # crossterm
                else:
                    raise ValueError("unsupported dimensionality + degree combo")
            else:
                raise ValueError("unsupported dimensionality + degree combo")
            # evaluate full network
            if ib==0:
                output = tf.reduce_sum(t_p * t_basis, axis=1)
            else:
                if ib==1:
                    output = tf.expand_dims(output, axis=1)
                temp = tf.expand_dims(tf.reduce_sum(t_p * t_basis, axis=1), axis=1)
                output = tf.concat([output, temp], 1)
        return output

    def build_training_graph(self, ivars, dvars, error_type='abs', constrain_positivity=False, istensor=False, verbose=False):
        """
        Construct the graph used during training (including defining the training errors) with the provided training data

        :param ivars:
            array of independent variables for training
        :param dvars:
            array of dependent variable(s) for training
        :param error_type:
            (optional, default ``'abs'``) the type of training error: relative ``'rel'`` or absolute ``'abs'``
        :param constrain_positivity:
            (optional, default False) when True, it penalizes the training error with :math:`f - |f|` for dependent variables :math:`f`. This can be useful when used in ``QoIAwareProjectionPOUnet``
        :param istensor:
            (optional, default False) whether to evaluate ivars and dvars as tensorflow Tensors or numpy arrays
        :param verbose:
            (options, default False) print out the number of the partition and basis parameters when True
        """
        if error_type != 'rel' and error_type != 'abs':
            raise ValueError("Supported error_type include rel and abs.")

        if istensor:
            if len(ivars.shape)!=3:
                raise ValueError("Expected ivars with 3 tensor dimensions but recieved",len(ivars.shape))
            self._t_xyt_prescale = ivars
            self._t_ft = dvars
        else:
            if len(ivars.shape)!=2:
                raise ValueError("Expected ivars with 2 array dimensions but recieved",len(ivars.shape))
            self._t_xyt_prescale = tf.Variable(np.expand_dims(ivars, axis=2), name='eval_pts', dtype=self._dtype)
            self._t_ft = tf.Variable(dvars, name='eval_qoi', dtype=self._dtype)

        if ivars.shape[1] != self._nd:
            raise ValueError("Dimensionality of ivars",ivars.shape[1],'does not match expected dimensionality',self._nd)
        self._t_ft = self.tf_transform(self._t_ft)
        self._t_xyt = (self._t_xyt_prescale - self._t_ivar_center)*self._t_inv_ivar_scale

        self._t_penalty = self._t_ft - tf.cast(tf.abs(self._t_ft), dtype=self._dtype)

        ftdim = len(self._t_ft.shape)
        self._nft = float(self._t_ft.get_shape().as_list()[1]) if ftdim>1 else 1.

        if self._basis_coeffs is None:
            if self._basis_type == 'constant':
                self._basis_coeffs = np.ones((1, self._np)) / self._np
            elif self._basis_type == 'linear':
                self._basis_coeffs = np.hstack((np.ones((1, self._np)) / self._np, np.zeros((1, self._np*self._nd))))
            elif self._basis_type == 'quadratic':
                if self._nd == 1:
                    self._basis_coeffs = np.hstack((np.ones((1, self._np)) / self._np, np.zeros((1, self._np*2))))
                elif self._nd == 2:
                    self._basis_coeffs = np.hstack((np.ones((1, self._np)) / self._np, np.zeros((1, self._np*5))))
                elif self._nd == 3:
                    self._basis_coeffs = np.hstack((np.ones((1, self._np)) / self._np, np.zeros((1, self._np*9))))
                else:
                    raise ValueError('unsupported dimensionality + degree combo')
            else:
                raise ValueError('unsupported dimensionality + degree combo')
            
            if ftdim>1:
                orig_coeffs = self._basis_coeffs.copy()
                for i in range(1,self._t_ft.shape[1]):
                    self._basis_coeffs = np.vstack((self._basis_coeffs, orig_coeffs))
            self._isready = True
        else:
            if ftdim>1:
                if self._basis_coeffs.shape[0] != self._t_ft.shape[1]:
                    raise ValueError("Expected",self._basis_coeffs.shape[0],"dependent variables but received",self._t_ft.shape[1])
            else:
                if self._basis_coeffs.shape[0] != 1:
                    raise ValueError("Expected",self._basis_coeffs.shape[0],"dependent variables but received 1")

        self._t_basis_coeffs = tf.Variable(self._basis_coeffs, name='basis_coeffs', dtype=self._dtype)
        if verbose:
            print('Constructing a graph for',int(self._nft),'dependent variables with',self._t_xp.shape[0],'partitions and',self._t_basis_coeffs.shape[1],'coefficients per variable.')

        # predictions
        self._t_p = self.tf_partitions(self._t_xyt)
        self._t_pred = self.tf_predict(self._t_xyt, self._t_p)

        # evaluate error
        self._rel_err = (self._t_pred - self._t_ft) # start with absolute error
        if error_type == 'rel':
            # divide by appropriate factor if want relative error
            self._rel_err /= (tf.cast(tf.abs(self._t_ft), dtype=self._dtype) + tf.constant(1.e-4, name='rel_den', dtype=self._dtype))

        self._t_msre = tf.reduce_mean(tf.math.square(self._rel_err))
        self._t_infre = tf.reduce_max(tf.cast(tf.abs(self._rel_err), dtype=self._dtype))
        self._t_ssre = tf.reduce_sum(tf.math.square(self._rel_err))/self._nft

        # set training error
        self._train_err = self._t_ssre
        if constrain_positivity:
            self._train_err += tf.cast(tf.abs(tf.reduce_sum(self._t_penalty)/self._nft), dtype=self._dtype)

        # optimizers
        self._part_l2_opt = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(self._train_err, var_list=[self._t_xp, self._t_sp])

        # set up lstsq
        if self._basis_type == 'constant':
            self._Amat = tf.identity(self._t_p)
        elif self._basis_type == 'linear':
            self._Amat = tf.concat([self._t_p, self._t_p*self._t_xyt[:, :1, 0]], 1)
            for i in range(1, self._nd):
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, i:i+1, 0]], 1)
        elif self._basis_type == 'quadratic':
            if self._nd==1:
                self._Amat = tf.concat([self._t_p, self._t_p*self._t_xyt[:, :1, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, :1, 0]*self._t_xyt[:, :1, 0]], 1)
            elif self._nd==2:
                self._Amat = tf.concat([tf.concat([self._t_p, self._t_p*self._t_xyt[:, :1, 0]], 1), self._t_p*self._t_xyt[:, 1:2, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, :1, 0]*self._t_xyt[:, :1, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, 1:2, 0]*self._t_xyt[:, 1:2, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, :1, 0]*self._t_xyt[:, 1:2, 0]], 1) # crossterm
            elif self._nd==3:
                self._Amat = tf.concat([tf.concat([self._t_p, self._t_p*self._t_xyt[:, :1, 0]], 1), self._t_p*self._t_xyt[:, 1:2, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, 2:3, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, :1, 0]*self._t_xyt[:, :1, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, 1:2, 0]*self._t_xyt[:, 1:2, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, 2:3, 0]*self._t_xyt[:, 2:3, 0]], 1)
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, :1, 0]*self._t_xyt[:, 1:2, 0]], 1) # crossterm
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, :1, 0]*self._t_xyt[:, 2:3, 0]], 1) # crossterm
                self._Amat = tf.concat([self._Amat, self._t_p*self._t_xyt[:, 2:3, 0]*self._t_xyt[:, 1:2, 0]], 1) # crossterm
            else:
                raise ValueError("unsupported dimensionality + degree combo")
        else:
            raise ValueError("unsupported dimensionality + degree combo")

        if ftdim>1:
            rhs = self._t_ft
        else:
            rhs = tf.expand_dims(self._t_ft, axis=1)

        if error_type == 'rel':
            # if training error is relative
            if self._nft>1:
                raise ValueError("Cannot support relative error lstsqr for more than 1 variable")
            else:
                lstsq_rel_coeff = 1. / (tf.abs(rhs) + 1.e-4) # must be same format as error for partitions
                self._Amat *= lstsq_rel_coeff
                rhs *= lstsq_rel_coeff

        # perform lstsq
        self._lstsq = self._t_basis_coeffs.assign(tf.transpose(tf.linalg.lstsq(self._Amat, rhs, l2_regularizer=self._l2reg)))

        self._sess.run(tf.global_variables_initializer())

        msre, infre, ssre = self._sess.run(self._t_msre), self._sess.run(self._t_infre), self._sess.run(self._t_ssre)
        self._iterations = list([0])
        self._training_archive = {'mse': list([msre]), 'inf': list([infre]), 'sse': list([ssre]), 'data':list([self.__getstate__()])}
        self._built_graph = True

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
        self._sess.run(self._l2reg.assign(l2reg))

    def lstsq(self, verbose=True):
        """
        update the basis coefficients with least-squares regression

        :param verbose:
            (optional, default True) prints when least-squares solve is performed when True
        """
        if not self._built_graph:
            raise ValueError("Need to call build_training_graph before lstsq.")
        if verbose:
            print('performing least-squares solve')
        self._sess.run(self._lstsq)

    def train(self, iterations, archive_rate=100, use_best_archive_sse=True, verbose=False):
        """
        Performs training using a least-squares gradient descent block coordinate descent strategy.
        This alternates between updating the partition parameters with gradient descent and updating the basis coefficients with least-squares.

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
            best_error = self._sess.run(self._t_ssre).copy()
            best_centers = self._sess.run(self._t_xp).copy()
            best_shapes = self._sess.run(self._t_sp).copy()
            best_coeffs = self._sess.run(self._t_basis_coeffs).copy()

        for i in range(iterations):
            self._sess.run(self._part_l2_opt) # update partitions
            self._sess.run(self._lstsq) # update basis coefficients

            if not (i + 1) % archive_rate:
                msre, infre, ssre = self._sess.run(self._t_msre), self._sess.run(self._t_infre), self._sess.run(self._t_ssre)
                if verbose:
                    print(f'  {i + 1:10} | {msre:10.2e} | {100. * infre:10.2f}% | {ssre:10.2e}')
                self._iterations.append(self._iterations[-1] + archive_rate)
                self._training_archive['mse'].append(msre)
                self._training_archive['inf'].append(infre)
                self._training_archive['sse'].append(ssre)
                self._training_archive['data'].append(self.__getstate__())

                if use_best_archive_sse:
                    if ssre < best_error:
                        if verbose:
                            print('resetting best error')
                        best_error = ssre.copy()
                        best_centers = self._sess.run(self._t_xp).copy()
                        best_shapes = self._sess.run(self._t_sp).copy()
                        best_coeffs = self._sess.run(self._t_basis_coeffs).copy()
        if use_best_archive_sse:
            self._sess.run(self._t_xp.assign(best_centers))
            self._sess.run(self._t_sp.assign(best_shapes))
            self._sess.run(self._t_basis_coeffs.assign(best_coeffs))

    @tf.function
    def tf_call(self, xeval):
        if self._isready:
            xeval_tf = (xeval - self._t_ivar_center)*self._t_inv_ivar_scale
            t_p = self.tf_partitions(xeval_tf)
            return self.tf_untransform(self.tf_predict(xeval_tf, t_p))
        else:
            raise ValueError("basis coefficients have not been set.")

    def __call__(self, xeval):
        """
        evaluate the POUnet

        :param xeval:
            array of independent variable query points

        :return:
            array of POUnet predictions
        """
        if xeval.shape[1] != self._nd:
            raise ValueError("Dimensionality of inputs",xeval.shape[1],'does not match expected dimensionality',self._nd)
        xeval_tf_prescale = tf.Variable(np.expand_dims(xeval, axis=2), name='eval_pts', dtype=self._dtype)
        self._sess.run(tf.variables_initializer([xeval_tf_prescale]))
        pred = self.tf_call(xeval_tf_prescale)
        return self._sess.run(pred)

    def __getstate__(self):
        """dictionary of current POUnet parameters"""
        return dict(partition_centers=self.partition_centers,
                    partition_shapes=self.partition_shapes,
                    basis_type=self.basis_type,
                    ivar_center=self.ivar_center,
                    ivar_scale=self.ivar_scale,
                    basis_coeffs=self.basis_coeffs,
                    transform_power=self._transform_power,
                    transform_shift=self._transform_shift,
                    transform_sign_shift=self._transform_sign_shift,
                    dtype=self.dtype
                   )

    def derivatives(self, xeval, dvar_idx=0):
        """
        evaluate the POUnet derivatives

        :param xeval:
            array of independent variable query points
        :param dvar_idx:
            (optional, default 0) index for the dependent variable whose derivatives are being evaluated

        :return:
            array of POUnet derivative evaluations
        """
        if xeval.shape[1] != self._nd:
            raise ValueError("Dimensionality of inputs",xeval.shape[1],'does not match expected dimensionality',self._nd)
        xeval_tf_prescale = tf.Variable(np.expand_dims(xeval, axis=2), name='eval_pts', dtype=self._dtype)
        self._sess.run(tf.variables_initializer([xeval_tf_prescale]))
        allpreds = []
        with tf.GradientTape() as tape:
            pred = self.tf_call(xeval_tf_prescale)
            if len(pred.shape)>1:
                pred = pred[:,dvar_idx]
        der = tape.gradient(pred, xeval_tf_prescale)
        return self._sess.run(der)[:,:,0]


    def partition_prenorm(self, xeval):
        """
        evaluate the POUnet partitions prior to normalization

        :param xeval:
            array of independent variable query points

        :return:
            array of POUnet RBF partition evaluations before normalization
        """
        if xeval.shape[1] != self._nd:
            raise ValueError("Dimensionality of inputs",xeval.shape[1],'does not match expected dimensionality',self._nd)
        xeval_tf_prescale = tf.Variable(np.expand_dims(xeval, axis=2), name='eval_pts', dtype=self._dtype)
        self._sess.run(tf.variables_initializer([xeval_tf_prescale]))
        xeval_tf = (xeval_tf_prescale - self._t_ivar_center)*self._t_inv_ivar_scale
        t_nnp = self.tf_partitions_prenorm(xeval_tf)
        return self._sess.run(t_nnp)

    def write_data_to_file(self, filename):
        """
        Save class data to a specified file using pickle. This does not include the archived data from training,
        which can be separately accessed with training_archive and saved outside of ``PartitionOfUnityNetwork``.

        :param filename:
            string
        """
        with open(filename, 'wb') as file_output:
            pickle.dump(self.__getstate__(), file_output)

    def write_data_to_txt(self, filename, nformat='%.14e'):
        """
        Save data to a specified txt file. This may be used to read POUnet parameters into other languages such as C++

        :param filename:
            string
        """
        if self._basis_coeffs.shape[0]>1:
            raise ValueError("Cannot read/write to txt for multiple dependent variables. Try ``write_data_to_pkl`` instead.")
        with open(filename, 'w') as f:
            f.write("POUNET" + '\n')

            f.write(_ndim_write + '\n')
            f.write(str(self._nd) + '\n')

            f.write(_npartition_write + '\n')
            f.write(str(self._np) + '\n')

            f.write(_nbasis_write + '\n')
            if self._basis_type == 'constant':
                nbasis=0
            elif self._basis_type == 'linear':
                nbasis=1
            elif self._basis_type == 'quadratic':
                nbasis=2
            else:
                raise ValueError("Unsupported basis type.")
            f.write(str(nbasis) + '\n')

            f.write(_floattype_write + '\n')
            f.write(self._dtype_str + '\n')

            f.write(_tpower_write + '\n')
            f.write(str(self._transform_power) + '\n')

            f.write(_tshift_write + '\n')
            f.write(str(self._transform_shift) + '\n')

            f.write(_tsignshift_write + '\n')
            f.write(str(self._transform_sign_shift) + '\n')

            f.write(_ivarcenter_write + '\n')
            np.savetxt(f, self._ivar_center.ravel(), fmt=nformat)
            f.write(_ivarscale_write + '\n')
            np.savetxt(f, 1./self._inv_ivar_scale.ravel(), fmt=nformat)

            f.write(_pcenters_write + '\n')
            pcenters = self.partition_centers.T
            np.savetxt(f, pcenters.ravel(order='F'), fmt=nformat)

            f.write(_pshapes_write + '\n')
            pshapes = self.partition_shapes.T
            np.savetxt(f, pshapes.ravel(order='F'), fmt=nformat)

            f.write(_coeffs_write + '\n')
            coeffs = self.basis_coeffs[0,:]
            totalbasis = coeffs.size//self._np
            coeffs = coeffs.reshape(totalbasis, self._np)
            np.savetxt(f, coeffs.ravel(order='F'), fmt=nformat)

    @property
    def iterations(self):
        return np.array(self._iterations)

    @property
    def training_archive(self):
        return {e: np.array(self._training_archive[e]) if e != 'data' else self._training_archive[e] for e in self._training_archive}

    def get_partition_sum(self, xeval):
        nnp = self.partition_prenorm(xeval)
        return np.sum(nnp, axis=1)

    @property
    def partition_centers(self):
        return self._sess.run(self._t_xp)

    @property
    def partition_shapes(self):
        return self._sess.run(self._t_sp)

    @property
    def basis_type(self):
        return self._basis_type

    @property
    def dtype(self):
        return self._dtype_str

    @property
    def basis_coeffs(self):
        return self._sess.run(self._t_basis_coeffs)

    @property
    def ivar_center(self):
        return self._ivar_center

    @property
    def ivar_scale(self):
        return 1./self._inv_ivar_scale
