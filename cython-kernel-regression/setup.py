from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from numpy import get_include as numpy_include

setup(
    ext_modules=cythonize(Extension(name='cykernel',
                                    sources=['cykernel.pyx'],
                                    extra_compile_args=['-O3', '-g', '-std=c++11', '-stdlib=libc++',
                                                        '-I' + numpy_include(),
                                                        '-ffast-math'],
                                    extra_link_args=['-framework', 'Accelerate',
                                                     '-L/usr/local/lib/'],
                                    language='c++'))
)
