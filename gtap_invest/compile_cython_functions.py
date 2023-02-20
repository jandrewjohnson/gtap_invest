# from distutils.core import setup
# from distutils.extension import Extension

from setuptools import setup
from setuptools import Extension

# try:
#     from setuptools import setup
#     from setuptools import Extension
# except ImportError:
#     from distutils.core import setup
#     from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension('carbon_biophysical',
                         ['global_invest/carbon_biophysical.pyx'],
                         )]
returned = setup(
    name='carbon_biophysical',
    include_dirs=[numpy.get_include()],
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
