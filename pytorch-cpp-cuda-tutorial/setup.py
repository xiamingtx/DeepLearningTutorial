#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:setup.py
# author:xm
# datetime:2024/6/6 15:04
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


# ROOT_DIR = osp.dirname(osp.abspath(__file__))
# include_dirs = [osp.join(ROOT_DIR, "include")]
#
# sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='cppcuda_tutorial',
    version='1.0',
    author='xiamingtx',
    author_email='xiamingtx03@gmail.com',
    description='cppcuda example',
    long_description='cppcuda example',
    ext_modules=[
        # CUDAExtension(
        #     name='cppcuda_tutorial',
        #     sources=sources,
        #     include_dirs=include_dirs,
        #     extra_compile_args={'cxx': ['-O2'],
        #                         'nvcc': ['-O2']}
        # )
        CppExtension(
            name="cppcuda_tutorial",
            sources=["interpolation.cpp"]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)