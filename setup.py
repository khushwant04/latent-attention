# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='attention_cuda',
    ext_modules=[
        CUDAExtension(
            name='attention_cuda',
            sources=['attention_cuda.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
