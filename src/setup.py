from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "puzzle_cython",
        ["puzzle_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],  # Optimization level
    )
]

setup(
    name="block_puzzle_solver",
    ext_modules=cythonize(ext_modules, annotate=True, language_level="3"),
    include_dirs=[np.get_include()],
    requires=["numpy", "scipy", "cython", "pygame"],
)
