from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Check for compiler optimizations
extra_compile_args = ["-O3"]  # Optimization level 3
if os.name == "posix":  # For Linux/Mac
    extra_compile_args.extend(["-march=native", "-mtune=native"])

# Define the extension
extensions = [
    Extension(
        "fast_algorithms",
        sources=["fast_algorithms.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="fast_algorithms",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    zip_safe=False,
)
