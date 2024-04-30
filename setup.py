from distutils.core import setup, Extension
import sysconfig

# Run this script with:
# python setup.py build
# python setup.py install

# What happens here? When we run "python setup.py build", 
# distutils will compile the C code and create a shared library.

# Therefore, the way python interface use C as its backend 
# happens here

def main():
    CFLAGS = ['-g', '-Wall', '-std=c99', '-fopenmp', '-mavx', '-mfma', '-pthread', '-O3']
    LDFLAGS = ['-fopenmp']
    # Use the setup function we imported and set up the modules.
    # You may find this reference helpful: https://docs.python.org/3.6/extending/building.html
    # TODO: YOUR CODE HERE
    module = Extension('numc.matrix', sources = ['matrix.c'], extra_compile_args = CFLAGS, extra_link_args = LDFLAGS)
    setup(name = 'numc', version = '1.0', description = 'This is a matrix package', ext_modules = [module])

if __name__ == "__main__":
    main()


