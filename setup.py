from setuptools import setup

setup(
    name='data_processing',
    version='0.0.0',

    url='https://github.com/GraysonWalaris/data_processing',
    author='Grayson Byrd',
    author_email='grayson.byrd@walaris.com',

    py_modules=['data_processing'],

    install_requires=[
        'numpy',
        'opencv-python',
        'matplotlib',
        'segment_anything'
    ]
)