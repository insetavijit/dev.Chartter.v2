from setuptools import setup, find_packages

setup(
    name='taframework',
    version='0.1.0',
    description='Enhanced Enterprise Technical Analysis Framework with TALib integration',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'ta-lib>=0.4.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
