import setuptools

setuptools.setup(
    name='black-scholes-model',
    version='1.0',
    packages=['black_scholes_model'],
    install_requires=['numpy', 'scipy', 'pytest', 'pytest-cov'],
    tests_require=['pytest', 'pytest-cov'],
    entry_points={
        'console_scripts': [
            'test=black_scholes_model.test:main',
        ],
    },
)