from distutils.core import setup

setup(
    name='LDADE',
    packages=['LDADE'],
    python_requires='>=2.7, < 3',
    install_requires=["numpy", "sklearn"]
)
