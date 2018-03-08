from distutils.core import setup

setup(
    name='LDADE',
    packages=['LDADE'],
    package_dir={'LDADE': 'LDADE'},
    package_data={'LDADE': ['data/*']},
    python_requires='>=2.7, < 3',
    install_requires=["numpy", "sklearn"]
)
