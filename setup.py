from setuptools import setup, find_packages


setup(
    name='LDADE',

    # TODO: NEED TO BE CHANGED LATER
    version='2018.03.27',

    author='Amritanshu Agrawal',
    author_email='aagrawa8@ncsu.edu',

    # TODO: NEED TO BE CHANGED LATER
    url='https://github.com/tonyfloatersu/LDADE-python-package',
    license='MIT',
    include_package_data=True,

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],

    packages=find_packages(),
    package_dir={'LDADE': 'LDADE'},
    package_data={'LDADE': ['data/*']},

    python_requires='>=2.7, < 3',
    install_requires=["numpy", "sklearn"]
)
