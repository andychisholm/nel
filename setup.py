from setuptools import setup, find_packages

__version__ = '0.4.0'
__pkg_name__ = 'nel'

setup(
    name = __pkg_name__,
    version = __version__,
    description = 'Named entity linker',
    author='Andrew Chisholm',
    packages = find_packages(),
    license = 'MIT',
    url = 'https://github.com/wikilinks/nel',
    entry_points = {
        'console_scripts': [
            'nel = nel.__main__:main'
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    install_requires = [
        "Flask",
        "Jinja2",
        "MarkupSafe",
        "Werkzeug",
        "argparse",
        "functools32",
        "itsdangerous",
        "msgpack-python",
        "numpy",
        "python-crfsuite",
        "pymongo",
        "python-dateutil",
        "redis",
        "scikit-learn",
        "scipy",
        "six",
        "wsgiref",
        "ujson",
        "progressbar2",
        "spacy"
    ],
    test_suite = __pkg_name__ + '.test'
)
