from setuptools import setup, find_packages

__version__ = '0.1'
__pkg_name__ = 'nel'

setup(
    name = __pkg_name__,
    version = __version__,
    description = 'Named entity linker',
    packages = find_packages(),
    license = 'MIT',
    url = 'https://github.com/wikilinks/nel',
    entry_points = {
        'console_scripts': [
            'nel = nel.__main__:main'
        ]
    },
    install_requires = [
        "Flask",
        "Jinja2",
        "MarkupSafe",
        "Werkzeug",
        "argparse",
        "functools32",
        "itsdangerous",
        "libschwa-python",
        "msgpack-python",
        "numpy",
        "pymongo",
        "python-dateutil",
        "redis",
        "scikit-learn",
        "scipy",
        "six",
        "thrift",
        "wsgiref",
        "neleval"
    ],
    dependency_links = [
        "http://github.com/wikilinks/neleval/tarball/master#egg=neleval-v2.0.0"
    ],
    test_suite = __pkg_name__ + '.test'
)
