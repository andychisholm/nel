from setuptools import setup, find_packages
from nel import __pkg_name__, __version__

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
        "wsgiref"
    ],
    test_suite = __pkg_name__ + '.test'
)
