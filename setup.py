from setuptools import setup, find_packages

setup(
    name = 'nel',
    version = '0.1',
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
