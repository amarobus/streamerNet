from setuptools import setup, find_packages

setup(
    name='streamernet',
    version='0.1.0',
    description='',
    author='',
    author_email='amarobus@gmail.com',
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
    ],
    test_suite='tests',
    tests_require=[
        'pytest',
    ],
)