from setuptools import setup, find_packages

setup(
    name='kgraphgnn',
    version='0.0.1',
    author='Marc Hadfield',
    author_email='marc@vital.ai',
    description='KGraph GNN',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/vital-ai/kgraph-gnn',
    packages=find_packages(exclude=["test"]),
    license='Apache License 2.0',
    install_requires=[
            'vital-ai-vitalsigns>=0.1.27',
            'vital-ai-domain>=0.1.4',
            'pyyaml',
            'vital-ai-haley-kg>=0.1.24',
            'kgraphservice>=0.0.6',
            'torch-geometric==2.6.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
