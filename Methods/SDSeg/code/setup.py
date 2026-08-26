# MODIFICATION NOTICE (release prepared 2026-07-19): this file differs from
# upstream SDSeg commit 0b0aa388a5e2def75abfbef90d7bcfc5c16f2704.
# Package discovery was narrowed to the retained ldm namespace for this source
# release. See the method-root MODIFICATIONS.md.

from setuptools import setup, find_namespace_packages

setup(
    name='latent-diffusion',
    version='0.0.1',
    description='',
    packages=find_namespace_packages(include=["ldm*"]),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
