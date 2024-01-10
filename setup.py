from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name = 'champ-dataset',
    version = '1.0.0',
    author = 'Yilun Zhou',
    author_email = 'zyilun94@gmail.com',
    description = 'Concept and Hint-Augmented Math Dataset (CHAMP)',
    license = 'MIT',
    keywords = ['natural language processing', 'large language model'],
    url = 'https://yujunmao1.github.io/CHAMP/',
    packages=['champ_dataset'],
    long_description = long_description,
    long_description_content_type='text/markdown', 
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research', 
        'Topic :: Scientific/Engineering :: Artificial Intelligence', 
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python', 
    ],
    install_requires = ['natsort'], 
    include_package_data = True, 
)