from setuptools import setup, find_packages


setup(name='torchcrf',
      version='0.0.1',
      description='Conditional random field in PyTorch',
      long_description='',
      url='https://github.com/yumoh/torchcrf.git',
      author='yumohc',
      author_email='yumohc@163.com',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      keywords='torch',
      packages=['torchcrf'],
      python_requires='>=3.6, <4')
