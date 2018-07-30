This is python code base for ORF 411.

## Getting Started

This section guides you how to setup your machine to run codes in this repository.

### Prerequisites

The software was developed with Python 3.6.

#### Get Python

Anaconda is an easy way to get a working python.
Get it [here](https://www.anaconda.com/download/).

The initial version of the simulator is tested on 64-bit python 3.6 as follows:

```
Python 3.6.1 | packaged by conda-forge | (default, May 23 2017, 14:21:39) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

#### Get Python Packages

The packages required by the simulator are:

- numpy: used for pseudorandom number generator and many useful functions
- openpyxl: used for Microsoft Excel xlsx input/output functionalities

You may get these using conda

```
$ conda install numpy openpyxl
```

or using pip

```
$ pip install numpy openpyxl
```

### Get the software

Download the source codes by

```
$ git clone https://github.com/donghun2018/orf411.git
```

If you do not have git, you may instead [download ZIP file from this link](https://github.com/donghun2018/orf411/archive/master.zip).


## Overall Structure

This section explains the overall code structure that is commonly found in all sequential decision making problems in this repository.

### Model

A model class is a particular modeling of a sequential decision problem.

### Policy

A policy class is a particular policy that solves a sequential decision problem.
There may be more than one policy class implemented for a single sequential decision problem.

### Driver Script

A driver script initializes policy object and model object, then runs the policy on the model.
Running a driver will generate some output, from which further tables, plots, and inferences can be made.
