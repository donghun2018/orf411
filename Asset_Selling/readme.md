# Asset Selling Problem

Instructions for Asset Selling Problem.

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

## How To Run Each Problem Class

### Asset Selling

All codes are under Asset_Selling directory.

#### TL;DR

```
python AssetSellingPolicy.py
```

#### Code Structure

- Model: AssetSellingModel.py
- Policy: AssetSellingPolicy
  -  sell_low_policy
  - high_low_policy
  - track_policy
- Driver: AssetSellingPolicy's main function

#### Detailed Explanation

1. To run the asset selling model with a random policy, simply run AssetSellingModel.py as is. You will be able to see the output and can experiment with changing the value of the parameter (it is set at 0.8).
2. To run the asset selling model with other available policies (sell-low, high-low and track policy), open AssetSellingPolicy.py, choose your desired policy by commenting out the other policies and run the code. Experiment with different policies and parameters.
3. To run a full-grid search over theta values in the policies (as in the problem set), modify the identical code for full-grid search in EnergyStoragePolicy.py.
