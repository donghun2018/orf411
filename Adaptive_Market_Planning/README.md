# Adaptive Market Planning Problem

All codes are under Adaptive_Market_Planning directory.

### TL;DR

python AdaptiveMarketPlanningDriverScript.py

### Code Structure

- Model: AdaptiveMarketPlanningModel.py
- Policy: AdaptiveMarketPlanningPolicy.py
- Driver: AdaptiveMarketPlanningDriverScript.py

### Detailed Explanation

1. To run the adaptive market planning model with Kesten's rule, simply run AdaptiveMarketPlanningDriverScript.py as is. You will see a plot with time
in logarithmic units (base 10) and the quantity ordered over time. The analytical solution is also provided to assess the algorithm. You can experiment
with changing the parameters values in the Base parameters.xls file.
2. To run the model with the harmonic policy, simply change the function in Driver Script from kesten_rule to harmonic_rule in the main loop of the
simulation (line 37).
3. For the extension, try creating a subclass of the Model class and change it accordingly. The policy class will still work.
