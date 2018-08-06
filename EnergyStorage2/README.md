HOW TO RUN

## Energy Storage 2

All codes are under EnergyStorage2 directory.

### TL;DR

py EnergyStorage2DriverScript.py

### Code Structure

- Model: EnergyStorage2Model.py
- Policy: EnergyStorage2Policy.py
- Driver: EnergyStorage2DriverScript.py
- Helper class: Forecast.py

### Detailed Explanation

1. To run the energy storage model with Kesten's rule, simply run EnergyStorage2DriverScript.py as is. You will see a list with the
iteration, the profit made with current parameters and the difference in profit made with the adjusted parameters. At the end you will see the
final theta vector and the average profit of the base lookahead policy compared to the final parametrized one.
You can experiment with changing the parameters values in the parameters.xls file.
2. To run the model with a different step size rule, simply change the implementation at line 115.
3. To change around the distributions for the forecasts, go to the Forecast.py file and manually change the code starting from line 47.
There is, however, no need to do this besides experimenting for yourself.
