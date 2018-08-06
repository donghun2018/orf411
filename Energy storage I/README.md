Instructions to run the energy storage model package with different policies:

1. To run the energy storage model with a random policy, simply run EnergyStorageModel.py as is (ensure that the Excel file of energy prices, PJM_Historical_DA_RT_LMPs_05_to_11.xlsx, is in the same folder as the Python files). You will be able to see the output and can experiment with changing the value of the parameter (it is set at 0.8).

2. To run the energy storage model with the buy low, sell high policy, open EnergyStoragePolicy.py, and use the function run_policy with the appropriate inputs (there is an example on line 250). Experiment with different parameters by changing the values in Sheet 1 of energy_storage_policy_parameters.xlsx.

3. In EnergyStoragePolicy.py, there are two ways to create data visualizations of the results:

	i) a heat map showing the results of a full grid search
	ii) a plot of performance as a function of the buy value for 5 selected theta_sell values

To run i): Obtain a list of theta values using the function grid_search_theta_values and input those theta values into the function vary_theta. To plot the results of vary_theta, input those results into plot_heat_map. An example is given from lines 262-273. Experiment with different lower and upper limits and increment sizes of the grid search by changing the values in Sheet 3 of energy_storage_policy_parameters.xlsx.
To run ii): Obtain a list of theta values using the function theta_buy_plot_values and input those theta values into the function vary_theta. To plot the results of vary_theta, input those results into plot_theta_buy. An example is given from lines 276-283. Similarly, you can experiment with different lower and upper limits and increment sizes by changing the values in Sheet 4 of energy_storage_policy_parameters.xlsx.

4. In TimeSeries.py, you will be able to run a very basic energy price model. Experiment with different initial values by changing the values in Sheets 5 and 6 of energy_storage_policy_parameters.xlsx.

5. In BDP.py, you will be able to run backward dynamic programming for the same energy storage model (backward dynamic programming is further detailed in section 8.4.2 in the course text in Chapter 8: Energy Storage I).
