"""
Asset selling model class
Adapted from code by Donghun Lee (c) 2018

"""
from collections import namedtuple
import numpy as np

class AssetSellingModel():
    """
    Base class for model
    """

    def __init__(self, state_variable, decision_variable, state_0, exog_info_fn=None, transition_fn=None,
                 objective_fn=None, seed=20180529):
        """
        Initializes the model

        :param state_variable: list(str) - state variable dimension names
        :param decision_variable: list(str) - decision variable dimension names
        :param state_0: dict - needs to contain at least the information to populate initial state using state_names
        :param exog_info_fn: function - calculates relevant exogenous information
        :param transition_fn: function - takes in decision variables and exogenous information to describe how the state
               evolves
        :param objective_fn: function - calculates contribution at time t
        :param seed: int - seed for random number generator
        """

        self.initial_args = {seed: seed}
        self.prng = np.random.RandomState(seed)
        self.initial_state = state_0
        self.state_variable = state_variable
        self.decision_variable = decision_variable
        self.State = namedtuple('State', state_variable)
        self.state = self.build_state(state_0)
        self.Decision = namedtuple('Decision', decision_variable)
        self.objective = 0.0

    def build_state(self, info):
        """
        this function gives a state containing all the state information needed

        :param info: dict - contains all state information
        :return: namedtuple - a state object
        """
        return self.State(*[info[k] for k in self.state_variable])

    def build_decision(self, info):
        """
        this function gives a decision

        :param info: dict - contains all decision info
        :return: namedtuple - a decision object
        """
        return self.Decision(*[info[k] for k in self.decision_variable])


    def exog_info_fn(self):
        """
        this function gives the exogenous information that is dependent on a random process (in the case of the the asset
        selling model, it is the change in price)

        :return: dict - updated price
        """
        # we assume that the change in price is normally distributed with mean 0 and variance 1
        updated_price = self.state.price + self.prng.normal(0, 1)
        # we account for the fact that asset prices cannot be negative by setting the new price as 0 whenever the
        # random process gives us a negative price
        new_price = 0.0 if updated_price < 0.0 else updated_price
        return {"price": new_price}

    def transition_fn(self, decision, exog_info):
        """
        this function takes in the decision and exogenous information to update the state

        :param decision: namedtuple - contains all decision info
        :param exog_info: any exogenous info (in this asset selling model,
               the exogenous info does not factor into the transition function)
        :return: dict - updated resource
        """
        new_resource = 0 if decision.sell is 1 else self.state.resource
        return {"resource": new_resource}

    def objective_fn(self, decision, exog_info):
        """
        this function calculates the contribution, which depends on the decision and the price

        :param decision: namedtuple - contains all decision info
        :param exog_info: any exogenous info (in this asset selling model,
               the exogenous info does not factor into the objective function)
        :return: float - calculated contribution
        """
        sell_size = 1 if decision.sell is 1 and self.state.resource != 0 else 0
        obj_part = self.state.price * sell_size
        return obj_part

    def step(self, decision):
        """
        this function steps the process forward by one time increment by updating the sum of the contributions, the
        exogenous information and the state variable

        :param decision: namedtuple - contains all decision info
        :return: none
        """
        exog_info = self.exog_info_fn()
        self.objective += self.objective_fn(decision, exog_info)
        exog_info.update(self.transition_fn(decision, exog_info))
        self.state = self.build_state(exog_info)

# unit testing
if __name__ == "__main__":
    # this is an example of creating a model, using a random policy, and running until the resource hits 0.
    state_variable = ['price', 'resource']
    initial_state = {'price': 10.0,
                  'resource': 1}
    decision_variable = ['sell', 'hold']
    M = AssetSellingModel(state_variable, decision_variable, initial_state)
    t = 0

    # the process continues until we sell the resource
    while M.state.resource != 0:
        # implement a random decision policy
        if np.random.uniform() > 0.8:
            decision = {'sell': 1, 'hold': 0}
        else:
            decision = {'sell': 0, 'hold': 1}
        x = M.build_decision(decision)
        print("t={}, obj={}, state.resource={}, state.price={}, x={}".format(t, M.objective, M.state.resource, M.state.price, x))
        M.step(x)
        # increment time
        t += 1

    print("t={}, obj={}, s.resource={}".format(t, M.objective, M.state.resource))

    pass