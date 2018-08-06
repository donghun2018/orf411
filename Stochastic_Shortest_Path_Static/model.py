"""
Sequential Decision Model base class

Donghun Lee (c) 2018

"""
from collections import namedtuple

import numpy as np


class Model():
    """
    Base class for model
    """

    def __init__(self, state_names, x_names, s_0, exog_info_fn=None, transition_fn=None, objective_fn=None, seed=20180529):
        """
        Initializes the model

        :param state_names: list(str) - state variable dimension names
        :param x_names: list(str) - decision variable dimension names
        :param s_0: dict - need to contain at least information to populate initial state using s_names
        :param exog_info_fn: function -
        :param transition_fn: function -
        :param objective_fn: function -
        :param seed: int - seed for random number generator
        """

        self.init_args = {seed: seed}
        self.prng = np.random.RandomState(seed)
        self.init_state = s_0
        self.state_names = state_names
        self.x_names = x_names
        self.State = namedtuple('State', state_names)
        self.state = self.build_state(s_0)
        self.Decision = namedtuple('Decision', x_names)
        self.obj = 0.0

    def build_state(self, info):
        return self.State(*[info[k] for k in self.state_names])

    def build_decision(self, info):
        return self.Decision(*[info[k] for k in self.x_names])

    def exog_info_fn(self, decision):
        return {"price": self.state.price + self.prng.normal(0, 1)}

    def transition_fn(self, decision, exog_info):
        new_resource = 0 if decision.sell is True else self.state.resource
        return {"resource": new_resource}

    def objective_fn(self, decision, exog_info):
        sell_size = 1 if decision.sell is True and self.state.resource != 0 else 0
        obj_part = self.state.price * sell_size
        return obj_part

    def step(self, decision):
        exog_info = self.exog_info_fn(decision)
        exog_info.update(self.transition_fn(decision, exog_info))
        self.obj += self.objective_fn(decision, exog_info)
        self.state = self.build_state(exog_info)


if __name__ == "__main__":
    # this is an example of creating a model, use a random policy, and run until resource hits 0.

    state_names = ['price', 'resource']
    init_state = {'price': 10.0,
                  'resource': 1}
    decision_names = ['sell']

    M = Model(state_names, decision_names, init_state)

    t = 0
    while M.state.resource != 0:
        decision = True if np.random.uniform() > 0.8 else False   # this is where your policy will be called
        x = M.build_decision({'sell': decision})
        print("t={}, obj={}, s.resource={}, s.price={}, x={}".format(t, M.obj, M.state.resource, M.state.price, x))
        M.step(x)
        t += 1
    print("t={}, obj={}, s.resource={}".format(t, M.obj, M.state.resource))

    pass