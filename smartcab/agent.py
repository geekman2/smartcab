import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q_table = {}
        self.net_reward = 0
        self.states = set()
        self.wins = 0
        self.losses = 0

    def Q(self, state, action):
        if state in self.Q_table:
            if action in self.Q_table[state]:
                return self.Q_table[state][action]
            else:
                self.Q_table[state][action] = 1
                return 1
        else:
            self.Q_table[state] = {"left": 1,
                                   "right": 1,
                                   "forward": 1,
                                   None: 1}
            return 1

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.net_reward = 0

        pd.DataFrame.from_dict(self.Q_table, orient="index").to_csv("Q.csv")

        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        gamma = 0.0
        try:
            epsilon = 1
        except ZeroDivisionError:
            epsilon = 0

        # TODO: Update state
        self.state = ["oncoming:{}".format(inputs["oncoming"]),
                      # inputs["right"],
                      "left:{}".format(inputs["left"]),
                      "light:{}".format(inputs["light"]),
                      "waypoint:{}".format(self.next_waypoint)]
        self.state = ','.join([str(v) for v in self.state])

        # TODO: Select action according to your policy
        if random.random() < epsilon and self.state in self.states:
            action = max(self.Q_table[self.state], key=self.Q_table[self.state].get)
        else:
            action = random.choice(["left", "right", "forward", None])
            self.states.add(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.net_reward += reward

        new_state = ["oncoming:{}".format(inputs["oncoming"]),
                     # inputs["right"],
                     "left:{}".format(inputs["left"]),
                     "light:{}".format(inputs["light"]),
                     "waypoint:{}".format(self.next_waypoint)]
        new_state = ','.join(str(v) for v in new_state)

        # TODO: Learn policy based on state, action, reward
        next_rewards = [self.Q(state, act) for state, act in
                        zip([new_state for x in range(3)], ["left", "right", "forward", None])]
        if self.state in self.Q_table:
            self.Q_table[self.state][action] = reward #+ gamma*max(next_rewards)
        else:
            self.Q_table[self.state] = {action: reward} #+ gamma*max(next_rewards)}



            # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline,inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5,
                    display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
