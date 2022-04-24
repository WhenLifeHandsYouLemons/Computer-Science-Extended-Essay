"""
Imports & Initialise
"""
# import time
import numpy as np
import gym
# import matplotlib.pyplot as plt

# Initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()


"""
VARIABLES
"""
# To get an average as it varies largely
average_runs = 25
# How much does it care about what it just learnt
learning_rate = 0.5
# How much does it care about the future
discount_rate = 0.3
# How greedy is the agent
greedy_action = 0.25
# Until how much does the greediness decrease
minimum_greed = 0
# How much is the greediness factor divided by
greediness_decrease_factor = 1
# How many consecutive wins must the agent get in order to consider it optimal
consecutive_wins = 3


"""
Run Q-Learning Algorithm
"""
print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
runs = 0
episodes = []
while runs < average_runs:
    # Reset variables
    lr = learning_rate
    dr = discount_rate
    ga = greedy_action
    mg = minimum_greed
    gdf = greediness_decrease_factor

    print(f"Run number: {runs+1}")

    # Determine size of discretized state space
    num_states = (env.observation_space.high - env.observation_space.low) *\
        np.array([10, 100])
    num_states = np.round(num_states, 0).astype(int) + 1

    # Initialize Q table
    Q = np.random.uniform(low=-1, high=1, size=(num_states[0], num_states[1], env.action_space.n))

    # Calculate episodic reduction in epsilon
    reduction = (ga - mg) / gdf

    state2 = [0]
    total_episodes = 0
    running_win = 0

    # Run Q learning algorithm
    while state2[0] < 0.5 or running_win != consecutive_wins:
        # Initialize parameters
        done = False
        state = env.reset()

        # Discretize "simplify" state
        state_adj = (state - env.observation_space.low) * np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)

        while not done:
            # Determine next action - epsilon greedy strategy
            if np.random.random() < 1 - ga:
                action = np.argmax(Q[state_adj[0], state_adj[1]])
            else:
                action = np.random.randint(0, env.action_space.n)

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            # Discretize "simplify" state2
            state2_adj = (state2 - env.observation_space.low) * np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)

            # End agent if they take too long
            if done and state2[0] < 0.5:
                running_win = 0

            # End agent if they reach the goal
            if done and state2[0] >= 0.5:
                running_win += 1
                print(f"This agent won {running_win} times in a row!")
                Q[state_adj[0], state_adj[1], action] = reward

            # Adjust Q value for current state
            else:
                # Updating the Q-value in the Q-table using the Bellman equation
                Q[state_adj[0], state_adj[1], action] = Q[state_adj[0], state_adj[1], action] + lr * (reward + dr * np.max(Q[state2_adj[0], state2_adj[1]]) - Q[state_adj[0], state_adj[1],action])

            # Make next state and action as current state and action
            state_adj = state2_adj

        # Decay epsilon
        if ga > mg:
            ga -= reduction

        if (total_episodes+1) % 100 == 0:
            print(f"Episode {total_episodes+1}")

        total_episodes += 1

    env.close()
    print()
    episodes.append(total_episodes)
    runs += 1

print("\nThe number of episodes per run:")
i = 0
while i != len(episodes):
    avg = (episodes[i] + episodes[i+1] + episodes[i+2] + episodes[i+3] + episodes[i+4]) / 5
    print(f"Run {round(i/5) + 1}: {avg} episodes")
    i += 5
print(f"\nAverage out of {average_runs} runs: {np.mean(episodes)} episodes")
