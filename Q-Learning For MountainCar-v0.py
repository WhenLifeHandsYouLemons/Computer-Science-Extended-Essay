# https://github.com/viniciusenari/Q-Learning-and-SARSA-Mountain-Car-v0/blob/main/train_qlearning.py for part of the code
"""
Imports & Initialise
"""
import numpy as np
import gym

# Initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()


"""
VARIABLES
"""
# Runs per trial
runs_per_trials = 5
# Number of trials
trials = 5
# Maximum episodes allowed
maximum_episodes = 2000
# How much does it care about what it just learnt
learning_rate = 0.5
# How much does it care about the future
discount_rate = 0.85
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
total_runs = runs_per_trials * trials
while runs < total_runs:
    # Reset variables
    lr = learning_rate
    dr = discount_rate
    ga = greedy_action
    mg = minimum_greed
    gdf = greediness_decrease_factor

    print(f"\nRun number: {runs+1}")

    ## Initialisation
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

    # Run Q-learning algorithm
    while running_win != consecutive_wins:
        ## Initialise state S by resetting the environment
        # Initialize parameters
        done = False
        state = env.reset()

        # Discretize "simplify" state
        state_adj = (state - env.observation_space.low) * np.array([10, 100])
        state_adj = np.round(state_adj, 0).astype(int)

        while not done:
            ## Choose action A from S using epsilon-greedy policy derived from Q
            # Determine next action - epsilon greedy strategy
            if np.random.random() < ga:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = np.argmax(Q[state_adj[0], state_adj[1]])

            ## Take action A, then observe reward R and next state S'
            # Get next state and reward
            state2, reward, done, info = env.step(action)

            # Discretize "simplify" state2
            state2_adj = (state2 - env.observation_space.low) * np.array([10, 100])
            state2_adj = np.round(state2_adj, 0).astype(int)

            # If the agent takes too long
            if done and state2[0] < 0.5:
                running_win = 0

            # If the agent reaches the goal
            if done and state2[0] >= 0.5:
                running_win += 1
                print(f"This agent won {running_win} times in a row!")

            # Updating the Q-value in the Q-table using the Bellman equation
            Q[state_adj[0], state_adj[1], action] = Q[state_adj[0], state_adj[1], action] + lr * (reward + dr * np.max(Q[state2_adj[0], state2_adj[1]]) - Q[state_adj[0], state_adj[1],action])

            # Make next state and action as current state and action
            state_adj = state2_adj

        # Decay epsilon
        if ga > mg:
            ga -= reduction

        if (total_episodes+1) % 100 == 0:
            print(f"Episode {total_episodes+1}")

        if total_episodes > maximum_episodes:
            running_win = consecutive_wins

        total_episodes += 1

    env.close()
    episodes.append(total_episodes)
    runs += 1

"""
Print Final Results
"""
print("\nThe number of episodes per run:")

i = 0
avg_episodes = []
while i != len(episodes):
    j = 0
    avg = 0
    while j != runs_per_trials - 1:
        avg += episodes[i+j]
        j += 1
    avg /= 5

    print(f"Run {round(i / runs_per_trials) + 1}: {avg} episodes")
    avg_episodes.append(avg)

    i += runs_per_trials

print(f"\nAverage out of {total_runs} runs: {np.mean(avg_episodes)} episodes\n\n\n\n\n")
