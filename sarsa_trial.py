import time
import gym
import numpy as np

env = gym.make('FrozenLake-v1')

epsilon = 0.9
# min_epsilon = 0.1
# max_epsilon = 1.0
# decay_rate = 0.01

total_episodes = 10000
max_steps = 100

lr_rate = 0.81
gamma = 0.96

Q = np.zeros((env.observation_space.n, env.action_space.n))

def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

def learn(state, state2, reward, action, action2):
    predict = Q[state, action]
    target = reward + gamma * Q[state2, action2]
    Q[state, action] = Q[state, action] + lr_rate * (target - predict)

# Start
start_time = time.time()
rewards=0

for episode in range(total_episodes):
    t = 0
    state = env.reset()
    action = choose_action(state)

    while t < max_steps:
        # Show the episode number
        # print(f"Episode: {episode}")
        env.render()

        state2, reward, done, info = env.step(action)
        print(action)

        action2 = choose_action(state2)

        learn(state, state2, reward, action, action2)

        state = state2
        action = action2

        t += 1
        rewards+=1

        if done:
            break

        # epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode) 
		# time.sleep(0.1)

end_time = time.time()
time_lapsed = end_time - start_time
print(f"Time taken for {total_episodes} episodes: {round(time_lapsed, 0)} seconds.")

print(f"""
{round(Q[0].sum(), 1)} {round(Q[1].sum(), 1)} {round(Q[2].sum(), 1)} {round(Q[3].sum(), 1)}
{round(Q[4].sum(), 1)} {round(Q[5].sum(), 1)} {round(Q[6].sum(), 1)} {round(Q[7].sum(), 1)}
{round(Q[8].sum(), 1)} {round(Q[9].sum(), 1)} {round(Q[10].sum(), 1)} {round(Q[11].sum(), 1)}
{round(Q[12].sum(), 1)} {round(Q[13].sum(), 1)} {round(Q[14].sum(), 1)} {round(Q[15].sum(), 1)}
""")

print("\nThe final Q-table:")
print(Q)

print ("Score over time: ", rewards/total_episodes)
