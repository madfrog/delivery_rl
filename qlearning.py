# coding=utf-8

from delivery_env import DeliveryEnv
import numpy as np
import random

def train():
    total_episodes = 500000
    total_test_episodes = 100
    max_steps = 99
    
    learning_rate = 0.7
    gamma = 0.618
    
    # Exploration parameters
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01
    
    env = DeliveryEnv()
    action_size = env.action_space.n
    state_size = env.observation_space.n
    print(f'action size: {action_size}, state_size: {state_size}')
    
    q_table = np.zeros((state_size, action_size))
    print(q_table)

    for episode in range(total_episodes):
        state = env.reset()
        #env.render()
        step = 0
        done = False

        for step in range(max_steps):
            # Choose an action
            tradeoff = random.uniform(0, 1)
            if tradeoff > epsilon:
                action = np.argmax(q_table[state, :])
            else:
                action = env.action_space.sample()

            # Take the action and get its next state and reward
            new_state, reward, done, info = env.step(action)

            # Update Q(s, a) := Q(s, a) + lr[R(s, a) + gamma * max Q(s', a') - Q(s, a)]
            q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
            #print(f'q_table[state, action]: {q_table[state, action]}')

            state = new_state

            if done:
                break
            # env.render()
        print(f'finish episode: {episode}')
        # env.render()
        epsilon = min_epsilon + (max_epsilon - min_epsilon) *np.exp(-decay_rate * episode)
    env.render()
    return q_table
            

if __name__=="__main__":
   # env = DeliveryEnv()
   # env.reset()
   # env.render()
   q_table = train()
   print(f'{q_table}')