# coding=utf-8

from delivery_env import DeliveryEnv
import numpy as np
import random
import matplotlib.pyplot as plt

def train(env, q_table):
    total_episodes = 50000
    max_steps = 99
    
    learning_rate = 0.7
    gamma = 0.618
    
    # Exploration parameters
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.01
    epsilons = []
    
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
        epsilons.append(epsilon)
        print(f'epsilon: {epsilon}, learning_rate: {learning_rate}')
    env.render()
    return q_table, epsilons


def test(env, q_table):
    total_test_episodes = 100
    env.reset()
    rewards = []

    for episode in range(total_test_episodes):
        print('=' * 10)
        print('New episode')
        state = env.reset()
        step = 0
        total_rewards = 0
        max_steps = 99

        for step in range(max_steps):
            env.render()
            action = np.argmax(q_table[state, :])
            new_state, reward, done, info = env.step(action)
            total_rewards += reward
            if done:
                rewards.append(total_rewards)
                break
            state = new_state
    env.close()
    print(f"Score over episode: {str(sum(rewards) / total_test_episodes)}")
            

if __name__=="__main__":
   env = DeliveryEnv()
   print('Begin to train...')
   #ction_size = env.action_space.n
   #state_size = env.observation_space.n
   #print(f'action size: {action_size}, state_size: {state_size}')

   #q_table = np.zeros((state_size, action_size))
   #q_table, epsilons = train(env, q_table)
   #print(f'{q_table}')

   ## Save q_table and env
   #np.save("./q_table.npy", q_table)
   #env.freeze()

   ## plot epsilon
   #epsilon_index = np.arange(1, len(epsilons) + 1)
   #print(f'epsilon index: {epsilon_index}')
   #plt.plot(epsilon_index.tolist(), epsilons)
   #plt.ylabel('epsilon value')
   #plt.xlabel('iter index')
   #plt.show()

   print("*"*30)
   print('Begin to test...')
   q_table = np.load("./q_table.npy")
   env.reconstruct_env()
   env.render()
   test(env, q_table)