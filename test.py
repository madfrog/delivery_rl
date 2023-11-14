# coding=utf-8
import gym

if __name__=="__main__":
    env = gym.make('Taxi-v3')
    env.reset()
    env.render()
    
    
