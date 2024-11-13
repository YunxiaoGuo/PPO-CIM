import gym
import numpy as np
from tqdm import tqdm
from common.Agent import Agent
from argument import parameters

if __name__ == '__main__':
    envs  = ['Swimmer-v2','Reacher-v2','Hopper-v2','Humanoid-v2','Ant-v2','Walker2d-v2']
    algorithms = ["CIM", "CIM-1", "CIM-2"]
    args = parameters.get_paras()
    args.continuous = isinstance(gym.make(args.env_name).action_space, gym.spaces.Box)
    for batch in tqdm(range(args.num_epoch)):
        env = gym.make(args.env_name)
        env.seed(np.random.seed(np.random.randint(10000)))
        agent = Agent(env, args)
        reward_data = []
        for episode in tqdm(range(args.num_episode)):
            rsum = 0
            ob = env.reset()
            for time_step in range(args.max_step_per_round):
                action = agent.select_action(ob)
                new_obs, reward, done, _ = env.step(action)
                agent.store(ob, action, new_obs, reward, done)
                ob = new_obs
                rsum += reward
                if done:
                    reward_data.append(rsum)
                    break
            if agent.timeToLearn():
                agent.learn_cim()
    agent.save_data(reward_data,episode)
    env.close()
