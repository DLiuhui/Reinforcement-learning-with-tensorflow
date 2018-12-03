"""
Reinforcement learning maze example.

"""

# from maze_env_extend import Maze
from maze_env_extend2 import Maze
from RL import QLearning
import numpy as np
import matplotlib.pyplot as plt

def transfunc(x, start, end):
    range = end - start
    if (x-start)/range > 0.5:
        return 1/(1 + (np.e**(10*(0.5-(x-start)/range))))
    else:
        return (x-start)/range

def update(RL, iteration=100):
    RL.q_func = np.zeros(iteration)
    for episode in range(iteration):
        # initial observation
        observation = env.reset()
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.chooseAction(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            # RL learn from this transition
            RL.learning(str(observation), action, str(observation_), reward, episode)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()
    plt.figure(1)
    plt.plot(np.arange(iteration), RL.q_func)
    plt.savefig("running_in_fixed_parameters.png")

if __name__ == "__main__":
    env = Maze()
    RL = QLearning(actions=list(range(env.n_actions)), learning_rate=0.05, reward_decay=0.9, e_greedy=0.9)
    iteration = 1000
    # 开始运行
    env.after(100, update(RL, iteration))
    env.mainloop()