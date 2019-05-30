from maze_env_extend import Maze
# from maze_env import Maze
# from RL_brain import DeepQNetwork
from RL_brain_extend import DeepQNetwork
import time


def run_maze(RL, iteration):
    step = 0
    start = time.clock()
    flag = True
    for episode in range(iteration):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation)
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            if 100==reward and flag:
                flag = False
                end = time.clock()
                print('iteration:%d'%(iteration), end-start)
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1
    if flag:
        print('iteration:%d not found destination'%(iteration))
    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    # iterations = [1, 2, 5, 10, 20, 50, 100]
    iterations = [1000,2000,5000,10000,20000,50000,100000]
    for iteration in iterations:
        env = Maze()
        RL = DeepQNetwork(env.n_actions, env.n_features,
                          learning_rate=0.0001,
                          reward_decay=0.85,
                          e_greedy=0.95,
                          replace_target_iter=250, # 每replace_target_iter步替换一次q-target神经网络参数
                          memory_size=3000,
                          batch_size=32
                          # output_graph=True
                          )
        env.after(100, run_maze(RL, iteration=iteration))
        env.mainloop()
        RL.plot_cost(iteration, RL.replace_target_iter)