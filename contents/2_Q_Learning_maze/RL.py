import numpy as np
import pandas as pd

PRINT_BUTTON = False
class QLearning:
    def __init__(self, actions, learning_rate=0.25, reward_decay=0.6, e_greedy=0.5):
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float32)
        self.q_func = None

    def chooseAction(self, observation):
        self.checkState(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # 选择权值最大的动作
            if PRINT_BUTTON:
                print(self.q_table.loc[observation])
            state_action = self.q_table.loc[observation]
            action = np.random.choice(state_action[state_action==np.max(state_action)].index)
        else:
            # 随机挑选动作,进行探索
            action = np.random.choice(self.actions)
        return action

    def learning(self, old_s, old_a, new_s, reward, episode):
        self.checkState(new_s)
        q_predict = self.q_table.loc[old_s,old_a] # Q(s,a)
        # 检查是否到达终点
        if 'terminal' == new_s:
            q_target = reward # 到达停止点，没有下一步，self.gamma * 0
        else:
            q_target = reward + self.gamma * self.q_table.loc[new_s].max()
        self.q_table.loc[old_s,old_a] = q_predict + self.alpha * (q_target - q_predict)
        self.q_func[episode] += (q_target - q_predict) ** 2

    def checkState(self, state): # 检查所在环境是否是新的环境
        if state not in self.q_table.index:
            # 如果是新的环境
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )