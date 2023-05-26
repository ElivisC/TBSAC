import random
# from random import random

import numpy as np

from gym import Env, spaces
import pathlib

import time
import datetime
from engines.engine import BaseEngine
from engines.real_engine import AcceleratorEngine
from engines.tracewin_engine import TraceWinEngine

BPM_MIN = -10
BPM_MAX = 10
MAGNET_DELTA_MAX = 1
MAGNET_DELTA_MIN = -1

#TODO: IMPORTANT!!!   When applied to real accelerator, you need to confirm the max current strength of the magnets.
MAGNET_MAX = 40
MAGNET_MIN = -40


def remove_bpm_x(func):
    def wrapper(cls, bpm_observation, *args):
        # print(bpm_observation)
        if type(bpm_observation[0]) == tuple:
            bpm_observation = [bpm_ob[1] for bpm_ob in bpm_observation]
        return func(cls, bpm_observation, *args)

    return wrapper


class OrbitCorrectionEnv(Env):
    def __init__(self, magnet_num,
                 bpm_position,
                 engine: BaseEngine,
                 max_step=10,
                 error_mode=False,
                 error_element=[],
                 error_rate=0.01,
                 target_accuracy=0.5,
                 max_error=0):
        '''
        :param magnet_num: 磁铁数量
        :param bpm_position:  BPM的位置列表
        :param engine:  使用哪个引擎进行计算（tracewin，nn，在线加速器）
        :param max_step:  最大的步数
        :param error_mode:  是否引入误差模式
        :param error_element:  引入误差的元件列表，需要匹配conts中的元件列表
        :param error_rate:   误差率
        :param target_accuracy:  希望所有bpm的数值小于多少后结束一轮，默认为0.5
        '''
        self.name = "OrbitCorrectionEnv"
        self.bpm_position = bpm_position
        self.bpm_num = len(self.bpm_position)
        self.bpm_last_delta = np.zeros(self.bpm_num)
        self.bpm_cur_delta = np.zeros(self.bpm_num)
        self.magnet_num = magnet_num
        self.target_accuracy = target_accuracy

        self.magnet_param = np.zeros(self.magnet_num)
        self.bpm_cur_observation = []
        self.bpm_last_observation = []
        self.last_distance_delta = 0
        self.distance_delta = 0

        self._max_episode_steps = max_step

        self.error_mode = error_mode
        self.error_element = error_element
        self.error_rate = error_rate
        base_dir = pathlib.Path("..").absolute()
        self.tracewin_path = base_dir / "CAFeII_MEBTV13"
        self.bpm_engine = engine
        self.observation_space = spaces.Box(low=BPM_MIN, high=BPM_MAX, shape=(self.bpm_num * 3 * 2,),
                                            dtype=np.float32)
        # observation_space 的x方向和y方向，所以要乘以2，同时观察读数和delta所以再乘以2
        self.action_space = spaces.Box(low=MAGNET_DELTA_MIN, high=MAGNET_DELTA_MAX, shape=(self.magnet_num,),
                                       dtype=np.float32)
        # self.action_space = spaces.MultiDiscrete([ 3 ]*10)
        # print(self.action_space.sample()-1)
        self.solved_counter = 0
        self.step_counter = 0
        self.max_error = max_error

        self.bpm_max = 0
        self.bpm_min = 0
        self.first_bpm_max = 0
        self.first_bpm_min = 0
        # dtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # self.magnet_data_file = open(f"magnet_data_{dtime}.txt","a")
        # self.bpm_data_file = open(f"bpm_data_{dtime}.txt","a")
        # random.seed(20)

    def caculate_rms(self,bpm_observation):
        return np.sqrt(np.sum(np.array(bpm_observation) ** 2) / len(bpm_observation))


    def is_task_complete(self, bpm_observation):
        rms = self.caculate_rms(bpm_observation)
        if rms < self.target_accuracy:
            return True
        else:
            return False


    @remove_bpm_x
    def mse_reward(self, bpm_observation, bpm_delta_abs):  # 第2个奖励函数实验  ,delta reward 系数设置为0.5 收敛了,0.3也收敛
        rms = -self.caculate_rms(bpm_observation)
        delta_reward = 0
        for b_delta in bpm_delta_abs.tolist():
            if b_delta < 0:
                delta_reward += 1 / len(bpm_delta_abs.tolist())
            else:
                delta_reward += -4 / len(bpm_delta_abs.tolist())

        delta_reward = delta_reward / len(bpm_observation)
        reward = rms + 0.4*delta_reward
        return reward

    def step(self, magnet_param_delta):
        done = False
        self.bpm_last_delta = self.bpm_cur_delta
        # print(f"magnet_param_delta = {magnet_param_delta}")
        self.bpm_last_observation = np.array(self.bpm_cur_observation).copy()
        # print(f"magnet_param_delta = {len(np.squeeze(magnet_param_delta))}")
        self.magnet_param += (np.squeeze(magnet_param_delta) * MAGNET_DELTA_MAX)
        self.magnet_param = np.clip(self.magnet_param, MAGNET_MIN, MAGNET_MAX)
        # self.magnet_param = magnet_param
        if isinstance(self.bpm_engine, AcceleratorEngine):
            status = self.bpm_engine.execute(self.magnet_param)
            if not status:
                raise Exception("参数设置失败")

        else:
            self.bpm_engine.execute(self.magnet_param)
        self.bpm_cur_observation = self.bpm_engine.get_output(direction=["X","Y"])
        if self.bpm_cur_observation is None:
            return None, None, None, None

        if type(self.bpm_cur_observation[0]) == tuple:
            self.bpm_cur_observation = [bpm[1] for bpm in self.bpm_cur_observation]

        self.bpm_cur_delta = np.array(self.bpm_cur_observation) - np.array(self.bpm_last_observation)
        bpm_delta_abs = np.abs(np.array(self.bpm_cur_observation)) - np.abs(np.array(self.bpm_last_observation))

        print(f"magnet_param = {[round(m, 3) for m in self.magnet_param]}")
        print(f"bpm_status = {[round(b, 3) for b in self.bpm_cur_observation]}")
        reward = self.mse_reward(self.bpm_cur_observation, bpm_delta_abs)
        if self.is_task_complete(self.bpm_cur_observation):  # 每个bpm的示数都小于0.5即可停止矫正
            reward += 3*(self._max_episode_steps - self.step_counter)  # 剩余的步数也作为奖励的一部分，调好的越快分数越高
            reward += 50
            done = True
            self.step_counter = 0
        elif self.step_counter + 1 >= self._max_episode_steps:
            done = True
            self.step_counter = 0
        else:
            self.step_counter += 1

        return np.hstack((self.bpm_last_delta, self.bpm_cur_delta, self.bpm_cur_observation)), reward, done, {}

    def reset(self):
        self.round_reward_list = []
        self.magnet_param = np.zeros(self.magnet_num)
        self.magnet_param = self.magnet_param.tolist()
        print("before reset error")
        if isinstance(self.bpm_engine, TraceWinEngine):
            print("reset error")

            self.bpm_engine.reset_error()
        print("after reset error")
        self.bpm_engine.execute(self.magnet_param,debug=False)
        # exit()
        self.bpm_last_observation = self.bpm_engine.get_output(direction=["X","Y"])
        if self.bpm_last_observation is None:
            return None, None, None, None


        self.magnet_param = np.zeros(self.magnet_num) + 0.3

        self.bpm_engine.execute(self.magnet_param)
        self.bpm_cur_observation = self.bpm_engine.get_output(direction=["X","Y"])
        if self.bpm_cur_observation is None:
            return None,None,None,None

        self.bpm_last_delta = np.zeros(len(self.bpm_cur_observation))

        if type(self.bpm_cur_observation[0]) == tuple:
            self.bpm_cur_observation = [bpm[1] for bpm in self.bpm_cur_observation]

        if type(self.bpm_last_observation[0]) == tuple:
            self.bpm_last_observation = [bpm[1] for bpm in self.bpm_last_observation]
        print(f"zero bpm observation = {[round(b, 3) for b in self.bpm_last_observation ]} ")
        maax = max(self.bpm_last_observation)
        miin = min(self.bpm_last_observation)
        if self.bpm_max < maax:
            self.bpm_max = round(maax,3)
        if self.bpm_min > miin:
            self.bpm_min = round(miin, 3)
        print(f"bpm max = {self.bpm_max}, bpm min = {self.bpm_min}")

        if self.bpm_last_observation[0] > self.first_bpm_max:
            self.first_bpm_max = self.bpm_last_observation[0]
        if self.bpm_last_observation[0] < self.first_bpm_min:
            self.first_bpm_min = self.bpm_last_observation[0]
        print(f"first_bpm_min max = {self.first_bpm_max}, first_bpm_min min = {self.first_bpm_min}")
        self.bpm_cur_delta = np.array(self.bpm_cur_observation) - np.array(self.bpm_last_observation)
        bpm_observation = np.array(self.bpm_cur_observation)

        print(f"init magnet_param = {[round(m, 3) for m in self.magnet_param]} ")
        print(f"init bpm observation = {[round(b, 3) for b in bpm_observation]} ")

        return np.hstack((self.bpm_last_delta, self.bpm_cur_delta, self.bpm_cur_observation))
