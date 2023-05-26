import os
import pathlib
import argparse
import pickle

import numpy as np
import torch

from brain.SAC import SAC
from engines.real_engine import AcceleratorEngine
from engines.tracewin_engine import TraceWinEngine
from envs.orbit_correction_env import OrbitCorrectionEnv
from utils.consts import Consts
from utils.particle import PreCreatedParticles
from utils.replaybuffer import ReplayBuffer


def args_config():
    parser = argparse.ArgumentParser(description='TBSAC beam control')
    # 给这个解析对象添加命令行参数
    parser.add_argument('--device', type=str, default='cpu', choices=('cpu', 'gpu'), help='')
    parser.add_argument('--engine', type=str, default='tracewin', choices=('tracewin', 'accelerator'), help='')
    parser.add_argument('--mode', type=str, default='train', choices=('train', 'test'))
    parser.add_argument('--random_steps', type=int, default=500, help="Steps of random data collection.")
    parser.add_argument('--max_steps', type=int, default=20000, help="Max steps of training process.")
    parser.add_argument('--evaluated_freq', type=int, default=1000, help="Evaluated frequency.")
    parser.add_argument('--evaluated_times', type=int, default=5, help="Evaluation times.")
    parser.add_argument('--model_save_path', type=str, default='trained_model')
    parser.add_argument('--model_index', type=int, default=10000, help="The model you want to load.")
    args = parser.parse_args()  # 获取所有参数
    return args


def evaluate_policy(args, model_index, env, seed):
    times = args.evaluated_times
    env.action_space.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = SAC(state_dim, action_dim, max_action, use_cnn=True)
    agent.load(f"{BASE_DIR}/trained_models_{particle.name}/sc_orbit_{model_index}")
    evaluate_reward = 0
    init_rms_list = []
    final_rms_list = []
    for index in range(times):
        s = env.reset()
        init_rms = np.sqrt(np.sum(np.array(env.bpm_last_observation) ** 2) / len(env.bpm_last_observation))
        print(f"init RMS ============== {init_rms}")
        done = False
        episode_steps = 0
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
            episode_steps += 1
        evaluate_reward += episode_reward
        rms = np.sqrt(np.sum(np.array(env.bpm_cur_observation) ** 2) / len(env.bpm_cur_observation))
        init_rms_list.append(init_rms)
        final_rms_list.append(rms)
        print(
            f"================Round {index + 1}:DONE within {episode_steps} steps, init RMS = {init_rms}, final RMS = {rms}, reward = {evaluate_reward}==========================")

    return int(evaluate_reward / times)


def train_policy(args, env, env_evaluate, seed):
    env.action_space.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    # print("env={}".format(env_name[env_index]))
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_action={}".format(max_action))
    print("max_episode_steps={}".format(max_episode_steps))

    agent = SAC(state_dim, action_dim, max_action, use_cnn=True)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    trained_rewards = []
    max_train_steps = args.max_steps  # Maximum number of training steps
    random_steps = args.random_steps  # Take the random actions in the beginning for the better exploration
    evaluate_freq = args.evaluated_freq  # Evaluate the policy every 'evaluate_freq' steps
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    model_path = f"{BASE_DIR}/trained_models_{particle.name}_seed{seed}"
    print(model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    while total_steps < max_train_steps:
        s = env.reset()
        episode_steps = 0
        done = False
        episode_reward = 0
        init_rms = np.sqrt(np.sum(np.array(env.bpm_last_observation) ** 2) / len(env.bpm_last_observation))
        while not done:
            episode_steps += 1
            if total_steps < random_steps:  # Take the random actions in the beginning for the better exploration
                a = env.action_space.sample()
            else:
                a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)
            episode_reward += r
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != max_episode_steps:
                dw = True
            else:
                dw = False
            replay_buffer.store(s, a, r, s_, dw)  # Store the transition
            s = s_

            if total_steps >= random_steps:
                agent.learn(replay_buffer)

            # Evaluate the policy every 'evaluate_freq' steps
            if (total_steps + 1) % evaluate_freq == 0:
                agent.save(f"{model_path}/sc_orbit_{total_steps + 1}")

            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                evaluate_rewards.append(evaluate_reward)

                print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))

                pickle.dump(evaluate_rewards, open(f"{model_path}/evalute.rewards.pkl", 'wb'))

            total_steps += 1
        rms = np.sqrt(np.sum(np.array(env.bpm_cur_observation) ** 2) / len(env.bpm_cur_observation))
        print(f"================DONE init step={episode_steps} RMS = {init_rms} RMS = {rms}==========================")
        print(f"Step:{total_steps} , Total reward : {episode_reward}")
        trained_rewards.append(episode_reward)
    pickle.dump(trained_rewards, open(f"{model_path}/trained_rewards.pkl", 'wb'))


if __name__ == "__main__":
    args = args_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BASE_DIR = pathlib.Path(".").absolute()
    tracewin_path = BASE_DIR / "CAFeII_SC_Proton"  # TODO: Change this to your own dir
    os.chdir(tracewin_path)
    seed = 10
    model_index = 10000  # The model you want to load to evaluate the policy

    MAGNET_NUM = 10
    bpm_position = [4.1557200, 4.7957200, 5.4357200, 6.0757200, 6.71572]
    particle = PreCreatedParticles.Ar_CM1
    if args.engine == 'tracewin':
        engine = TraceWinEngine(
            particle=particle,
            bpm_position=bpm_position,
            tracewin_path=tracewin_path,
            ini_name="SC_Ca.ini",
            parent_template_name="sc_template.dat",
            son_template_name="sc_bpm_template.dat",
            final_lattice_name="SC_Ca.dat",
            tracewin_out_path="temp",
            max_error=3,
            # During training process error_mode should be False. Enable error mode can add current strenth error to magnets
            error_mode=False,
            error_element=[Consts.SOL],
            error_rate=0.05
        )
    elif args.engine == 'accelerator':
        engine = AcceleratorEngine()

    env = OrbitCorrectionEnv(MAGNET_NUM,
                             bpm_position,
                             engine,
                             target_accuracy=0.5,
                             max_step=50)
    env_evaluate = OrbitCorrectionEnv(MAGNET_NUM,
                                      bpm_position,
                                      engine,
                                      target_accuracy=0.5,
                                      max_step=50)
    if args.mode == 'train':
        train_policy(args, env, env_evaluate, seed)
    if args.mode == 'test':
        evaluate_policy(args, model_index, env_evaluate, seed + 100)
