# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

from hmac import new
import time

from scipy.stats import kendalltau, spearmanr
import logging
import gym

import numpy as np
import torch

import time
import warnings
warnings.filterwarnings("ignore")

from utils import calculation_metrics
from models.attention_model_wrapper import Agent
from wrappers.syncVectorEnvPomo import SyncVectorEnv
from wrappers.syncVectorEnvPomo import SyncVectorEnv
from wrappers.recordWrapper import RecordEpisodeStatistics

from utils import save_merged_data

import pdb
import pandas as pd
from scipy.stats import kendalltau

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = './runs/container-v0__ppo_or__2025-11-27_20_10/ckpt/270.pt' #
    
    agent = Agent(device=device, name='container').to(device)
    # agent.load_state_dict(torch.load(ckpt_path))
    agent.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
    
    env_id = 'container-v0'
    num_steps = 51
    num_envs = 10
    n_traj = 1

    env_entry_point = 'envs.container_vector_env:ContainerVectorEnv'
    seed = 10

    gym.envs.register(
        id=env_id,
        entry_point=env_entry_point,
    )

    def make_env(env_id, seed, cfg={}):
        def thunk():
            env = gym.make(env_id, **cfg)
            env = RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    envs = SyncVectorEnv(  [make_env(env_id, seed, dict(mode="test",index=i)) for i in range(num_envs) ]  )

    
    trajectories = []
    data_keys = []
    agent.eval()
    obs = envs.reset()
    valid_node = obs['valid_mask']
    data_keys = obs['data_key']     # num_envs,

    envs.reset()
    for step in range(0, num_steps):
        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logits = agent(obs)
        obs, reward, done, info = envs.step(action.cpu().numpy())
        trajectories.append(action.cpu().numpy())

    resulting_traj = np.array(trajectories).transpose(1, 2, 0)  # (env,traj,step)
        
    save_merged_data(obs, resulting_traj, data_keys, valid_node)
    
    """
    trajectories : Step,(env,traj)
    episode_returns : (env,traj)
    test_obs['observations'] : (env, node,obs_dim)
    """
    
    target  = np.concatenate([  np.arange(resulting_traj.shape[-1] -1), [0]  ])
    target = np.tile(target, (resulting_traj.shape[0], resulting_traj.shape[1], 1))  #(env,traj,step)
   
    # tau, _ = kendalltau(target, resulting_traj)
    tau_list = []
    rho_list = []
    for i in range(resulting_traj.shape[0]):
        # 调整从0开始的索引
        arr = resulting_traj[i,0,:-1]
        zero_index = np.where(arr == 0)[0][0]
        new_arr = np.concatenate([arr, arr])[zero_index : zero_index + len(arr)+1]
        resulting_traj[i,0] = new_arr
        
        tau, _ = kendalltau(target[i,0,], resulting_traj[i,0,])
        rho, _ = spearmanr(target[i,0,], resulting_traj[i,0,])

        # print(tau, rho)
        # if valid_node[i] < resulting_traj.shape[-1]:
        #     valid_mask = resulting_traj[i, 0, :] <= valid_node[i]
        #     new_result = resulting_traj[i, 0, valid_mask]

        #     new_target = target[i, 0, :len(new_result)]
        #     tau, _ = kendalltau(new_target, new_result)
        #     rho, _ = spearmanr(new_target, new_result)
        #     print(f"valid_node: {valid_node[i]}, tau: {tau}, rho: {rho}")
        #     pdb.set_trace()

    
        if not np.isnan(tau):
            tau_list.append(tau)
            rho_list.append(rho)


    pdb.set_trace()
    tau_mean = np.mean(tau_list) if tau_list else 0
    rho_mean = np.mean(rho_list) if rho_list else 0
    tau_max = np.max(tau_list) if tau_list else 0
    rho_max = np.max(rho_list) if rho_list else 0
    
    print("tau_mean:", tau_mean)
    print("rho_mean:", rho_mean)
    print("tau_max:", tau_max)
    print("rho_max:", rho_max)
   
    # print("tau_median:", tau_median)
    # print("rho_median:", rho_median)
    # avg_episodic_return = np.mean(np.mean(episode_returns, axis=1))
    # max_episodic_return = np.mean(np.max(episode_returns, axis=1))
    # avg_episodic_length = np.mean(episode_lengths)
   
    # logging.info(
    #     "--------------------------------------------"
    #     f"[test] episodic_return={max_episodic_return}\n"
    #     f"avg_episodic_return={avg_episodic_return}\n"
    #     f"max_episodic_return={max_episodic_return}\n"
    #     f"avg_episodic_length={avg_episodic_length}\n"
    #     f"rehandle_rate={rehandle_rate}\n"
    #     f"tau={tau}\n"
    #     f"rho={rho_mean}\n"
    #     "--------------------------------------------"
    # )
    # logging.info("")


    # writer.add_scalar("test/episodic_return_mean", avg_episodic_return, global_step)
    # writer.add_scalar("test/episodic_return_max", max_episodic_return, global_step)
    # writer.add_scalar("test/episodic_length", avg_episodic_length, global_step)

    # writer.add_scalar("test/rehandle_rate", rehandle_rate, global_step)
    # writer.add_scalar("test/tau", tau, global_step)
    # writer.add_scalar("test/rho", rho_mean, global_step)


    # envs.close()
    # writer.close()
