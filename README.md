## Training OR model with PPO

### Container

python ppo_or.py --num-steps 51 --env-id container-v0 --env-entry-point envs.container_vector_env:ContainerVectorEnv --problem container

### TSP

```shell
python ppo_or.py --num-steps 51 --env-id tsp-v0 --env-entry-point envs.tsp_vector_env:TSPVectorEnv --problem tsp
```

### CVRP

```shell
python ppo_or.py --num-steps 60 --env-id cvrp-v0 --env-entry-point envs.cvrp_vector_env:CVRPVectorEnv --problem cvrp
```

### Enable WandB

```shell
python ppo_or.py ... --track
```

首先结束默认启动的TensorBoard进程，执行命令：ps -ef | grep tensorboard | awk '{print $2}' | xargs kill -9 
在终端中执行以下命令启动TensorBoard
$ tensorboard --port 6007 --logdir ./runs

