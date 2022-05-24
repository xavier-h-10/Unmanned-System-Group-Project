import sys
sys.path.append("../")
import logging as log
from env import combat_env, discrete_env
from utils.define import PI, COMBAT_OBS_INFO
from trainer import sac_trainer, dqn_trainer
import random
import tf_agents
import functools
from numpy.linalg import norm
import numpy as np
from absl import app
from absl import logging
from typing import Tuple
from tf_agents.policies.random_py_policy import RandomPyPolicy

log.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = log.getLogger("train")

_IP = '10.119.10.174'
_PORT = 10088

sac_prefix = './save/20220522_v4/sac'
dqn_prefix = './save/20220522_v4/dqn'

tot = 0


def get_state(obs: np.ndarray) -> np.ndarray:
    """
    TODO: 把拿到的原始Observation转换成用于训练的State
    :param obs[0:3]: 己方位移，xyz
    :param obs[3:6]: 己方欧拉角，uvw
    :param obs[6:9]: 己方线速度，VxVyVz
    :param obs[9:12]: 己方角速度，VuVvVw
    :param obs[12]: 己方在上一个步长中受到的伤害
    :param obs[13]: 己方当前生命值
    :param obs[14:28]: 敌方信息，具体含义同obs[0:14]
    :return: 在训练中作为环境表示的State向量，需以np.ndarray返回
    """

    state = obs
    return np.array(state)


# TODO: 计算自己定义的State的最小值与最大值。
# 此处因为get_state函数直接返回了原始Observation，State的最大最小值同Observation。
state_min = [-1e5, -1e5, 0., -PI / 4., -PI / 3., 0., -
             1e3, -1e3, -1e3, -PI / 4, -PI / 4, -PI / 4, 0., 0.] * 2
state_max = [1e5, 1e5, 1e5, PI / 4., PI / 3., PI * 2.,
             1e3, 1e3, 1e3, PI / 4, PI / 4, PI / 4, 5., 100.] * 2


def get_reward(obs: np.ndarray, prev_obs: np.ndarray) -> Tuple[float, bool]:
    """
    TODO: 自定义奖励函数，以及终止条件判断。
    一般而言，整个episode的奖励绝对值应该在1000以内，最好不超过2000。
    :param obs: 当前回合的原始Observation（不是State），每个索引的含义见get_state函数的注释
    :param prev_obs: 上一时刻的Observation，回合刚开始时为None
    :return: (Reward, 是否终止) -> Tuple[float, bool]
    """
    global tot
    tot += 1
    damage_cause, damage_suffer, health, opp_health = obs[26], obs[12], obs[13], obs[27]

    pos, opp_pos= obs[0:3], obs[14:17]
    angle, opp_angle = obs[3:6], obs[17: 20]
    linear_speed, opp_linear_speed = obs[6:9], obs[20: 23]
    angular_speed, opp_angular_speed = obs[9:12], obs[23:26]

    speed, opp_speed = norm(linear_speed), norm(opp_linear_speed)
    dis= norm(pos-opp_pos)

    # 如果任意一方生命值降为0，给予高奖励或惩罚，并终止当前episode
    if health <= 0. or opp_health <= 0.:
        return -500. if health < opp_health else 500., True

    # 需要特殊判断prev_obs是否为None
    if prev_obs is None:
        return 0., False

    prev_pos, opp_prev_pos =  prev_obs[0:3], prev_obs[14:17]
    prev_dis = norm(prev_pos-opp_prev_pos)
    r_v_self = (speed/600-0.5)*100 if speed < 300 else 0   # 速度太低给予惩罚
    r_v = (speed-opp_speed)*1                              # 速度比对手快给予奖励   
    r_pos = (750-dis) * 0.1                                # 距离越近奖励越高 
    r_damage = (damage_cause-damage_suffer)*30             
    r_delta_pos = (prev_dis-dis)*0.5                       # 距离变近给奖励 

    if tot % 100 == 0:
        logger.info("tot:{}, pos:{}, opp_pos:{}, speed:{}, dis:{}, prev_dis:{}, r_delta_pos:{}, r_v_self:{}, r_v:{}, r_pos:{}, r_damage:{} damage_cause:{}, damage_suffer:{}".format(
            tot, pos, opp_pos, speed, dis, prev_dis, r_delta_pos, r_v_self, r_v, r_pos, r_damage, damage_cause, damage_suffer))

    return r_delta_pos+r_v_self+r_v+r_pos+r_damage, False


def _gen_random_init_pos():
    """
    用于生成战局开始时两架战机出初始位移和姿态角
    这里限制了初始xy在[-20000., 20000.]，初始z在[500., 10000.]范围内
    翻滚、俯仰、偏航角按照环境限制最大范围进行随机（环境各个观测量的取值范围见utils.define.COMBAT_OBS_INFO）
    TODO[可选]: 可以自己修改xyz的范围，先从容易探索的小范围开始，训练一段时间发现算法能够收敛之后终止训练程序，扩大范围，然后重新运行训练程序。
    """
    return [([random.uniform(-2000, 2000) for _ in range(2)] + [random.uniform(5000, 10000)] +
             [random.uniform(COMBAT_OBS_INFO[1][i], COMBAT_OBS_INFO[2][i]) for i in range(3, 6)]) for _ in range(2)]


def env_constructor():
    """
    用于构建模拟对战环境的函数
    max_step过大会延长训练时间，过小有可能还没分出胜负就导致回合结束，一般在1000~5000内
    其余参数一般不变动
    """
    return combat_env.CombatEnv(
        ip=_IP, port=_PORT,
        mock_policy_fcn=RandomPyPolicy,
        state_size=len(state_min), state_min=state_min, state_max=state_max,
        get_state_fcn=get_state, get_reward_fcn=get_reward,
        max_step=2000,
        gen_init_pos_fcn=_gen_random_init_pos,
        introduce_damage=True
    )


def get_dqn_trainer():
    """
    使用dqn方法进行训练。
    该方法将动作空间离散化了，方法详见'env/discrete_env.py'文件。
    TODO[可选]: 如果有更好的离散化策略可以自行修改_convert_back函数。
    parallel_num: 与环境交互采样数据的进程个数，一般越多训练速度越快。但是多开会占用大量内存，建议根据电脑性能调整。
    """
    parallel_num = 15

    # TODO: 调节超参数
    return dqn_trainer.DQNTrainer(
        env_constructor=env_constructor,                                    # 无需变动
        # 无需变动
        env_wrappers=[discrete_env.DirectionActionWrapper],
        collect_episodes_per_iter=parallel_num,                             # 无需变动
        train_rounds_per_iter=parallel_num * 8,                             # 无需变动
        initial_collect_episodes=parallel_num * 3,                          # 无需变动
        metric_buffer_num=parallel_num,                                     # 无需变动
        # Critic全连接隐层节点数量（两层各256节点）
        fc_layer_params=(256, 256),
        # 探索时采用随机策略的概率
        epsilon_greedy=0.1,
        # Critic网络学习率
        q_net_lr=5e-4,
        gamma=0.99,                                                         # 折扣因子
        # 奖励放缩，调整参数使其乘上一局游戏总奖励绝对值小于1000
        reward_scale_factor=0.1,
        target_update_tau=0.005,                                            # 目标网络软更新参数
        target_update_period=1,                                             # 更新目标网络间隔时间
        replay_cap=1000000,                                                 # 经验回放的大小
        # 训练数据记录路径，用于tensorboard可视化
        train_summary_dir=dqn_prefix+'/summary/train',
        # 验证数据记录路径，用于tensorboard可视化
        eval_summary_dir=dqn_prefix+'/summary/eval',
        train_checkpoint_dir=dqn_prefix+'/checkpoint/train',         # 用于中断后继续训练
        policy_checkpoint_dir=dqn_prefix+'/checkpoint/policy',       # 用于中断后继续训练
        replay_checkpoint_dir=dqn_prefix+'/checkpoint/replay',       # 用于中断后继续训练
        eval_policy_save_dir=dqn_prefix+'/eval_policy',              # 训练好的策略保存的位置
        # 是否使用Double-DQN算法
        using_ddqn=True,
        parallel_num=parallel_num                                           # 并行采样进程数
    )


def get_sac_trainer():
    """
    使用sac方法进行训练
    parallel_num: 与环境交互采样数据的进程个数，一般越多训练速度越快。但是多开会占用大量内存，建议根据电脑性能调整。
    """
    parallel_num = 15

    # TODO: 调节超参数
    return sac_trainer.SACTrainer(
        env_constructor=env_constructor,                                # 无需变动
        collect_episodes_per_iter=parallel_num,                         # 无需变动
        # 与环境交互一次之后训练网络的轮数，若交互时间长，可以适当调大
        train_rounds_per_iter=parallel_num * 8,
        initial_collect_episodes=parallel_num * 3,                      # 无需变动
        metric_buffer_num=parallel_num,                                 # 无需变动
        # 训练数据记录路径，用于tensorboard可视化
        train_summary_dir=sac_prefix+'/summary/train',
        # 验证数据记录路径，用于tensorboard可视化
        eval_summary_dir=sac_prefix+'/summary/eval',
        train_checkpoint_dir=sac_prefix+'/checkpoint/train',     # 用于中断后继续训练
        policy_checkpoint_dir=sac_prefix+'/checkpoint/policy',   # 用于中断后继续训练
        replay_checkpoint_dir=sac_prefix+'/checkpoint/replay',   # 用于中断后继续训练
        eval_policy_save_dir=sac_prefix+'/eval_policy',          # 训练好的策略保存的位置
        # Actor全连接层隐层节点数量（两层各256节点）
        actor_fc_layer_params=(256, 256),
        # Critic用于编码环境的全连接层隐层节点数量
        critic_observation_fc_layer_params=(128, 128),
        # Critic用于编码动作的全连接层隐层节点数量
        critic_action_fc_params=(128, 128),
        # Critic经过编码层后连接的全连接层隐层节点数量
        critic_joint_fc_layer_params=(256, 256),
        actor_lr=1e-3,                                                  # Actor学习率
        critic_lr=1e-3,                                                 # Critic学习率
        # SAC算法中用于更新alpha变量的学习率
        alpha_lr=1e-3,
        gamma=0.99,                                                     # 折扣因子
        # 奖励放缩，调整参数使其乘上一局游戏总奖励绝对值小于1000
        reward_scale_factor=0.1,
        target_update_tau=0.005,                                        # 目标网络软更新参数
        target_update_period=1,                                         # 更新目标网络间隔时间
        # 训练网络的batch大小
        batch_size=256,
        replay_cap=1000000,                                             # 经验回放的大小
        parallel_num=parallel_num                                       # 并行采样进程数
    )


def train(_):
    trainer = get_dqn_trainer()
    trainer.train()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    tf_agents.system.system_multiprocessing.handle_main(
        functools.partial(app.run, train))
