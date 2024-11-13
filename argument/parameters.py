import argparse

def get_paras():

    parser = argparse.ArgumentParser("Parameters for PPO-CIM")
    parser.add_argument("-env-name", "--env_name", type=str, default='Swimmer-v2')
    parser.add_argument("-algorithm", "--algorithm", type=str, default='CIM-1')
    parser.add_argument("-number-of-episode", "--num_episode", type=int, default=2000)
    parser.add_argument("-max-step-per-round", "--max_step_per_round", type=int, default=100)
    parser.add_argument("-num_epoch", "--num_epoch", type=int, default=10)
    parser.add_argument("-data-path", "--data_path", type=str, default='./data/')
    parser.add_argument("-fig-path", "--fig_path", type=str, default='./fig/')
    parser.add_argument("-learning-rate-of-actor", "--lr_a", type=float, default=0.0001)
    parser.add_argument("-learning-rate-of-critic", "--lr_c", type=float, default=0.0001)
    parser.add_argument("-gamma", "--gamma", type=float, default=0.99)
    parser.add_argument("-gae_lambda", "--gae_lambda", type=float, default=0.99)
    parser.add_argument("-batch-size", "--batch_size", type=int, default=256)
    parser.add_argument("-cim-sigma", "--cim_sigma", type=int, default=1)
    parser.add_argument("-cim-alpha", "--cim_alpha", type=int, default=5)
    args = parser.parse_args()
    return args