

from collections import deque
from utils import Utils
from rl_algorithms.dqn.peragent import PERAgent
from environment.env import Env
from utils.net_parser import NetParser



def main_training_loop(config):
    """
    Main training loop for the reinforcement learning agents.
    """

    env = create_env(config=config)

    num_vehicles=8
   
    agents = [create_agent(config=config, agent_id=i) for i in range(num_vehicles)]
    best_reward = float('-inf')
    rolling_rewards = deque(maxlen=100)
    

    for episode in range(config['training_settings']['episodes']):
        cumulative_rewards = [0] * num_vehicles
        dones = [False] * num_vehicles

        steps = 0

        if episode % 1000 == 0:
            env.render("human")
        else:
            env.render()

        states = env.reset()
        taxi_fleet = env.vehicles
        dispatched_indices = [i for i, vehicle in enumerate(taxi_fleet) if vehicle.dispatched]
     
        while not all(dones):

            actions = [None] * num_vehicles
            dispatched_indices = [i for i, vehicle in enumerate(taxi_fleet) if vehicle.dispatched]


            for i in dispatched_indices:
                if not dones[i]:
           
                    actions[i] = agents[i].choose_action(states[i])


            if any(x is not None for x in actions):
                next_states, rewards, dones, infos = env.step(actions)
                steps += 1


            for i in dispatched_indices:

                if actions[i] is not None and next_states[i] != 0:

                    agents[i].remember(states[i], actions[i], rewards[i],next_states[i], dones[i])
                    # agents[i].remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
                    if len(agents[i].memory) > agents[i].batch_size:

                        agents[i].replay(agents[i].batch_size)
                        agents[i].hard_update()
                    # states[i] = next_states[i]
                    
                    states[i] = next_states[i]


                    
            for i in dispatched_indices:
                cumulative_rewards[i] += rewards[i]

            

        episode_reward = sum(cumulative_rewards)
        rolling_rewards.append(episode_reward) 
        rolling_avg = sum(rolling_rewards) / len(rolling_rewards)

    
        print(f"{episode} {[f'{reward:.3f}' for reward in cumulative_rewards]} {rolling_avg:.3f} {steps}'" )
        for i in dispatched_indices:
                agents[i].decay()
        env.quiet_close()
        if episode_reward > best_reward:
            best_reward = episode_reward



def create_env(config):
    """
    Create the simulation environment.

    Args:
        config (dict): Configuration dictionary.

    Returns:
            Env: Environment instance.
    """

    path = config['training_settings']['experiment_path']
    sumo_config_path = path + config['training_settings']['sumoconfig']
    parser = NetParser(sumo_config_path)
    edge_locations = (
            parser.get_edge_pos_dic()
        )  

    out_dict = parser.get_out_dic()
    index_dict = parser.get_edge_index()

    return Env(config, edge_locations,  out_dict, index_dict)


def create_agent(config,agent_id,
                #  central_memory,target_net
                 ):
    """
    Create the DQN agent.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        PERAgent: DQN agent instance.
    """
  
    experiment_path = config['training_settings']['experiment_path']
    learning_rate = config['agent_hyperparameters']['learning_rate']
    gamma = config['agent_hyperparameters']['gamma']
    epsilon_decay = config['agent_hyperparameters']['epsilon_decay']
    batch_size = config['agent_hyperparameters']['batch_size']
    memory_size = config['agent_hyperparameters']['memory_size']
    epsilon_max = config['agent_hyperparameters']['epsilon_max']
    epsilon_min = config['agent_hyperparameters']['epsilon_min']
    savepath = config['training_settings']['savepath']
    loadpath = config['training_settings']['savepath']
    alpha = config['per_hyperparameters']['alpha']
    beta = config['per_hyperparameters']['beta']
    priority_epsilon = config['per_hyperparameters']['priority_epsilon']
    

    return PERAgent(20, 6, experiment_path,
                    agent_id,
                    learning_rate,
                    gamma, 
                    epsilon_decay, 
                    epsilon_max, 
                    epsilon_min, 
                    memory_size, 
                    batch_size,
                    savepath,
                    loadpath,
                    alpha,
                    beta,
                    priority_epsilon,
                    # memory=central_memory,
                    # target_net=target_net
                    )

if __name__ == "__main__":
    
    try:
        config = Utils.load_yaml_config('src/configurations/config.yaml')
        main_training_loop(config)
    except Exception as e:
        print(f"Error occurred: {e}")




