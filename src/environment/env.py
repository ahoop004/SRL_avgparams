import numpy as np
from utils.connect import SUMOConnection
from .observation import Observation
from .vehicle import Vehicle
from utils.utils import Utils
import copy


class Env():
    """
    Environment class for the SUMO-RL project.
    """

    def __init__(self, config, edge_locations, out_dict, index_dict):
        """
        Initialize the environment with the given configuration and parameters.

        Args:
            config (dict): Configuration dictionary.
            edge_locations (dict): Edge locations dictionary.
            out_dict (dict): Output dictionary.
            index_dict (dict): Index dictionary.
        """
        self.config = config  
        self.path = config['training_settings']['experiment_path']
        self.sumo_config_path = self.path + config['training_settings']['sumoconfig']
        self.num_of_agents = 2

        self.penalty = config['agent_hyperparameters']['penalty']
        self.start_life = self.config['training_settings']['initial_life']


        self.direction_choices = ['R', 'r', 's', 'L', 'l', 't']

        self.obs = Observation()

        self.sumo_con = SUMOConnection(self.sumo_config_path)

        self.edge_locations = edge_locations
        self.sumo = None   
        self.accumulated_reward = [[] for i in range(self.num_of_agents)]

        self.rewards = []
        self.epsilon_hist = []
        self.vehicles = []
        # self.people = []
        self.life = self.start_life
        self.distcheck = [0] * self.num_of_agents

        self.edge_distance = [0] * self.num_of_agents
        # self.route = [[] for _ in range(self.num_of_agents)]


        self.stage = "reset"
        self.dones = [0] * self.num_of_agents
        self.dispatched = [False] * self.num_of_agents  
        self.vedges = [None] * self.num_of_agents  
        self.old_vedges = [None] * self.num_of_agents

        self.infos = [None]* self.num_of_agents 

        self.reservations = []
        observations = [0] * self.num_of_agents

        self.out_dict = out_dict
        self.index_dict = index_dict

    def reset(self, seed=42):
        """
        Reset the environment to its initial state.

        Args:
            seed (int): Random seed for reproducibility.

        Returns:
            observations (list): Initial observations for each agent after reset.
        """
        # self.route = [[] for _ in range(self.num_of_agents)]
        self.distance_traveled = 0
        dones = [True] * self.num_of_agents
        rewards = [0] * self.num_of_agents
        self.dispatched = [False] * self.num_of_agents  
        self.sumo.simulationStep()
        self.vehicle_ids = self.sumo.vehicle.getTaxiFleet(-1)
        # self.empty_vehicles = self.sumo.vehicle.getTaxiFleet(0)
        # self.enroute_vehicles = self.sumo.vehicle.getTaxiFleet(1)
        # self.occupied_vehicles = self.sumo.vehicle.getTaxiFleet(2)
        # self.pickup_and_occupied = self.sumo.vehicle.getTaxiFleet(3)
        observations = [0] * self.num_of_agents
        

        self.vehicles = [
            Vehicle(
                id,
                self.out_dict,
                self.index_dict,
                self.edge_locations,
                self.sumo,
                self.life
            )
            for i, id in enumerate(self.vehicle_ids)
        ] #attaches vehicle object to vehicles in sim




        while True:# parks everything befor getting started
            self.sumo.simulationStep()
            vehicle_stop_states = [v.stop_info() for v in self.vehicles]
            if all(state == 135 for state in vehicle_stop_states):
                break 

        self.reservation_check_and_dispatch() # auto dispatch for first round
        self.sumo.simulationStep()



        dispatched_indices = [i for i, v in enumerate(self.vehicles) if v.dispatched]

        for i in dispatched_indices:
            observations[i] = self.get_observation(self.vehicles[i])
            # self.vehicles[i].can_choose = True

        return observations
    
    def reservation_check_and_dispatch(self):

        new_reservations = self.sumo.person.getTaxiReservations(1)
        retrieved_reservations = self.sumo.person.getTaxiReservations(2)
        assigned_reservations = self.sumo.person.getTaxiReservations(4)
        picked_up_reservations = self.sumo.person.getTaxiReservations(8)

        while len(retrieved_reservations)!=0:
            self.dispatch_vehicle_to_person()
            # all_reservations = self.sumo.person.getTaxiReservations(0)
            new_reservations = self.sumo.person.getTaxiReservations(1)
            retrieved_reservations = self.sumo.person.getTaxiReservations(2)
            assigned_reservations = self.sumo.person.getTaxiReservations(4)
            picked_up_reservations = self.sumo.person.getTaxiReservations(8)

        

    def dispatch_vehicle_to_person(self):
        reservations = self.sumo.person.getTaxiReservations(2)
        person_id = reservations[0].persons[0]
        reservation_id = reservations[0].id

        person_type = self.sumo.person.getParameter(person_id, "type")
        person_position = self.sumo.person.getPosition(person_id)

        closest_vehicle = None
        closest_distance = float('inf')
        
        for vehicle in self.vehicles:
            vehicle_type = self.sumo.vehicle.getParameter(vehicle.vehicle_id, "type")
            vehicle_state = vehicle.get_state()

            if vehicle_type == person_type:
            
                vehicle_position = self.sumo.vehicle.getPosition(vehicle.vehicle_id)

                distance = Utils.manhattan_distance(vehicle_position[0],vehicle_position[1], person_position[0],person_position[1])
                
         
                if distance < closest_distance and vehicle_state in ['0', '3']:
                    closest_distance = distance
                    closest_vehicle = vehicle
                    closest_vehicle.current_destination = reservations[0].fromEdge
                    closest_vehicle.passenger_pick_up_edge = reservations[0].fromEdge
                    closest_vehicle.bus_stop_drop_edge = reservations[0].toEdge
                    closest_vehicle.final_destination = reservations[0].toEdge

        if closest_vehicle is not None:
            # Set the vehicle as dispatched
            closest_vehicle.dispatched = True
            closest_vehicle.current_reservation = reservations[0]
            # Proceed with the dispatch logic
            self.sumo.vehicle.dispatchTaxi(closest_vehicle.vehicle_id, reservation_id)
        else:
            print(f"No available vehicle found for person type: {person_type}")






    def get_observation(self, vehicle):
        dest_loc = self.edge_locations[vehicle.current_destination ]
        final_loc = self.edge_locations[vehicle.final_destination ]
        return self.obs.get_state(self.sumo,
                                  vehicle,
                                  vehicle.agent_step,
                                  vehicle.life,
                                  dest_loc,
                                  final_loc,
                                  vehicle.distcheck,
                                  vehicle.final_distcheck,
                                  vehicle.picked_up, 
                                  
                                  vehicle.done)

    def step(self, actions):
        observations = [0] * self.num_of_agents
        dones = [True] * self.num_of_agents
        rewards = [0] * self.num_of_agents

        dispatched_indices = [i for i, vehicle in enumerate(self.vehicles) if (vehicle.dispatched  )]

        # Iterate over dispatched vehicles
        for i in dispatched_indices:
            vehicle = self.vehicles[i]
            vehicle.index = i

            vehicle.agent_step += 1
            vehicle.life -= 0.01  # Reduce vehicle life
            vehicle.reward = 0  # Initialize reward for this step

            vedge = vehicle.get_lane()
            while vedge not in self.index_dict:
                self.sumo.simulationStep()
                vedge = vehicle.get_lane()

            v_loc = vehicle.get_position()
            choices = vehicle.get_out_dict()
            choices_keys = choices.keys()

            # Use the action associated with the dispatched index
            choice = self.direction_choices[actions[i]]


            # Handle vehicle termination or invalid choices
            if (vehicle.life <= 0) or (choice not in choices_keys):

                vehicle.done = True
                self.dispatched[i] = False
                vehicle.dispatched = False
                vehicle.reward += -0.2
                rewards[i] = vehicle.reward
                self.infos[i] = vedge
                self.dones[i] = vehicle.done
                observations[i] = self.get_observation(vehicle)
                dispatched_indices = [i for i, vehicle in enumerate(self.vehicles) if vehicle.dispatched]

            
                vehicle.park()


            else:
                # vehicle.reward += 0.0005

                vehicle.destination_edge_location = self.edge_locations[vehicle.current_destination]
                vehicle.final_destination_edge_location = self.edge_locations[vehicle.final_destination]

                vehicle.destination_distance = Utils.manhattan_distance(
                    v_loc[0], v_loc[1],
                    vehicle.destination_edge_location[0], vehicle.destination_edge_location[1]
                )
                vehicle.final_destination_distance = Utils.manhattan_distance(
                    v_loc[0], v_loc[1],
                    vehicle.final_destination_edge_location[0], vehicle.final_destination_edge_location[1]
                )

                vehicle.destination_old_distance = vehicle.destination_distance
                vehicle.final_destination_old_distance = vehicle.final_destination_distance
                vehicle.set_destination(choice)
                # vehicle.can_choose = False
                



        # Proceed with SUMO step
        self.sumo.simulationStep()
        

        # Post-step logic for dispatched vehicles
        for i in dispatched_indices:
            vehicle = self.vehicles[i]
            # vehicle.get_current_edge()
            if vehicle.life <= 0 or actions.count(None) == len(actions):
                continue

            vedge = vehicle.get_road()
            # choices = vehicle.get_out_dict()
            v_loc = vehicle.get_position()

            # Update destination distance
            vehicle.destination_distance = Utils.manhattan_distance(
                v_loc[0], v_loc[1],
                vehicle.destination_edge_location[0], vehicle.destination_edge_location[1]
            )

            vehicle.final_destination_distance = Utils.manhattan_distance(
                v_loc[0], v_loc[1],
                vehicle.final_destination_edge_location[0], vehicle.final_destination_edge_location[1]
            )
            


            vehicle.distance_checks()

            if vehicle.fin:
                vehicle.reward += 1 + self.life - (vehicle.agent_step * 0.1)
                # print("Successful dropoff")

            # Update the lists with the current vehicle's values
            observations[i] = self.get_observation(vehicle)
            rewards[i] = vehicle.reward
            self.infos[i] = vedge #fix the self
            dones[i] = vehicle.done

        # Return values only for dispatched vehicles
        return (
            observations, 
            rewards, 
            dones, 
            self.infos

        )

    def render(self, mode='text'):
        """
        Render the environment.

        Args:
            mode (str): Mode of rendering. Options are 'human', 'text', 'no_gui'.

        .. todo:: Figure out how to determine os for rendering
        """
        if mode == "human":
            self.sumo = self.sumo_con.connect_gui()
        elif mode == "text":
            self.sumo = self.sumo_con.connect_libsumo_no_gui()
        elif mode == "no_gui":
            self.sumo = self.sumo_con.connect_no_gui()

    def pre_close(self, episode, agent, accu, current_epsilon):

        acc_r = float(accu)
        self.accumulated_reward[agent].append(acc_r)
        self.epsilon_hist.append(current_epsilon)
        # avg_reward[agent] = np.mean(self.accumulated_reward[agent][-100:])

        return
    
    def quiet_close(self):
        """
        Quietly close the environment without logging.
        """
        self.sumo.close()
        return
    
    def get_route_length(self, route):
        """
        Get the total length of the route.

        Args:
            route (list): List of edges in the route.

        Returns:
            int: Total length of the route.
        """
        distances = []
        for edge in route:
            distances.append(self.sumo.lane.getLength(''.join([edge, '_0'])))
        return round(sum(distances))
