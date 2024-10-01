
# import numpy as np
import random

# directions from https://github.com/guangli-dai/Selfless-Traffic-Routing-Testbed/blob/master/core/STR_SUMO.py
STRAIGHT = "s"
TURN_AROUND = "t"
LEFT = "l"
RIGHT = "r"
SLIGHT_LEFT = "L"
SLIGHT_RIGHT = "R"

class Vehicle:
    """
    Vehicle class representing a vehicle in the simulation.

    Attributes:
        vehicle_id (str): Unique identifier for the vehicle.
        direction_choices (list): List of possible direction choices.
        out_dict (dict): Dictionary of outgoing edges.
        index_dict (dict): Dictionary of edge indices.
        sumo: SUMO simulation instance.
        edge_position (dict): Dictionary of edge positions.
    """

    def __init__(self, vehicle_id, out_dict, index_dict, edge_position, sumo, life) -> None:
        """
        Initialize a Vehicle instance with the given parameters.

        Args:
            vehicle_id (str): Unique identifier for the vehicle.
            types (int): Number of vehicle types.
            out_dict (dict): Dictionary of outgoing edges.
            index_dict (dict): Dictionary of edge indices.
            edge_position (dict): Dictionary of edge positions.
            sumo: SUMO simulation instance.
        """

        self.direction_choices = [SLIGHT_RIGHT, RIGHT, STRAIGHT, SLIGHT_LEFT, LEFT, TURN_AROUND]
       
        self.index = None

        self.current_stage = 0
        self.current_destination = None
        self.final_destination = None
        self.passenger_pick_up_edge = None
        self.bus_stop_drop_edge = None

        self.route = []

        self.passenger_id = None


        self.done = False
        self.reward = 0

        self.agent_step = 0
        self.life = life

        self.distcheck = 0
        self.final_distcheck = 0


        self.destination_edge_location = None
        self.final_destination_edge_location = None

        self.destination_distance = 0
        self.destination_old_distance = 0
        
        self.final_destination_distance = 0
        self.final_destination_old_distance = 0
        
        self.picked_up = False
        self.dropped_off = False
        self.fin = False
       
        self.vehicle_id = vehicle_id
        self.out_dict = out_dict
        self.index_dict = index_dict
        self.sumo = sumo
        self.edge_position = edge_position

        self.dispatched = False
        self.current_reservation = None


        self.reward_history = []

        self.current_edge = self.get_current_edge()
        self.can_choose = False


        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane.partition("_")[0]



    def get_lane(self):
        """
        Get the current lane of the vehicle.

        Returns:
            str: Current lane ID.
        """
 
        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane.partition("_")[0]
        return self.cur_loc

    def get_lane_id(self):
        """
        Get the full lane ID of the vehicle.

        Returns:
            str: Full lane ID.
        """
 
        self.current_lane = self.sumo.vehicle.getLaneID(self.vehicle_id)
        self.cur_loc = self.current_lane
        return self.cur_loc

    def location(self):
        """
        Get the current location of the vehicle.

        Returns:
            list: Current (x, y) position of the vehicle.
        """

        x,y = self.sumo.vehicle.getPosition(self.vehicle_id)
        return [x, y]

    def get_out_dict(self):
        """
        Get the dictionary of possible outgoing edges from the current lane.

        Returns:
            dict: Dictionary of outgoing edges.
        """
 
        lane = self.get_lane()
        if lane not in self.out_dict.keys():
            options = None
        else:
             options = self.out_dict[lane]

        return options

    def set_destination(self, action):

        """
        Set the destination edge for the vehicle.

        Args:
            action (str): Chosen action.
            destination_edge (str): Destination edge ID.

        Returns:
            str: Target lane ID.
        """

        self.cur_loc = self.current_lane.partition("_")[0]
        outlist = list(self.out_dict[self.cur_loc].keys())
        if action in outlist:

            target_lane = self.out_dict[self.cur_loc][action]

            new_route = [self.cur_loc,target_lane]

            self.sumo.vehicle.setVia(self.vehicle_id,new_route)
            self.sumo.vehicle.rerouteEffort(self.vehicle_id)
            self.teleport(target_lane)
            if target_lane==self.current_destination:
                if self.current_stage==0:


                    self.life+=.1
                    self.picked_up=True

                    self.current_destination = self.bus_stop_drop_edge
                    self.current_stage=1

                elif self.current_stage==1:

                    self.life+=.1
                    self.picked_up=True
                    self.dropped_off = True
                    self.done = True
                    self.fin = True
                    self.dispatched = False
                    self.current_destination = self.park()


    def get_stop_state(self):
        return self.sumo.vehicle.getStopState(self.vehicle_id)
    
    def get_state(self):
        return self.sumo.vehicle.getParameter(self.vehicle_id,"device.taxi.state")

    def pickup(self, reservation):
        """
        Dispatch the vehicle to pick up a passenger.
        """

        reservation_id = reservation.id
        self.passenger_pick_up_edge = reservation.fromEdge
        self.bus_stop_drop_edge = reservation.toEdge
        self.sumo.vehicle.dispatchTaxi(self.vehicle_id,reservation_id)
        self.current_destination = self.passenger_pick_up_edge

        self.dispatched = True

        
    def get_road(self): 
        """
        Get the current road ID of the vehicle.

        Returns:
            str: Current road ID.
        """ 
        self.current_edge = self.sumo.vehicle.getRoadID(self.vehicle_id)

        return self.current_edge

    def random_relocate(self):
        """
        Relocate the vehicle to a random lane.
        """
 
        new_lane=random.choice(list(self.index_dict.keys()))      
        self.sumo.vehicle.changeTarget(self.vehicle_id,edgeID=new_lane)
        self.sumo.vehicle.moveTo(self.vehicle_id,new_lane+"_0",5)

    def get_type(self):
        """
        Get the type of the vehicle.

        Returns:
            str: Vehicle type.
        """

        return self.sumo.vehicle.getParameter(self.vehicle_id,
                                              "type")
        
    def teleport(self, dest):
        """
        Teleport the vehicle to the destination edge.

        Args:
            dest (str): Destination edge ID.
        """
   
        self.sumo.vehicle.changeTarget(self.vehicle_id, edgeID=dest)
        self.sumo.vehicle.moveTo(self.vehicle_id, dest+"_0", 1)

    def retarget(self, dest):
        """
        Retarget the vehicle to the destination edge.

        Args:
            dest (str): Destination edge ID.
        """

        self.sumo.vehicle.changeTarget(self.vehicle_id, edgeID=dest)


    

    def get_route(self):
        """
        Get vehicle route

        Returns:
            str: Vehicle route.
        
        """

        return self.sumo.vehicle.getRouteIndex(self.vehicle_id),self.sumo.vehicle.getRoute(self.vehicle_id)
    
    def park(self):
            fleet_area = self.sumo.vehicle.getTypeID(self.vehicle_id)
            if fleet_area=="taxi:A":
                self.retarget('-49664167#6')
                self.sumo.vehicle.setParkingAreaStop(self.vehicle_id, "FleetA")
                return '-49664167#6'
            if fleet_area=="taxi:B":
                self.retarget('192469469#3')
                self.sumo.vehicle.setParkingAreaStop(self.vehicle_id, "FleetB")
                return '192469469#3'


            



    def get_dist(self):
        return self.sumo.vehicle.getDistance(self.vehicle_id)
    
    def get_position(self):
        return self.sumo.vehicle.getPosition(self.vehicle_id)
        
    
    def distance_checks(self):
        

        if self.destination_old_distance > self.destination_distance:
            
            self.distcheck = 1
            
        elif self.destination_old_distance < self.destination_distance:
            
            self.distcheck = 0
        
        if self.final_destination_old_distance > self.final_destination_distance:
            
            self.final_distcheck = 1
            
        elif self.final_destination_old_distance < self.final_destination_distance:
            
            self.final_distcheck = 0
        
        return self.distcheck, self.final_distcheck
            

    
    def stop_info(self):
        return self.sumo.vehicle.getStopState(self.vehicle_id)
    
    def current_edge_location(self):
        current_road = self.get_road()  # Get the current road or edge
        if current_road in self.edge_position:  # Check if the edge exists in edge_position dictionary
            return self.edge_position[current_road]  # Return the edge position if it exists
        else:
            return self.location()

    def get_current_edge(self):
        """
        Returns the current edge the vehicle is on.
        """

        return  self.sumo.vehicle.getRoadID(self.vehicle_id)
    

    def update_edge(self):
        """ Check if the vehicle has moved to a new edge. """
        new_edge = self.get_current_edge()
        
        # If the vehicle has moved to a new edge, only allow choosing if no decision has been made yet.
        if self.current_edge != new_edge and self.get_lane() in self.index_dict:
            if self.can_choose is False:  # Ensure an action hasn't been taken yet
                self.can_choose = True
            else:
                # Decision is made, so update to the new edge
                self.current_edge = new_edge
                self.can_choose = False  # After update, reset flag
            return True
        
        return False
    def distance_traveled(self):
        return self.sumo.vehicle.getDistance(self.vehicle_id)

