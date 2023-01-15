import traci
import numpy as np
import traci
from random import randint
import math


class CustomEnv:

    def __init__(self):
        self.MAX_EPISODES = 20
        self.MAX_STEPS = 200
        self.CAR_SIZE = 5
        # actions
        self.BRAKE = 3
        self.ACCELERATE = 1
        self.NOTHING = 0
        self.STRONG_BRAKE = 4
        self.BRAKE_EMERGENCY = 5
        self.STRONG_ACCELERATE = 2
        self.ACTIONS = {self.NOTHING,  self.ACCELERATE, self.STRONG_ACCELERATE, self.BRAKE, self.STRONG_BRAKE, self.BRAKE_EMERGENCY}
        # rewards
        self.SAFE_DISTANCE_REWARD = 62
        self.COLLISION_REWARD = -1000
        self.NOT_SAFE_DISTANCE_REWARD_FRONT = -20
        self.NOT_SAFE_DISTANCE_REWARD_BACK = -10
        self.RECOMMENDED_DIST = 25
        # states
        self.SPEED_VALUES = range(0, 61, 1)
        self.DISTANCE_VALUES = range(0, 402, 1)

        # Create a dictionary that maps each state to a unique integer index
        # initial state map
        self.STATE_DICT = {}
        state_index = 0
        for speed in self.SPEED_VALUES:
          for distance in self.DISTANCE_VALUES:
              for distance2 in self.DISTANCE_VALUES:
                  self.STATE_DICT[(speed, distance, distance2)] = state_index
                  state_index += 1
                
        # final state map
        self.STATE_DICT_FINAL = {}
        state_index = 0
        for speed in self.SPEED_VALUES:
          for distance in self.DISTANCE_VALUES:
                  self.STATE_DICT_FINAL[(speed, distance)] = state_index
                  state_index += 1
        self.NUM_STATES_FINAL = len(self.STATE_DICT)
        # Define the number of states
        self.NUM_STATES = len(self.STATE_DICT)

    def reset(self):
        currentState = 0
        return currentState
    
    def getDistance(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
    
    def get_distance_from_leader(self, target_car):
        # We are using only car_12 at the moment, so I check the distance with car_13
        return min(traci.vehicle.getDistance("car_13") - traci.vehicle.getDistance(target_car), 400)
        
    def get_distance_from_follower(self):
        # We are using only car_12 at the moment, so I check the distance with car_11
        return min(traci.vehicle.getDistance("car_12") - traci.vehicle.getDistance("car_11"), 400)

    # old function
    def getReward(self, action, targetVehicleId):
        waiting_time = traci.vehicle.getAccumulatedWaitingTime(targetVehicleId)
        #print("Distance :" + str(self.get_distance_from_leader(targetVehicleId)))
        # Check if the car has collided with another car
        collisonList = traci.simulation.getCollidingVehiclesIDList()
        #print("Distance: " + str(self.get_distance_from_leader(targetVehicleId)))
        reward = 0
        speed_m = traci.vehicle.getSpeed(targetVehicleId)
        recommended_dist = math.floor(speed_m * 2.3)
        if len(collisonList) > 0 and targetVehicleId in collisonList:
            reward += self.COLLISION_REWARD
        elif self.get_distance_from_leader(targetVehicleId) > 100 and action >= 3:
            # Give the car a negative reward for too much distance with the car in front
            #reward = -self.get_distance_from_leader(targetVehicleId) + 90
            #reward = self.NOT_SAFE_DISTANCE_BRAKING_REWARD
            reward += math.floor(-self.get_distance_from_leader(targetVehicleId) / 8)
        elif self.get_distance_from_leader(targetVehicleId) > 100 and action == 0:
            # Give the car a negative reward for too much distance with the car in front
            #reward = self.NOT_SAFE_DISTANCE_REWARD
            reward += math.floor(-self.get_distance_from_leader(targetVehicleId) / 11)
        elif self.get_distance_from_leader(targetVehicleId) < recommended_dist and (action >= 0 and action < 3):
            #reward += -(51-self.get_distance_from_leader(targetVehicleId))
            reward += -((recommended_dist+5)-self.get_distance_from_leader(targetVehicleId))
            #reward = self.NOT_SAFE_DISTANCE_REWARD
        else:
            # Give the car a positive reward for maintaining a safe distance
            reward += self.SAFE_DISTANCE_REWARD

        if (speed_m > 25):
            reward -= 4
        elif (speed_m < 5 and speed_m >= 0):
            reward -= 6
            
        return reward
    
    # Final version of the reward, with some improvements to the previous one
    def getRewardFINAL(self, action, targetVehicleId):
        waiting_time = traci.vehicle.getAccumulatedWaitingTime(targetVehicleId)
        # Check if the car has collided with another car
        collisonList = traci.simulation.getCollidingVehiclesIDList()
        reward = 0
        speed_m = traci.vehicle.getSpeed(targetVehicleId)
        recommended_dist = self.RECOMMENDED_DIST
        ideal_dist = math.floor((recommended_dist + recommended_dist+ 60) / 2)
        if len(collisonList) > 0 and targetVehicleId in collisonList:
            reward += self.COLLISION_REWARD
        elif self.get_distance_from_leader(targetVehicleId) > (recommended_dist+45):
            # Give the car a negative reward for too much distance with the car in front
            if (action == 0 or action >= 3):
                reward += -6
            reward += self.NOT_SAFE_DISTANCE_REWARD_BACK
        elif self.get_distance_from_leader(targetVehicleId) < recommended_dist:
            # Give the car a negative reward for not enough distance with the car in front
            if (action > 0 and action < 3):
                reward += -10
            reward += self.NOT_SAFE_DISTANCE_REWARD_FRONT
        else:
            # Give the car a positive reward for maintaining a safe distance
            dist = self.get_distance_from_leader(targetVehicleId)
            if (dist > ideal_dist):
                reward += (self.SAFE_DISTANCE_REWARD - (dist - ideal_dist))
            elif (dist < ideal_dist):
                reward += (self.SAFE_DISTANCE_REWARD - (ideal_dist - dist))
            else:
                reward += self.SAFE_DISTANCE_REWARD

        # too much speed or too less speed
        if (speed_m > 25):
            reward -= 1
        elif (speed_m < 5 and speed_m >= 0):
            reward -= 1
            
        return reward