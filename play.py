import traci
import numpy as np
import math
import os
import sys
import random
import custom_env
import traci
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt

# TEST FILE, here we can use the qtable returned by qlearning and sarsa to test the algorithms

def choose_action(epsilon):
    # Choose an action according to the Q-table and an exploration strategy (e.g. Epsilon-Greedy)
    if np.random.uniform() < epsilon:
        # Choose a random action with probability epsilon
        action = np.random.randint(num_actions)
    else:
        # Choose the action with the highest Q-value with probability (1 - epsilon)
        action = np.argmax(q_table[state, :])
    return action

def take_action(action, previous_state, target_car):
    if (action == 1):
        traci.vehicle.setAcceleration(target_car, 1, traci.simulation.getDeltaT())
    elif (action == 3):
        traci.vehicle.setAcceleration(target_car, -1, traci.simulation.getDeltaT())
    elif (action == 2):
        traci.vehicle.setAcceleration(target_car, 2, traci.simulation.getDeltaT())
    elif (action == 4):
        traci.vehicle.setAcceleration(target_car, -3, traci.simulation.getDeltaT())
    elif (action == 5):
        traci.vehicle.setAcceleration(target_car, -7, traci.simulation.getDeltaT())
    else:
        traci.vehicle.setAcceleration(target_car, 0, traci.simulation.getDeltaT())
        
    # Advance the simulation by one step
    traci.simulationStep()
    
    # Get the resulting state
    state, check_state = get_state(target_car)

    if (check_state >= 0):
        reward = env.COLLISION_REWARD
        if DEBUG:
            print("By this: " + str(check_state))
        return previous_state, reward, True, "collision"
    else:
        # Calculate the reward
        reward = env.getRewardFINAL(action, target_car)

        # Check if the episode has ended
        done, done_reason = check_done(target_car)
        return state, reward, done, done_reason
    
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_state(vehicle_id):
    speed = traci.vehicle.getSpeed(vehicle_id)
    distance_to_front = env.get_distance_from_leader(vehicle_id)

    if (speed <= -1073741820) :
        return 0, 0 # car was teleported, so there is a crash
    elif (distance_to_front < 0):
        return 0, 1 # car was teleported, so there is a crash
    else:
        n_speed = math.floor(speed)
        n_distance_to_front = math.floor(distance_to_front)
        return env.STATE_DICT_FINAL[(n_speed,n_distance_to_front)], -1

def check_done(vehicle_id):
    # Returns True if the episode has ended, False otherwise.
    # Check for collision
    collisonList = traci.simulation.getCollidingVehiclesIDList()
    if len(collisonList) > 0:
        if vehicle_id in collisonList:
            return True, "collision"
    if traci.vehicle.getLanePosition(vehicle_id) < 0:  # DELETE THIS
        return True, "outoflane"
    return False, "none"

if __name__ == "__main__":
    runGUI = False
    sumoBinary = "sumo"
    if runGUI:
        sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"

    env = custom_env.CustomEnv()
    episodes = env.MAX_EPISODES
    steps = env.MAX_STEPS
    
    DEBUG = 0
    tableName = 'qtableQL.npy' # choose the qtable, right now is with qlearning
    
    TargetVehicleStatistics = np.zeros((math.floor(episodes/10), 2))
    
    rewards_taken = np.zeros(episodes)
    epsilons = np.zeros(episodes)

    # Define the number of states and actions
    # Change to the states and actions we define
    num_states = env.NUM_STATES_FINAL
    num_actions = len(env.ACTIONS)

    # Define the learning rate (alpha) and the discount factor (gamma)
    #alpha = 0.2
    alpha = 0.5
    gamma = 0.9

    # Define the maximum number of simulation steps
    max_steps = env.MAX_STEPS

    # Define the exploration rate (epsilon) and the exploration rate decay (epsilon_decay)
    epsilon = 0.0
    #epsilon_decay = 0.99
    
    sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"
    traci.start([sumoBinary, "-c", "./data/demo.sumocfg", "--no-step-log", '-W'])
    
    for c in range(0,1):
        traci.load(["-c", "./data/demo.sumocfg", "--no-step-log", '-W'])
        
        with open(tableName,'rb') as f: 
            q_table = np.load(f)
            
        # ini steps
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()

        targetVehicle = "car_12"
        state, _ = get_state(targetVehicle)

        traci.vehicle.setSpeedMode(vehID=targetVehicle, sm=0)

        traci.vehicle.setAccel(targetVehicle, 0)
        traci.vehicle.setColor(targetVehicle,(255,0,0))
        
        total_reward = 0
        
        # Run the simulation for the maximum number of steps
        for step in range(max_steps):
            # Choose an action according to the Q-table and an exploration strategy (e.g. Epsilon-Greedy)
            if np.random.uniform() < epsilon:
                # Choose a random action with probability epsilon
                action = np.random.randint(num_actions)
            else:
                # Choose the action with the highest Q-value with probability (1 - epsilon)
                action = np.argmax(q_table[state, :])
                print(np.argmax(q_table[state, :]))
            
            # Execute the action and observe the new state and reward
            new_state, reward, done, done_reason = take_action(action, state, targetVehicle)
            
            total_reward += reward
            
            # Set the new state as the current state
            state = new_state
            
            if done:
                print("Crash")
                break
        
    traci.close()
    
    print("Total reward: " + str(total_reward))
    
    print(str(q_table))