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

# SARSA

# Method for choosing an action, neccesary for sarsa
def choose_action(epsilon, state):
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
        traci.vehicle.setAcceleration(target_car, 3, traci.simulation.getDeltaT())
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

    if (speed < 0) :
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
    
    DEBUG = 0 # to see debug prints
    
    TargetVehicleStatistics = np.zeros((math.floor(episodes/10), 2))
    
    rewards_taken = np.zeros(episodes)
    epsilons = np.zeros(episodes)

    # Define the number of states and actions
    # Change to the states and actions we define
    num_states = env.NUM_STATES_FINAL
    num_actions = len(env.ACTIONS)

    # Define the Q-table with zero initial values
    q_table = np.zeros((num_states, num_actions))

    # Define the learning rate (alpha) and the discount factor (gamma)
    alpha = 0.6
    gamma = 0.9

    # Define the maximum number of simulation steps
    max_steps = env.MAX_STEPS

    # Set the initial state of the car
    state = 0

    # Define the exploration rate (epsilon) and the exploration rate decay (epsilon_decay)
    epsilon = 1.0
    epsilon_min = 0.0
    epsilon_episode = 10000
    
    # first action # DIFFERENCE WITH QLEARNING, WE NEED THIS FUNCTION
    
    traci.start([sumoBinary, "-c", "./data/demo.sumocfg", "--no-step-log", '-W'])

    for episode in tqdm(range(episodes)):
        if DEBUG: 
            print("------------Starting episode", episode, "-----------------")
                
        traci.load(["-c", "./data/demo.sumocfg", "--no-step-log", '-W'])
        episodeIndex = math.floor(episode/10)
        
        # ini steps
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()
        traci.simulationStep()

        mycars = [10,11,12,13]  

        targetVehicle = "car_12"

        # we control car 12 now
        traci.vehicle.setSpeedMode(vehID=targetVehicle, sm=0)
        
        traci.vehicle.setSpeed(targetVehicle,random.randint(0,20)) # for starting
        traci.vehicle.setColor(targetVehicle,(255,0,0))
        
        total_reward = 0
        
        state, _ = get_state(targetVehicle)
        action = choose_action(epsilon, state)

        # Run the simulation for the maximum number of steps
        for step in range(max_steps):
            # Choose an action according to the Q-table and an exploration strategy (e.g. Epsilon-Greedy)
            
            # Execute the action and observe the new state and reward
            new_state, reward, done, done_reason = take_action(action, state, targetVehicle)
            
            total_reward += reward
            
            next_action = choose_action(epsilon, new_state)
            
            oldq_table = q_table
            
            # Update the Q-table using the Q-learning formula - DIFFERENCE WITH QLEARNING
            q_table[state, action] = q_table[state, action] + alpha * ((reward + gamma * q_table[new_state, next_action]) - q_table[state, action])
            
            # Set the new state as the current state
            state = new_state
            
            # Set the new action # DIFFERENCE WITH QLEARNING
            action = next_action
            
            if done:
                break
        
        if done_reason == "teleport":
            TargetVehicleStatistics[episodeIndex, 0] += 1
        elif done_reason == "collision":
            if DEBUG:
                print("Crash, :" + str(step))
            TargetVehicleStatistics[episodeIndex, 1] += 1
            
        rewards_taken[episode] = total_reward
        epsilons[episode] = epsilon
        
        epsilon = epsilon - 1 / epsilon_episode
        epsilon = max(epsilon, 0)
        
        if DEBUG:
            print("################ Ending episode", episode, "################")
            
    traci.close()
    
    sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"
    traci.start([sumoBinary, "-c", "./data/demo.sumocfg", "--no-step-log", '-W'])
    
    traci.load(["-c", "./data/demo.sumocfg", "--no-step-log", '-W'])
        
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

    traci.vehicle.setSpeedMode(vehID=targetVehicle, sm=0)
    #print("training vehicle: " + targetVehicle)
    
    traci.vehicle.setSpeed(targetVehicle,random.randint(0,20)) # for starting
    #traci.vehicle.setAccel(targetVehicle, 0)
    traci.vehicle.setColor(targetVehicle,(255,0,0))
    
    total_reward = 0
    
    # we save the qtable
    with open('qtableSARSA.npy', 'wb') as f:
        np.save(f, q_table)
        
    
    #epsilon = epsilon_ini

    # Advance the simulation by one step to make the cars appear
    #traci.simulationStep()
    
    #traci.vehicle.setSpeed(targetVehicle,random.randint(0,60))
    #print("Initial speed: " + str(traci.vehicle.getSpeed(targetVehicle)))
    state, _ = get_state(targetVehicle)
    # Run the simulation for the maximum number of steps
    for step in range(max_steps):
        # Choose an action according to the Q-table and an exploration strategy (e.g. Epsilon-Greedy)
        if np.random.uniform() < epsilon:
            # Choose a random action with probability epsilon
            action = np.random.randint(num_actions)
            #print("Random action: " + str(action))
        else:
            # Choose the action with the highest Q-value with probability (1 - epsilon)
            action = np.argmax(q_table[state, :])
            #print("Chosen action: " + str(action))
        
        # Execute the action and observe the new state and reward
        new_state, reward, done, done_reason = take_action(action, state, targetVehicle)
        
        total_reward += reward
        
        # Update the Q-table using the Q-learning formula
        #q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])
        
        # Set the new state as the current state
        state = new_state
        
        # Decrease the exploration rate
        #epsilon *= epsilon_decay
        
        #print("Speed: " + str(traci.vehicle.getSpeed("car_11")))
        #print("Speed: " + str(traci.vehicle.getSpeed(targetVehicle)) + ", action:" + str(action))
        
        if done:
            break
    
    traci.close()
    
    
    print("################# Training simulation has ended #################")
    x_axis = list(range(0, math.floor(episodes / 10)))
    plt.plot(x_axis, TargetVehicleStatistics[:, 1], label="Collision", linewidth = 1, color="orange")
    plt.xlabel('Episodes / 10')
    plt.ylabel('Events')
    plt.title('Target Vehicle Statistics')
    plt.legend()
    plt.rcParams["figure.figsize"] = (30,6)
    plt.show()
    
    mean_crashes = running_mean(TargetVehicleStatistics[:, 1],env.MAX_STEPS)
    #x_axis = list(range(0, episodes))
    plt.plot(mean_crashes, label="Mean crashes", linewidth = 1)
    plt.xlabel('Episodes')
    plt.ylabel('Mean crashes')
    plt.title('Target Vehicle Statistics')
    plt.legend()
    plt.rcParams["figure.figsize"] = (30,5)
    plt.show()
    
    x_axis = list(range(0, episodes))
    plt.plot(x_axis, rewards_taken, label="Reward", linewidth = 1)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Target Vehicle Statistics')
    plt.legend()
    plt.rcParams["figure.figsize"] = (30,5)
    plt.show()
    
    mean_rewards = running_mean(rewards_taken,env.MAX_STEPS)
    #x_axis = list(range(0, episodes))
    plt.plot(mean_rewards, label="Mean reward", linewidth = 1)
    plt.xlabel('Episodes')
    plt.ylabel('Mean reward')
    plt.title('Target Vehicle Statistics')
    plt.legend()
    plt.rcParams["figure.figsize"] = (30,5)
    plt.show()
    
    x_axis = list(range(0, episodes))
    plt.plot(x_axis, epsilons, label="Epsilon", linewidth = 1)
    plt.xlabel('Episodes')
    plt.ylabel('Different values of epsilon')
    plt.title('Target Vehicle Statistics')
    plt.legend()
    plt.rcParams["figure.figsize"] = (30,5)
    plt.show()
    
    print("End of execution. Value of collisions: ")    
    print(TargetVehicleStatistics[:, 1])