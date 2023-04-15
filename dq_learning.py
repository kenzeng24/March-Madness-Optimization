from collections import deque
import random

import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential  # To compose multiple Layers
from tensorflow.keras.layers import Dense       # Fully-Connected layer
from tensorflow.keras.layers import Activation  # Activation functions
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

from march_madness import MarchMadnessEnvironment

GAMMA = 0.95

def train_model(
    EPOCHS = 300,
    epsilon = 1.0,
    EPSILON_REDUCE = 0.995,
    LEARNING_RATE = 0.001, 
):
    env = MarchMadnessEnvironment()
    n_actions = 2
    n_teams = len(env.teams_list)

    model = Sequential()
    model.add(Dense(16, input_shape=(1, n_teams)))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(n_actions))
    model.add(Activation('linear'))
    print(model.summary())
    
    target_model = clone_model(model)
    replay_buffer = deque(maxlen=20000)
    update_target_model = 10

    model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))

    best_so_far = 0
    for epoch in range(EPOCHS):
        
        observation, info = env.reset()  # Get inital state
        observation =  tf.reshape(
            tf.constant(list(observation.values())), (1,-1)
        )
        
        # Keras expects the input to be of shape [1, X] thus we have to reshape
        # [Jeremy] Original state is an array of shape (4): [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]
        done = False  
        
        points = 0
        while not done:  # As long current run is active
            
            # Select action according to strategy
            action = epsilon_greedy_action_selection(model, epsilon, observation)
            
            # Perform action and get next state
            next_observation, reward, done, info = env.step(action)
            next_observation =  tf.reshape(
                tf.constant(list(next_observation.values())), (1,-1)
            )
            
            replay_buffer.append((observation, action, reward, next_observation, done))  # Update the replay buffer
            
            observation = next_observation  # Update the observation
            
            points += 1

            # Train the model by replaying
            #print(f"*** Debug: Done = {done}")
            replay(replay_buffer, 32, model, target_model)

        epsilon *= EPSILON_REDUCE  # Reduce epsilon
        
        # Check if we need to update the target model
        update_model_handler(epoch, update_target_model, model, target_model)
        
        if points > best_so_far:
            best_so_far = points
        if epoch %25 == 0:
            print(f"========== {epoch}: Points reached: {points} - epsilon: {epsilon} - Best: {best_so_far}")

    return model

def epsilon_greedy_action_selection(model, epsilon, observation):
    obs=[]
    if np.random.random() > epsilon:
        #print(f"*** Taking Greedy Action, observation shape 1: {observation.shape}")
        observation = tf.reshape(observation, [1, 1, -1]) 
        #print(f"*** Taking Greedy Action, observation shape 2: {observation.shape}")
        prediction = model.predict(observation)  # Perform the prediction on the observation
        action = np.argmax(prediction)           # Chose the action with the highest value
    else:
        #print(f"*** Taking a random action")
        action = np.random.randint(0, 2)  # Select random action with probability epsilon
    return action

def update_model(replay_buffer, batch_size, model, target_model, gamma):
    """Update the model using experience replay.

    Args:
        replay_buffer (list): List of experiences in the form of (state, action, reward, next_state, done).
        batch_size (int): Number of experiences to sample from the replay buffer.
        model (tf.keras.Model): The main model to be trained.
        target_model (tf.keras.Model): The target model to estimate the TD target.
        gamma (float): Discount factor for future rewards.

    Returns:
        None
    """

    # Take a random sample from the buffer with size batch_size
    samples = random.sample(replay_buffer, batch_size)  

    # Efficient way to handle the sample by using the zip functionality
    states, actions, rewards, next_states, dones = list(zip(*samples))   

    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_states = np.array(next_states)
    dones = np.array(dones)

    # Compute the TD targets using the target model
    next_qs = target_model.predict(next_states)
    targets = rewards + gamma * np.max(next_qs, axis=1) * (1 - dones)

    # Compute the Q-values for the current state and action using the main model
    q_values = model.predict(states)
    q_values[np.arange(batch_size), actions] = targets

    # Train the main model using the updated Q-values
    model.fit(states, q_values, epochs=1, verbose=0)

def replay(replay_buffer, batch_size, model, target_model):
    
    # As long as the buffer has not enough elements we do nothing
    if len(replay_buffer) < batch_size: 
        return
    
    # Take a random sample from the buffer with size batch_size
    samples = random.sample(replay_buffer, batch_size)  
    
    # Initialize variable to store the targets predicted by the target network for training
    target_batch = []  
    
    # Efficient way to handle the sample by using the zip functionality
    zipped_samples = list(zip(*samples))  
    states, actions, rewards, new_states, dones = zipped_samples  
    
    # Predict targets for all states from the sample
    #print(f"*** *** *** *** EXPERIENCE REPLAY, length states: {len(states)}")
    #print(f"*** *** *** *** EXPERIENCE REPLAY, states: {np.array(states).shape}")
    #print(f"*** *** *** *** EXPERIENCE REPLAY, states: {np.array(states[0]).shape}")
    targets = target_model.predict(np.array(states))
    
    # Predict Q-Values for all new states from the sample
    q_values = model.predict(np.array(new_states))  
    
    # Now we loop over all predicted values to compute the actual targets
    for i in range(batch_size):  
        
        # Take the maximum Q-Value for each sample
        q_value = max(q_values[i][0])  
        
        # Store the ith target in order to update it according to the formula
        target = targets[i].copy()  
        if dones[i]:
            target[0][actions[i]] = rewards[i]
        else:
            target[0][actions[i]] = rewards[i] + q_value * GAMMA
        target_batch.append(target)

    # Fit the model based on the states and the updated targets for 1 epoch
    model.fit(np.array(states), np.array(target_batch), epochs=1, verbose=0)  

def update_model_handler(epoch, update_target_model, model, target_model):
    if epoch > 0 and epoch % update_target_model == 0:
        target_model.set_weights(model.get_weights())
        #print(f"*** Debug: Updating model")


def exploit_model(model):
    env = march_madness.MarchMadnessEnvironment()
    observation, info = env.reset()
    total_reward = 0 
    done = False
    while not done:
        # Choose action from predicted Q-values
        observations =  tf.reshape(
            tf.constant(list(observation.values())),
            (1,1,-1)
        )
        action = np.argmax(model.predict(observations)) 
        
        # Perform the action 
        observation, reward, done, info = env.step(action)
        total_reward  += reward
        # clear_output(wait=True)
        
        if done:
            print(f"Total Reward: {total_reward}")
            env.reset()
            #break
            
    env.close()