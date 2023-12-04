import numpy as np 
import random
import matplotlib.pyplot as plt # Graphical library
from sklearn.metrics import mean_squared_error # Mean-squared error function
import warnings
warnings.filterwarnings('ignore')

"""# Coursework 1 :
See pdf for instructions. 
"""

# WARNING: fill in these two functions that will be used by the auto-marking script
# [Action required]

def get_CID():
  return "01722077" # Return your CID (add 0 at the beginning to ensure it is 8 digits long)

def get_login():
  return "ed1922" # Return your short imperial login

"""## Helper class"""

# This class is used ONLY for graphics
# YOU DO NOT NEED to understand it to work on this coursework

class GraphicsMaze(object):

  def __init__(self, shape, locations, default_reward, obstacle_locs, absorbing_locs, absorbing_rewards, absorbing):

    self.shape = shape
    self.locations = locations
    self.absorbing = absorbing

    # Walls
    self.walls = np.zeros(self.shape)
    for ob in obstacle_locs:
      self.walls[ob] = 20

    # Rewards
    self.rewarders = np.ones(self.shape) * default_reward
    for i, rew in enumerate(absorbing_locs):
      self.rewarders[rew] = 10 if absorbing_rewards[i] > 0 else -10

    # Print the map to show it
    self.paint_maps()

  def paint_maps(self):
    """
    Print the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders)
    plt.show()

  def paint_state(self, state):
    """
    Print one state on the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    states = np.zeros(self.shape)
    states[state] = 30
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders + states)
    plt.show()

  def draw_deterministic_policy(self, Policy):
    """
    Draw a deterministic policy
    input: Policy {np.array} -- policy to draw (should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    for state, action in enumerate(Policy):
      if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
        continue
      arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
      action_arrow = arrows[action] # Take the corresponding action
      location = self.locations[state] # Compute its location on graph
      plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
    plt.show()

  def draw_policy(self, Policy):
    """
    Draw a policy (draw an arrow in the most probable direction)
    input: Policy {np.array} -- policy to draw as probability
    output: /
    """
    deterministic_policy = np.array([np.argmax(Policy[row,:]) for row in range(Policy.shape[0])])
    self.draw_deterministic_policy(deterministic_policy)

  def draw_value(self, Value):
    """
    Draw a policy value
    input: Value {np.array} -- policy values to draw
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    for state, value in enumerate(Value):
      if(self.absorbing[0, state]): # If it is an absorbing state, don't plot any value
        continue
      location = self.locations[state] # Compute the value location on graph
      plt.text(location[1], location[0], round(value,2), ha='center', va='center') # Place it on graph
    plt.show()

  def draw_deterministic_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple deterministic policies
    input: Policies {np.array of np.array} -- array of policies to draw (each should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Policies)): # Go through all policies
      ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each policy
      ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
      for state, action in enumerate(Policies[subplot]):
        if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
          continue
        arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
        action_arrow = arrows[action] # Take the corresponding action
        location = self.locations[state] # Compute its location on graph
        plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
      ax.title.set_text(title[subplot]) # Set the title for the graph given as argument
    plt.show()

  def draw_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policies (draw an arrow in the most probable direction)
    input: Policy {np.array} -- array of policies to draw as probability
    output: /
    """
    deterministic_policies = np.array([[np.argmax(Policy[row,:]) for row in range(Policy.shape[0])] for Policy in Policies])
    self.draw_deterministic_policy_grid(deterministic_policies, title, n_columns, n_lines)

  def draw_value_grid(self, Values, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policy values
    input: Values {np.array of np.array} -- array of policy values to draw
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Values)): # Go through all values
      ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each value
      ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
      for state, value in enumerate(Values[subplot]):
        if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
          continue
        location = self.locations[state] # Compute the value location on graph
        plt.text(location[1], location[0], round(value,1), ha='center', va='center') # Place it on graph
      ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
    plt.show()

"""## Maze class"""

# This class define the Maze environment

class Maze(object):

  # [Action required]
  def __init__(self):
    """
    Maze initialisation.
    input: /
    output: /
    """
    
    # [Action required]
    # Properties set from the CID
    y, z = list(map(float, get_CID()[-2:]))
    self._prob_success = 0.8 + (0.02 * (9 - y)) # float
    self._gamma = 0.8 + (0.02*y) # float
    #self._prob_success = 0.84
    #self._gamma = 0.3
    self._goal = z%4 # integer (0 for R0, 1 for R1, 2 for R2, 3 for R3)
    #self._goal = 1

    # Build the maze
    self._build_maze()
                              

  # Functions used to build the Maze environment 
  # You DO NOT NEED to modify them
  def _build_maze(self):
    """
    Maze initialisation.
    input: /
    output: /
    """

    # Properties of the maze
    self._shape = (13, 10)
    self._obstacle_locs = [
                          (1,0), (1,1), (1,2), (1,3), (1,4), (1,7), (1,8), (1,9), \
                          (2,1), (2,2), (2,3), (2,7), \
                          (3,1), (3,2), (3,3), (3,7), \
                          (4,1), (4,7), \
                          (5,1), (5,7), \
                          (6,5), (6,6), (6,7), \
                          (8,0), \
                          (9,0), (9,1), (9,2), (9,6), (9,7), (9,8), (9,9), \
                          (10,0)
                         ] # Location of obstacles
    self._absorbing_locs = [(2,0), (2,9), (10,1), (12,9)] # Location of absorbing states
    self._absorbing_rewards = [ (500 if (i == self._goal) else -50) for i in range (4) ]
    self._starting_locs = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9)] #Reward of absorbing states
    self._default_reward = -1 # Reward for each action performs in the environment
    self._max_t = 500 # Max number of steps in the environment

    # Actions
    self._action_size = 4
    self._direction_names = ['N','E','S','W'] # Direction 0 is 'N', 1 is 'E' and so on
        
    # States
    self._locations = []
    for i in range (self._shape[0]):
      for j in range (self._shape[1]):
        loc = (i,j) 
        # Adding the state to locations if it is no obstacle
        if self._is_location(loc):
          self._locations.append(loc)
    self._state_size = len(self._locations)

    # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
    self._neighbours = np.zeros((self._state_size, 4)) 
    
    for state in range(self._state_size):
      loc = self._get_loc_from_state(state)

      # North
      neighbour = (loc[0]-1, loc[1]) # North neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('N')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('N')] = state

      # East
      neighbour = (loc[0], loc[1]+1) # East neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('E')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('E')] = state

      # South
      neighbour = (loc[0]+1, loc[1]) # South neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('S')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('S')] = state

      # West
      neighbour = (loc[0], loc[1]-1) # West neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('W')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('W')] = state

    # Absorbing
    self._absorbing = np.zeros((1, self._state_size))
    for a in self._absorbing_locs:
      absorbing_state = self._get_state_from_loc(a)
      self._absorbing[0, absorbing_state] = 1

    # Transition matrix
    self._T = np.zeros((self._state_size, self._state_size, self._action_size)) # Empty matrix of domension S*S*A
    for action in range(self._action_size):
      for outcome in range(4): # For each direction (N, E, S, W)
        # The agent has prob_success probability to go in the correct direction
        if action == outcome:
          prob = 1 - 3.0 * ((1.0 - self._prob_success) / 3.0) # (theoritically equal to self.prob_success but avoid rounding error and garanty a sum of 1)
        # Equal probability to go into one of the other directions
        else:
          prob = (1.0 - self._prob_success) / 3.0
          
        # Write this probability in the transition matrix
        for prior_state in range(self._state_size):
          # If absorbing state, probability of 0 to go to any other states
          if not self._absorbing[0, prior_state]:
            post_state = self._neighbours[prior_state, outcome] # Post state number
            post_state = int(post_state) # Transform in integer to avoid error
            self._T[prior_state, post_state, action] += prob

    # Reward matrix
    self._R = np.ones((self._state_size, self._state_size, self._action_size)) # Matrix filled with 1
    self._R = self._default_reward * self._R # Set default_reward everywhere
    for i in range(len(self._absorbing_rewards)): # Set absorbing states rewards
      post_state = self._get_state_from_loc(self._absorbing_locs[i])
      self._R[:,post_state,:] = self._absorbing_rewards[i]

    # Creating the graphical Maze world
    self._graphics = GraphicsMaze(self._shape, self._locations, self._default_reward, self._obstacle_locs, self._absorbing_locs, self._absorbing_rewards, self._absorbing)
    
    # Reset the environment
    self.reset()


  def _is_location(self, loc):
    """
    Is the location a valid state (not out of Maze and not an obstacle)
    input: loc {tuple} -- location of the state
    output: _ {bool} -- is the location a valid state
    """
    if (loc[0] < 0 or loc[1] < 0 or loc[0] > self._shape[0]-1 or loc[1] > self._shape[1]-1):
      return False
    elif (loc in self._obstacle_locs):
      return False
    else:
      return True


  def _get_state_from_loc(self, loc):
    """
    Get the state number corresponding to a given location
    input: loc {tuple} -- location of the state
    output: index {int} -- corresponding state number
    """
    return self._locations.index(tuple(loc))


  def _get_loc_from_state(self, state):
    """
    Get the state number corresponding to a given location
    input: index {int} -- state number
    output: loc {tuple} -- corresponding location
    """
    return self._locations[state]

  # Getter functions used only for DP agents
  # You DO NOT NEED to modify them
  def get_T(self):
    return self._T

  def get_R(self):
    return self._R

  def get_absorbing(self):
    return self._absorbing

  # Getter functions used for DP, MC and TD agents
  # You DO NOT NEED to modify them
  def get_graphics(self):
    return self._graphics

  def get_action_size(self):
    return self._action_size

  def get_state_size(self):
    return self._state_size

  def get_gamma(self):
    return self._gamma

  # Functions used to perform episodes in the Maze environment
  def reset(self):
    """
    Reset the environment state to one of the possible starting states
    input: /
    output: 
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """
    self._t = 0
    self._state = self._get_state_from_loc(self._starting_locs[random.randrange(len(self._starting_locs))])
    self._reward = 0
    self._done = False
    return self._t, self._state, self._reward, self._done

  def step(self, action):
    """
    Perform an action in the environment
    input: action {int} -- action to perform
    output: 
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """

    # If environment already finished, print an error
    if self._done or self._absorbing[0, self._state]:
      print("Please reset the environment")
      return self._t, self._state, self._reward, self._done

    # Drawing a random number used for probaility of next state
    probability_success = random.uniform(0,1)

    # Look for the first possible next states (so get a reachable state even if probability_success = 0)
    new_state = 0
    while self._T[self._state, new_state, action] == 0: 
      new_state += 1
    assert self._T[self._state, new_state, action] != 0, "Selected initial state should be probability 0, something might be wrong in the environment."

    # Find the first state for which probability of occurence matches the random value
    total_probability = self._T[self._state, new_state, action]
    while (total_probability < probability_success) and (new_state < self._state_size-1):
     new_state += 1
     total_probability += self._T[self._state, new_state, action]
    assert self._T[self._state, new_state, action] != 0, "Selected state should be probability 0, something might be wrong in the environment."
    
    # Setting new t, state, reward and done
    self._t += 1
    self._reward = self._R[self._state, new_state, action]
    self._done = self._absorbing[0, new_state] or self._t > self._max_t
    self._state = new_state
    return self._t, self._state, self._reward, self._done

"""## DP Agent"""

# This class define the Dynamic Programing agent 

class DP_agent(object):

  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
    def solve(self, env):
        """
        Solve a given Maze environment using Dynamic Programming
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - V {np.array} -- Corresponding value function 
        """
    
        # Initialisation (can be edited)
        policy = np.zeros((env.get_state_size(), env.get_action_size())) 
        V = np.zeros(env.get_state_size())
        

        #### 
        # Add your code here
        # WARNING: for this agent only, you are allowed to access env.get_T(), env.get_R() and env.get_absorbing()
        ####

        # Value iteration algorithm for policy improvement using synchronous DP, since only does full backup once
        # 
        threshold = 0.001
        delta = 2*threshold
        
        while delta>threshold:
            delta = 0
            for prior in range(env.get_state_size()):
                if env.get_absorbing()[0, prior]==1:
                    continue
                vals = []
                v = V[prior]
                for a in range(env.get_action_size()):
                    new_val = 0
                    for post in range(env.get_state_size()):
                        new_val += env.get_T()[prior, post, a] * (env.get_R()[prior, post, a] + env.get_gamma()*V[post])
                    vals.append(new_val)
                V[prior] = max(vals)
                delta = max(delta, abs(V[prior] - v))
            

        for prior in range(env.get_state_size()):
            vals1 = []
            for action in range(env.get_action_size()):
                t = 0
                for post in range(env.get_state_size()):
                    t += env.get_T()[prior, post, action] * (env.get_R()[prior, post, action] + env.get_gamma()*V[post])
                vals1.append(t)
            policy[prior, np.argmax(vals1)] = 1

        return policy, V

"""## MC agent"""

# This class define the Monte-Carlo agent

class MC_agent(object):
  
      # [Action required]
      # WARNING: make sure this function can be called by the auto-marking script
    def solve(self, env):
        """
        Solve a given Maze environment using Monte Carlo learning
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
        """

        # Initialisation (can be edited)
        Q = np.zeros((env.get_state_size(), env.get_action_size()))
        V = np.zeros(env.get_state_size())
        policy = np.ones((env.get_state_size(), env.get_action_size())) / float(env.get_action_size())
        #Initialised as equiprobable policy

        num_ep = 10000
        values = np.empty((num_ep, env.get_state_size()))
        total_rewards = np.zeros(num_ep)

        #### 
        # Add your code here
        # WARNING: this agent only has access to env.reset() and env.step()
        # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
        ####
        
        #Every visit Batch MC epsilon greedy
        batch_size = 1000
        state_counter = {}
        for s in range(env.get_state_size()):
            for a in range(env.get_action_size()):
                key = str(s)+","+str(a)
                state_counter[key] = 0
        
        
        for episode in range(num_ep):
            _,state,_,done = env.reset()
            states = [state]
            actions = []
            rewards = []
            
            
            while not done:
                a = np.random.choice(4,p=policy[state])
                t,state,reward,done = env.step(a)
                actions.append(a)
                states.append(state)
                rewards.append(reward)
                total_rewards[episode]+=reward

            R = 0    
            for i in range(len(rewards), 0, -1):
                
                s = states[i-1]
                a = actions[i-1]
                k = str(s)+","+str(a)
                
                state_counter[k]+=1
                R = R*env.get_gamma() + rewards[i-1]
                Q[s,a] += (R-Q[s,a])/float(state_counter[k])
                V[s] = np.max(Q[s])
            
            values[episode] = V
            
            
            if (episode+1)%batch_size==0:
                epsilon = min(1, (num_ep/8)/float(episode+1))
                
                for s in range(env.get_state_size()):
                    best = np.argmax(Q[s])
                    for a in range(env.get_action_size()):
                        if a != best:
                            policy[s,a] = epsilon/float(env.get_action_size())
                        else:
                            policy[s,a] = 1 - epsilon + (epsilon/env.get_action_size())
                
                for d in state_counter.keys():
                    state_counter[d] = 0
                
                    

        return policy, values, total_rewards


"""## TD agent"""

# This class define the Temporal-Difference agent

class TD_agent(object):

    # [Action required]
    # WARNING: make sure this function can be called by the auto-marking script
    def solve(self, env):
        """
        Solve a given Maze environment using Temporal Difference learning
        input: env {Maze object} -- Maze to solve
        output: 
          - policy {np.array} -- Optimal policy found to solve the given Maze environment 
          - values {list of np.array} -- List of successive value functions for each episode 
          - total_rewards {list of float} -- Corresponding list of successive total non-discounted sum of reward for each episode 
        """

        # Initialisation (can be edited)
        Q = np.zeros((env.get_state_size(), env.get_action_size())) #initalised to 0
        V = np.zeros(env.get_state_size())
        policy = np.ones((env.get_state_size(), env.get_action_size())) /float(env.get_action_size())
        num_ep = 5000
        values = np.empty((num_ep, env.get_state_size()))
        total_rewards = np.zeros(num_ep)
        
        #### 
        # Add your code here
        # WARNING: this agent only has access to env.reset() and env.step()
        # You should not use env.get_T(), env.get_R() or env.get_absorbing() to compute any value
        ####
        
        for episode in range(num_ep):
            _,state,_,done = env.reset()
            alpha = 0.1
            epsilon = min(1, (num_ep/2.8)/float(episode+1))
      
            while not done:
                a = np.random.choice(4, p=policy[state])
                t,new_state,reward,done = env.step(a)
                total_rewards[episode]+=reward
                delta = reward + (env.get_gamma()*np.max(Q[new_state]) - Q[state, a])
                Q[state, a] += alpha*delta
                best = np.argmax(Q[state])
                for i in range(env.get_action_size()):
                    if i!=best:
                        policy[state,i] = epsilon/float(env.get_action_size())
                    else:
                        policy[state,i] = 1 - epsilon + epsilon/float(env.get_action_size())
                state = new_state
                
            for s in range(env.get_state_size()):
                V[s] = np.max(Q[s])
            values[episode] = V
            
        return policy, values, total_rewards

"""## Example main"""
if __name__ == "__main__":

  # Example main (can be edited)

  ### Question 0: Defining the environment

  print("Creating the Maze:\n")
  maze = Maze()

  ### Question 1: Dynamic programming

  dp_agent = DP_agent()
  dp_policy, dp_value = dp_agent.solve(maze)

  print("Results of the DP agent:\n")
  maze.get_graphics().draw_policy(dp_policy)
  maze.get_graphics().draw_value(dp_value)

  maze = Maze()
  dp_agent = DP_agent()
  #gmaze = GraphicsMaze(maze)
  #dp_value, dp_policy = dp_agent.solve(maze)

  thresh = np.logspace(-6, -2, num=5)
  #print(len(thresh))
  iters = []
  policies = []
  values = []
  titles = []

  # Use policy iteration for each gamma value
  for t in thresh:
      policy, V, iter = dp_agent.solve(maze, threshold=t)
      policies.append(policy)
      iters.append(iter)
      values.append(V)
      titles.append("Threshold = {}".format(t))

  # Plot the number of epochs vs gamma values
  print("Impact of threshold value on the number of epochs needed for the policy iteration algorithm:\n")
  plt.figure()
  plt.plot(thresh, iters)
  plt.xscale("log")
  plt.xlabel("Threshold range")
  plt.ylabel("Number of epochs")
  plt.title("Impact of threshold value on the number of iterations for DP algorithm")
  plt.show()

  # Print all value functions and policies for different values of gamma
  print("\nGraphical representation of the value function for each threshold:\n")
  maze.get_graphics().draw_value_grid(values, titles, 1, 6)

  print("\nGraphical representation of the policy for each threshold:\n")
  maze.get_graphics().draw_policy_grid(policies, titles, 1, 6)

  ### Question 2: Monte-Carlo learning

  mc_agent = MC_agent()
  mc_policy, mc_values, total_rewards = mc_agent.solve(maze)

  print("Results of the MC agent:\n")
  maze.get_graphics().draw_policy(mc_policy)
  maze.get_graphics().draw_value(mc_values[-1])


  #batch size 500 frac = 2.0 works well, 10k iterations

  xs = np.arange(len(mc_values))
  ys = []
  ys1 = []
  for i in range(len(mc_values)):
      ys.append(mean_squared_error(mc_values[i], dp_value))
      ys1.append(mean_squared_error(td_values[i]))
      #print(mean_squared_error(mc_values[i], dp_value))
  plt.plot(xs, ys)
  plt.xlabel("Episode number")
  plt.ylabel("MSE")
  plt.title("MSE against episode number")
  plt.show()
  #print(mean_squared_error(mc_values[0], dp_value))
  #print(mc_values[0])
  print(ys[-1])
  print(type(total_rewards[0]))

  def mov_avg(a, n=500):
      ret = np.cumsum(a, dtype=float)
      ret[n:] = ret[n:]-ret[:-n]
      return ret[n-1:]/n

  plt.plot(mov_avg(total_rewards))
  plt.xlabel("Episode number")
  plt.ylabel("Total sum of undiscounted reward")
  plt.title("Moving average of total sum of undiscounted rewards")
  plt.show()
  #print(np.where(mov_avg(total_rewards)<-340))

  plt.scatter(mov_avg(total_rewards), mov_avg(ys), s=.5)
  plt

  policies = []
  vals = []
  rew = []
  for k in range(10):
      mc_pol, mc_val, mc_rew = mc_agent.solve(maze)
      policies.append(mc_pol)
      vals.append(mc_val)
      rew.append(mc_rew)

  pols = []
  vals = []
  rew = []

  for k in range(30):
      policy, values, rewards = mc_agent.solve(maze)
      pols.append(policy)
      vals.append(values)
      rew.append(rewards)

  #Batch size 500, frac=2.0.

  vals = np.array(vals)
  errs = []

  for i in range(30):
      errs.append(mean_squared_error(dp_value, vals[i,-1]))
      

  print(np.shape(vals[:,-1]))
  plt.plot(errs)
  print(np.std(errs))

  plt.plot(np.arange(10000), total_rewards)
  roll_mean = []
  G = 0
  for i in range(len(total_rewards)):
      G+=total_rewards[i]
      roll_mean.append(G/float(i+1))

  plt.plot(xs, roll_mean)
  plt.show()

  #print(np.min(roll_me[2]))
  print(np.mean(total_rewards))
  print(total_rewards[9800:])

  mc_agent = MC_agent()
  errors = []
  for k in range(10):
      mc_pol, mc_value, mc_rewards = mc_agent.solve(maze)
      errors.append(mean_squared_error(mc_value[-1], dp_value))

  ### Question 3: Temporal-Difference learning

  td_agent = TD_agent()
  td_policy, td_values, td_total_rewards = td_agent.solve(maze)

  print("Results of the TD agent:\n")
  maze.get_graphics().draw_policy(td_policy)
  maze.get_graphics().draw_value(td_values[-1])

  xs = np.arange(len(td_values))
  ys = []
  for i in range(len(td_values)):
      ys.append(mean_squared_error(td_values[i], dp_value))
      #print(mean_squared_error(mc_values[i], dp_value))
  plt.plot(xs, ys)
  plt.show()
  print(ys[-1])

  sns.lineplot(mov_avg(td_total_rewards), mov_avg(ys))

  alphas = np.linspace(0,1,11)
  fracs = np.logspace

  pols = []
  tot = []
  vals = []
  titles = []

  td_agent = TD_agent()
  for a in alphas:
      print("\n")
      print("Working on alpha = {}".format(a), end="\r")
      td_policy, td_values, total_rewards = td_agent.solve(maze, alpha=np.round(a, 1))
      pols.append(td_policy)
      tot.append(total_rewards)
      vals.append(td_values)
      titles.append("Learning rate: {}".format(a))

  k = 0
  xs = np.arange(len(vals[k]))
  for k in range(len(vals)):
      if k<6:
          ys = []
          for i in range(len(vals[k])):
              ys.append(mean_squared_error(vals[k][i], dp_value))
              #print(mean_squared_error(mc_values[i], dp_value))
          plt.plot(xs, ys, label=f"alpha = {np.round(alphas[k], 1)}")
  plt.title("Impact of alpha parameter on convergence rate")
  plt.legend()
  plt.show()

  plt.figure()
  plt.scatter(np.arange(2000), tot[2], s=2)
  plt.xlabel("Episodes")
  plt.ylabel("Total undiscounted reward")
  plt.title("Learning curve")
  plt.show()
  vals = np.array(vals)
  pols = np.array(pols)
  # Print all value functions and policies for different values of gamma
  print("\nGraphical representation of the value function for each threshold:\n")
  maze.get_graphics().draw_value_grid(vals[:,-1], titles, 2, 7)

  print("\nGraphical representation of the policy for each threshold:\n")
  maze.get_graphics().draw_policy_grid(pols, titles, 2, 7)

  #Check for ideal epsilon and alpha

  alphas = np.linspace(0.1,1,10)
  fracs = np.logspace(1, 6, endpoint=True, base=np.e)

  pols = [[] for i in range(len(alphas))]
  tot = [[] for i in range(len(alphas))]
  vals = [[] for i in range(len(alphas))]
  titles = [[] for i in range(len(alphas))]

  maze=Maze()
  td_agent = TD_agent()
  for d,i in enumerate(alphas):
      for j in fracs:
          print("Working on alpha = {}, frac = {}".format(i, j), end="\r")
          td_policy, td_values, total_rewards = td_agent.solve(maze, alpha=np.round(i, 1), frac=j)
          pols[d].append(td_policy)
          tot[d].append(total_rewards)
          vals[d].append(td_values)
          titles.append("Learning rate: {}, exploration duration as fraction: 1/{}".format(i,j))

  mat = np.empty((len(alphas), len(fracs)))
  for i in range(len(alphas)):
      for j in range(len(fracs)):
          mat[i][j] = mean_squared_error(vals[i][j][-1], dp_value)

  plt.imshow(mat)
  plt.colorbar()
  plt.show()

  print(mat[2][3])
  print(alphas[2])
  print(fracs[3])

  fracs2 = np.logspace(1, 5, num=6, endpoint=True, base=np.e)
  print(fracs)

  td_pols= []
  td_vals = []
  td_rews = []

  for k in range(30):
      td_pol, td_val, td_rew = td_agent.solve(maze)
      td_pols.append(td_pol)
      td_vals.append(td_val)
      td_rews.append(td_rew)

  mc_agent = MC_agent()

  mc_pols= []
  mc_vals = []
  mc_rews = []

  for k in range(30):
      mc_pol, mc_val, mc_rew = mc_agent.solve(maze)
      mc_pols.append(mc_pol)
      mc_vals.append(mc_val)
      mc_rews.append(mc_rew)

  td_vals[0]

  td_rews = np.array(td_rews)

  mean_rews = np.mean(td_rews, axis = 0)
  std_rews = np.std(td_rews, axis=0)
  print(np.shape(mean_rews))
  #plt.scatter(np.arange(len(mean_rews)),mean_rews, s=0.1)
  plt.plot(mean_rews)
  plt.fill_between(np.arange(5000), mean_rews-std_rews, mean_rews+std_rews, alpha=0.5)
  plt.xlabel("Episode Number")
  plt.ylabel("Total undiscounted reward")
  plt.title("Learning curve for TD agent showing mean and std for 30 replications")

  td_agent = TD_agent()

  td_pols= []
  td_vals = []
  td_rews = []

  for k in range(30):
      td_pol, td_val, td_rew = td_agent.solve(maze)
      td_pols.append(td_pol)
      td_vals.append(td_val)
      td_rews.append(td_rew)

  mc_agent = MC_agent()

  mc_pols= []
  mc_vals = []
  mc_rews = []

  for k in range(30):
      mc_pol, mc_val, mc_rew = mc_agent.solve(maze)
      mc_pols.append(mc_pol)
      mc_vals.append(mc_val)
      mc_rews.append(mc_rew)

  td_vals = np.array(td_vals)
  mc_vals = np.array(mc_vals)

  td_mse = np.empty((30,5000))
  mc_mse = np.empty((30,10000))

  for i in range(30):
      for j in range(10000):
          if j<5000:
              td_mse[i,j] = mean_squared_error(td_vals[i,j], dp_value)
          mc_mse[i,j] = mean_squared_error(mc_vals[i,j], dp_value)
      
  mean_td_mse = np.mean(td_mse, axis=0)
  td_mse_std = np.std(td_mse, axis=0)
  mean_mc_mse = np.mean(mc_mse, axis=0)
  mc_mse_std = np.std(mc_mse, axis=0)

  plt.plot(np.arange(10000), mean_mc_mse, label="MC agent")
  plt.fill_between(np.arange(10000), mean_mc_mse-mc_mse_std, mean_mc_mse+mc_mse_std, alpha=0.5)
  plt.plot(np.arange(5000), mean_td_mse, label="TD agent")
  plt.fill_between(np.arange(5000), mean_td_mse-td_mse_std, mean_td_mse+td_mse_std, alpha=0.5)
  plt.xlabel("Episode")
  plt.ylabel("MSE with respect to DP value function")
  plt.legend()
  plt.grid()
  plt.title("Estimation errors of MC and TD agents")
  plt.show()

  td_agent = TD_agent()
  mc_agent = MC_agent()

  _, tval, trew = td_agent.solve(maze)
  _, mval, mrew = mc_agent.solve(maze)

  td_mse = [mean_squared_error(tval[i], dp_value) for i in range(len(tval))]
  mc_mse = [mean_squared_error(mval[i], dp_value) for i in range(len(mval))]

  def mov_avg(a, n=500):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:]-ret[:-n]
    return ret[n-1:]/n

  plt.scatter(mrew, mc_mse, s=0.01, label="MC agent")
  plt.scatter(trew, td_mse, s=0.01, label="TD agent")
  plt.scatter(mov_avg(trew), mov_avg(td_mse), s=0.01, label="TD agent moving average")
  plt.scatter(mov_avg(mrew), mov_avg(mc_mse), s=0.01, label="MC agent moving average")




  plt.xlabel("Total undiscounted sum of rewards")
  plt.ylabel("Mean square error")
  plt.title("Estimation errors vs total rewards for learning agents")
  plt.grid()
  plt.legend(loc='upper right',markerscale=50)
  plt.show()
