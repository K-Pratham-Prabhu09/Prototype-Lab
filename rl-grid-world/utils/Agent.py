import random

class Agent:
    """
    This is a class written by K Pratham Prabhu, to practice deep reinforcement learning.
    The simple agent behaves in the given input envioronment and has methods to explore the environment.
    This class covers:
        1. random_movement() (100% Explore)
        2. execute_action() (Explore-Exploit Balance, with epsilon Greedy)
            - Uses choose_action()- that uses the epsilon Greedy logic.
        3. monte_carlo_update() uses:
            - Update state value using Discounted Return Calculation.
            - Monte Carlo Prediction.
        4. get_greedy_policy() updates the policy with experience.
    """
    def __init__(self,env,gamma=0.9):
        self.env = env
        self.agent_pos = self.env.start # Agent at the start position
        self.action_set = []
        self.prev_states_list = []
        self.c_state_list = []
        self.c_reward_list = []
        self.V = {state:0 for state in self.env.grid.keys()}
        self.greedy_grid = {state:0 for state in self.env.grid.keys()}

    def reset(self):
        self.agent_pos = self.env.start
        self.action_set = []
        self.prev_states_list = []
        self.c_state_list = []
        self.c_reward_list = []
        
    def execute_action(self,verbose=False,epsilon=0.1,stopping_iter = 250):
        isDone = False
        iter = 1
        self.reset()
        curr_prev_state = self.env.start
        G = 0 #Return
        while not isDone:

            if iter==stopping_iter:
                print(f"[Warning] Force Stopping after {stopping_iter} iterations, goal not found...")
                break
            action = self.choose_action(epsilon,self.agent_pos,verbose)
            if verbose:
                print(f"[{iter}]:Attempting {self.env.action_map[action]} at {self.agent_pos}")
            feedback = self.env.move(action,self.agent_pos,verbose)
            if feedback != None:
                new_agent_pos, reward, isDone = feedback
                self.temporal_difference_zero_update(self.agent_pos, reward, new_agent_pos, isDone,verbose=verbose)
                curr_action = self.env.action_map[action]
                self.action_set.append(action)
                self.prev_states_list.append(curr_prev_state)
                self.c_state_list.append(new_agent_pos)
                self.c_reward_list.append(reward)
                self.agent_pos = new_agent_pos
                curr_prev_state = new_agent_pos
                G+=reward
            else:
                if verbose:
                    print("Invalid Action")
            iter+=1
        else:
            if verbose:
                print(f"GOAL ATTAINED IN {iter} iterations and {len(self.action_set)} actions with return G = {G}")
        
    def monte_carlo_update(self,gamma = 0.9,alpha=0.1): #Implementing first Visit Monte Carlo, gamma= discount factor
        G = 0
        for t in reversed(range(len(self.c_state_list))): #We move backwards
            s = self.c_state_list[t]
            r = self.c_reward_list[t]
            G = r + gamma*G 
            if s not in self.c_state_list[:t]: #Main logic of first visit monte carlo.
                curr_V = self.V.get(s,0)
                new_V = curr_V+alpha*(G-curr_V)
                self.V[s] = new_V

    def temporal_difference_zero_update(self, s, r, s_prime, isDone, alpha=0.1, gamma=0.9,verbose=False):
        curr_V = self.V.get(s, 0)
        # If we reach the goal, we are done.
        if isDone:
            next_V = 0 
        else:
            next_V = self.V.get(s_prime, 0)
            self.V[s] = curr_V + alpha * ((r + gamma * next_V) - curr_V)
            if verbose:
                print(f"TD || {s}[V={round(curr_V,4)}]||{s_prime}[V={round(next_V,4)}] || V new for {s}={round(self.V[s],4)}")
                
    def show_valid_actions(self):
        for idx in range(len(self.action_set)):
            curr_action = self.env.action_map[self.action_set[idx]]
            # print(f"From {self.prev_states_list[idx]} 'A' did {curr_action}")
        self.env.show_move(self.prev_states_list,self.action_set) 

    def get_greedy_policy(self):
        for pos in self.V.keys():
            #First get the nearest neighbor positions
            if self.env.is_goal(pos):
                action = 0
            else:
                nearby_cells =self.env.get_nearby_cells(pos)
                nearby_values = [self.V.get(pos_i, float('-inf')) for pos_i in nearby_cells]
                besy_value = max(nearby_values)
                action = nearby_values.index(besy_value)+1
            self.greedy_grid[pos] = action

    def choose_action(self,epsilon,pos,verbose=False):
        r_val = random.uniform(0, 1)
        if r_val<epsilon: #Explore
            action = random.randint(1, 4)
            if verbose:
                print(f"Exploring at {pos} chosen action: {self.env.action_map[action]}")
        else: #Exploit
            action = self.greedy_grid[pos]
            if action==0:
                action = random.randint(1, 4)
                if verbose:
                    print(f"Exploring at {pos} chosen action: {self.env.action_map[action]}")
        return action
