import numpy as np

class GridWorld:
    """
    This is a class written by K Pratham Prabhu, to practice deep reinforcement learning.
    The simple Grid world creates an envioronment for the agent to move and learn. 
    1. We can set the Start State using set_start() 
    2. Goal state and the reward associated using set_goal()
    3. We can let the agent move in the grid environment using move() method.
    """
    def __init__(self, rows=3, columns=3):
        self.rows = rows
        self.columns = columns
        self.goal = ()
        self.start = ()
        self.rewards = []        
        self.action_map = {0:'-',1:'↑',2:'↓',3:'←',4:'→'}
        self.step_cost = -0.1
        self.boundry_cost = -0.5
        self.grid = self.prepare_grid()

    #This prepares the grid, ie a dictionary 
    # The grid has grid cell (tuples) as the key and state value as the value
    def prepare_grid(self): 
        grid = {}
        for row in range(self.rows):
            for column in range(self.columns):
                grid[(row,column)] = [self.step_cost,'.'] #Key is a position tuple, value= (reward,name).
        return grid
    
    def set_goal(self,cell,reward = 0):
        if not self.is_pos_valid(cell):
            print(f"Invalid goal Position {cell}!!")
            return
        if self.start==cell:
            print("Goal and start can not be the same cell!!")
            return
        self.goal = cell
        self.set_reward(cell,reward)
        self.set_cell_name(cell,'G')

    def set_start(self,cell, reward = 0):
        if not self.is_pos_valid(cell):
            print(f"Invalid start Position {cell}!!")
            return
        if self.goal==cell:
            print("Goal and start can not be the same cell!!")
            return
        self.start = cell
        self.set_reward(cell,reward)
        self.set_cell_name(cell,'S')

    def set_reward(self,cell,reward = 0):
        self.grid[cell][0] = reward

    def set_cell_name(self,cell,name):
        self.grid[cell][1] = name

    def set_penalty(self,step_cost=-0.1,boundry_cost=-0.5):
        self.step_cost = step_cost
        self.boundry_cost = boundry_cost
        self.grid = self.prepare_grid()

    ############ VALIDATION Related Methods ############
    def is_start(self,cell): #cell must be a tuple  
        """
        Check If input cell is at the start
        """
        if self.start:
            if self.start==cell:
                return True
        return False

    def is_goal(self,cell): #cell must be a tuple
        """
        Check If input cell is at the goal
        """
        if self.goal:
            if self.goal==cell:
                return True
        return False
       
    def is_pos_valid(self,cell):
        """
        Check If input cell inside the grid
        """
        if cell[0]>=self.rows or cell[1]>=self.columns:
            return False
        elif cell[0]<0 or cell[1]<0:
            return False
        return True

    ############ MOVEMENT related Methods ############
    def move(self,action,agent_pos, verbose=False):
        r = agent_pos[0] #get current agent position
        c = agent_pos[1]
        old_cell = (r,c)
        isDone =  False
        match action:
            case 1: #up
                r-=1
            case 2: #Down
                r+=1
            case 3: #left
                c-=1
            case 4: #right
                c+=1
            case _:
                return None
        new_cell = (r,c) #update
        # print(f"Attempt:{old_cell}->{new_cell}")            
        if self.is_pos_valid(new_cell):
            agent_pos = new_cell
            reward = self.grid[agent_pos][0]
            if self.is_goal(agent_pos):
                isDone =  True
            return (agent_pos, reward, isDone)
        else:
            if verbose:
                print(f"Invalid Action| Penalty={self.boundry_cost}")
            return (agent_pos, self.boundry_cost, False)

    def get_nearby_cells(self,cell):
        r = cell[0]
        c = cell[1]

        c_up = (r-1,c)
        c_down = (r+1,c)
        c_left = (r,c-1)
        c_right = (r,c+1)

        cell_up = c_up if self.is_pos_valid(c_up) else ()
        cell_down = c_down if self.is_pos_valid(c_down) else ()
        cell_left = c_left if self.is_pos_valid(c_left) else ()
        cell_right = c_right if self.is_pos_valid(c_right) else ()

        return [cell_up,cell_down,cell_left,cell_right]

    def grid_to_action_map(self,G):
        action_grid = self.prepare_grid()
        for c,v in G.items():
            action_grid[c] = self.action_map[v]
        return action_grid


    ############ DISPLAY related methods ############
    def show_grid(self):
        for c,v in self.grid.items():
            cell_name = v[1]
            print(f"[{cell_name}]",end="")
            if c[1]==self.columns-1:
                print("\n")
                
    def show_rewards(self):
        for c,v in self.grid.items():
            r = v[0]
            if self.is_goal(c) and not self.is_start(c):
                print(f"[#{r}]",end="")    #Point the goal
            elif not self.is_goal(c) and self.is_start(c):
                print(f"[^{r}]",end="")    #Point the start
            else:
                print(f"[{r}]",end="")
            if c[1]==self.columns-1:
                print("\n")

    def print_grid_contents(self,G): #to print any grid G, G is dict
        for c,v in G.items():
            if isinstance(v, (int, float)):
                print(f"[{round(v,2)}]",end="")
            elif isinstance(v, str):
                 print(f"[{v}]",end="")
            if c[1]==self.columns-1:
                print("\n")

    def show_move(self,prev_cells,actions):
        temp_grid = self.prepare_grid()
        for idx in range(len(actions)):
            curr_action = self.action_map[actions[idx]]
            curr_prev_cell = prev_cells[idx]
            name = temp_grid[curr_prev_cell][1]
            if name=='.':
                temp_grid[curr_prev_cell][1] = curr_action
            else:
                temp_grid[curr_prev_cell][1] += curr_action

        for c,v in temp_grid.items():
            if self.is_goal(c):
                print(f"[G]",end="")
            else:
                cell_name = v[1]
                print(f"[{cell_name}]",end="")
            if c[1]==self.columns-1:
                print("\n") 
    