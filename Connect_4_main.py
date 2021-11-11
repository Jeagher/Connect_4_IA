
# Connect 4 main program


# Imports 
import pygame
import math
import sys
from collections import namedtuple, OrderedDict
from itertools import count
import time
import torch.optim as optim
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#import tesorflow as tf 

INFINITY = 1_000_000
BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)
SQUARESIZE = 100
WIDTH = 7 * SQUARESIZE
HEIGHT = 7 * SQUARESIZE
SIZE = (WIDTH, HEIGHT)

RADIUS = int(SQUARESIZE/2 - 5)

class Connect4():
    def __init__(self):
        self.height = 6 
        self.width = 7
        self.board = self.create_board()
        self.turn = 0
        self.done = False
        # For conv2d op tensor must be of rank 4 BCHW
        self.kernel_dic = {'horizontal' : torch.tensor([1 ,1 ,1 ,1], dtype=torch.float32).reshape(1,1,1,4), 
                           'vertical' : torch.tensor([1 ,1 ,1 ,1], dtype=torch.float32).reshape(1,1,4,1),
                           'main_diag' : torch.eye(4, dtype=torch.float32).reshape(1,1,4,4),              
                           'second_diag':  torch.eye(4, dtype=torch.float32).fliplr().reshape(1,1,4,4)}
        
                            
    def create_board(self):
        return torch.zeros((self.height,self.width), dtype=torch.float32)

    def reset(self):
        self.turn = 0
        self.done = False
        self.board = self.create_board() 

    def get_valid_action(self):
        return self.board[0].abs().eq(0).nonzero(as_tuple=True)[0]
    
    def action_played(self,action):
        column = self.board.index_select(dim=1, index=action).squeeze().tolist() # index_select get the column 
        for i in range(self.height-1,-1,-1):
            if column[i] == 0 :
                self.board[i][action] = 1 - 2*(self.turn%2) # auto braodcasting : 1 = torch.tensor(1)    
                self.turn += 1
                return i
            
                
    
    def reset_last_action(self,action):
        action = action.to(torch.int32)
        column = self.board.index_select(dim=1, index=action).squeeze().tolist() # index_select get the column 
        for i in range(self.height):
            if column[i] != 0 :
                self.board[i][action] = 0  
                self.turn -= 1
                break
        
    
    def conv2D(self,kernel):
        return F.conv2d(self.board.reshape(1,1,self.height,self.width),kernel)
    
    def check_end_game(self):
        if len(self.get_valid_action()) == 0 :
            return 'draw'
        for direction in self.kernel_dic :
            if self.conv2D(self.kernel_dic[direction]).eq(4).any().item() : # If there is a value equal to 4
                return 'player1_win'
            elif self.conv2D(self.kernel_dic[direction]).eq(-4).any().item() :
                return 'player2_win'
        return 'False'
    
    def draw_board(self,screen):
    	for col in range(self.width):
    		for line in range(self.height):
    			pygame.draw.rect(screen, BLUE, (col*SQUARESIZE, line*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
    			pygame.draw.circle(screen, BLACK, (int(col*SQUARESIZE+SQUARESIZE/2), int(line*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
    	
    	for col in range(self.width):
    		for line in range(self.height):		
    			if self.board[line][col] == 1:
    				pygame.draw.circle(screen, RED, (int(col*SQUARESIZE+SQUARESIZE/2), int((line+1)*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    			elif self.board[line][col] == -1: 
    				pygame.draw.circle(screen, YELLOW, (int(col*SQUARESIZE+SQUARESIZE/2), int((line+1)*SQUARESIZE+SQUARESIZE/2)), RADIUS)
    	pygame.display.update()

class IaRandom():
    
    def play(self,env):
        return random.choice(env.get_valid_action())
        
class TranspositionTable():
    
    def __init__(self):
        self.hash_table_size = 1_000_000
        self.nb_hash_table_arg = 6
        self.hash_table_arg_dic = {'key':0, 'depth':1, 'value':2, 'flag':3, 'best_moove':4, 'age':5}
        self.flag_dic = {'exact':0, 'upper_bound':1, 'lower_bound':2}
        self.hash_table = None
        self.height = 6
        self.width = 7
        self.nb_pieces = 2
        self.zob_table = self.create_zoberist_table()
        self.hash_table = self.create_hash_table()
    
    def create_hash_table(self) :
       # The hash table contains the zobrist full key, the depth, the position eval, the flag, the best moove, the age 
       return torch.zeros((self.hash_table_size,self.nb_hash_table_arg), dtype=torch.int32)

    def reset_hash_table(self):
        self.hash_table =  self.create_hash_table()

    def create_zoberist_table(self) :
        # torch.manual_seed(0)
        return torch.randint(low=1, high=(2**31) ,size = (self.height,self.width,self.nb_pieces), dtype=torch.int32)
        
    def generate_key(self,position):
        k = torch.tensor(0)
        for i in range(self.height):
            for j in range(self.width):
                if position[i][j] == 1 :
                    k ^= self.zob_table[i][j][0]
                elif position[i][j] == -1 :
                    k ^= self.zob_table[i][j][1]
        return k

    def hash_key(self,key):
        return key % self.hash_table_size

    def search(self,key):
        hash_key = self.hash_key(key)
        entry = self.hash_table[hash_key]
        zoberist_key = entry[0]
        if key == zoberist_key : # If the info in the hash table correspond to the good position 
            return entry
        else : # The position is either not stored in the table or their is a collision, collision management won't be implememnted 
            return ['None']

    def store(self,key,entry):
        try :
            hash_key = self.hash_key(key)
            hash_row = self.search(key)
            if hash_row[0] != 'None':
                hash_depth = hash_row[self.hash_table_arg_dic['depth']] 
                hash_age = hash_row[self.hash_table_arg_dic['age']]
                entry_depth =  entry[self.hash_table_arg_dic['depth']] 
                entry_age = entry[self.hash_table_arg_dic['age']]
                # if entry_age > hash_age : # The age correpond to the real turn when the research is done, then hash_entry can not be greater than entry_age
                #     self.hash_table[hash_key] = entry
                if entry_depth >= hash_depth :
                    self.hash_table[hash_key] = entry
            else :
                self.hash_table[hash_key] = entry
        except Exception as e:
            print(str(e))
            print('Wrong Entry format : should be an array of shape (6) dtype = int32')
            print(entry)
            print(entry.dtype)

class IaMinimax():
    def __init__(self,depth,player):
        self.depth = depth
        self.player = player
        self.heat_map = torch.tensor([[0, 1, 3, 3, 1, 0],  # Experimental heat map
                                      [1, 3, 6, 6, 3, 1],
                                      [2, 4, 7, 7, 4, 2],
                                      [1, 5, 9, 9, 5, 1],
                                      [1, 4, 5, 5, 4, 1]], dtype = torch.float32)
        
    def static_evaluation(self,env):
        
        def center(col,width):
            center = width //2
            if col < center :
                return col %center
            else :
                return width - col
            
        def evaluation(val,tensor): # Need to add combo diag and horizontal 
            nonzero = tensor.eq(val).nonzero().tolist()
            nonzero_copy = nonzero.copy()
            eval_ = 0
            for i in range(len(nonzero)): # Check the number of 3 horizontaly, the lower and more center the pieces the better it is 
                eval_ += 100*(2*(nonzero[i][0] + 1) +  center(nonzero[i][1],len(tensor[0])))   
            for elt in nonzero :
                if [elt[0]+1, elt[1]] in nonzero_copy : # If 2 3 are consecutive on the same column
                                                        # You have 2 connect 4 possible on the column in the direction 
                    eval_ += 10_000
                    nonzero_copy.remove([elt[0]+1, elt[1]])
            return eval_
    
        eval_ = 0
        for direction in env.kernel_dic :  
            conv2d_tensor = env.conv2D(env.kernel_dic[direction]).squeeze()
            
            if conv2d_tensor.eq(4).any().item() : # End of the game player 1 win
                return +INFINITY - env.turn
            
            elif conv2d_tensor.eq(-4).any().item() : # End of the game player 2 win 
                return -INFINITY + env.turn
            
        # return random.choice([-3,-2,-1,0,1,2,3])
            
            elif direction == 'vertical': # check if 3 connect in vertical, only at most one 3 or -3 can be found in a column of this tensor  
                nonzero_pos = conv2d_tensor.eq(3).nonzero()
                nonzero_neg = conv2d_tensor.eq(-3).nonzero()
                eval_ += 100*len(nonzero_pos)
                eval_ -= 100*len(nonzero_neg)
            
            else :
                eval_ += evaluation(3,conv2d_tensor)
                eval_ -= evaluation(-3,conv2d_tensor)
        
        eval_kernel = torch.tensor([[[[1,1],
                                      [1,1]]]], dtype=torch.float32)
        conv2d_tensor = env.conv2D(eval_kernel).squeeze()
        eval_ += torch.mul(conv2d_tensor,self.heat_map).sum().item()
        
        return eval_
    
    
    def minimax(self, env, depth, alpha, beta, maximizingPlayer): 
        if depth == 0 or env.check_end_game() != 'False' : 
            return None,self.static_evaluation(env)
     
        valid_action = env.get_valid_action()
        idx = torch.randperm(valid_action.shape[0])
        valid_action = valid_action[idx]
        
        if maximizingPlayer : 
            maxEval = -INFINITY
            best_action = random.choice(valid_action)
            for action in valid_action: 
                line = env.action_played(action)
                eval_= self.minimax(env, depth-1, alpha, beta, False)[1]
                env.board[line][action] = 0
                env.turn -= 1
                
                if eval_ > maxEval :
                    maxEval = eval_
                    best_action = action
                
                alpha = max(alpha, maxEval)
                if beta <= alpha : 
                    break
            return best_action, maxEval
            
        else : 
            minEval = INFINITY
            best_action = random.choice(valid_action)
            for action in valid_action : 
                line = env.action_played(action)
                eval_= self.minimax(env, depth-1, alpha, beta, True)[1]
                env.board[line][action] = 0
                env.turn -= 1
    
                if eval_ < minEval : 
                    minEval = eval_
                    best_action = action
                
                beta = min(beta, minEval)
                if beta <= alpha : 
                    break
            return best_action, minEval
            
    def play(self,env):
        # start_time = time.time()
        maximizingPlayer = True if self.player == 1 else False
        sign = 1 if self.player == 1 else -1
        action, eval_ = self.minimax(env, self.depth, -INFINITY, INFINITY, maximizingPlayer)
        # end_time = time.time()
        #print(f' player {self.player}, action : {best_action}, evaluation : {eval_}, computation time : {end_time - start_time} s ')
        return action
        
# Nagamx Ia == minimax with transposition table should give the same result as minimax IA but faster 
# Transposition table can generate errors, check zobrist hash tables         
class IaNegamax(IaMinimax):
    def __init__(self, depth, player):
        super().__init__(depth, player)
        self.transposition_table = TranspositionTable()
        
    # Add key as negamax arg to avoid key recomputation at each step 
    def negamax(self, env, age, depth, alpha, beta, maximizing_player):
        transposition_table = self.transposition_table
        alpha_origin = alpha
        
        # # Look if the position is stored in the hash table 
        # key = transposition_table.generate_key(env.board)
        # entry = transposition_table.search(key)
        # if entry[0] != 'None' and entry[transposition_table.hash_table_arg_dic['depth']] >= depth : # Aging management ? 
        #     entry_value = entry[transposition_table.hash_table_arg_dic['value']]
        #     entry_flag = entry[transposition_table.hash_table_arg_dic['flag']]
        #     entry_best_action = entry[transposition_table.hash_table_arg_dic['best_moove']]
        #     if entry_flag == transposition_table.flag_dic['exact'] :
        #         return entry_best_action, entry_value
        #     elif entry_flag == transposition_table.flag_dic['lower_bound'] : 
        #         alpha = max(alpha, entry_value)
        #     elif entry_flag == transposition_table.flag_dic['upper_bound'] :
        #         beta = min(beta, entry_value)
    
        #     if alpha >= beta :
        #         return entry_best_action, entry_value
    
        if depth == 0 or env.check_end_game() != 'False' :
            sign = 1 if maximizing_player else -1
            return None, sign * self.static_evaluation(env)
    
        maxEval = - INFINITY
        valid_action = env.get_valid_action()
        idx = torch.randperm(valid_action.shape[0])
        valid_action = valid_action[idx]
        best_action = random.choice(valid_action) # Implement moove ordering 
        for action in valid_action :
            line = env.action_played(action)
            eval_ = -self.negamax(env, age, depth-1, -alpha, -beta, not maximizing_player)[1]
            env.board[line][action] = 0
            env.turn -= 1
            
            if eval_ >= maxEval :
                maxEval = eval_
                best_action = action
                    
            # alpha = max(alpha, maxEval)
            # if beta <= alpha : 
                break
                    
        # Store in the hash table 
        # entry_value = maxEval
        # if maxEval <= alpha_origin :
        #     entry_flag = transposition_table.flag_dic['upper_bound']
        # elif maxEval >= beta :
        #     entry_flag =  transposition_table.flag_dic['lower_bound']
        # else :
        #     entry_flag =  transposition_table.flag_dic['exact']
        # entry_depth = depth	
        # entry = torch.tensor([key, entry_depth, entry_value, entry_flag, best_action, age])
        # transposition_table.store(key, entry)
    
        return best_action, maxEval
    
    def play(self,env):
        
         # start_time = time.time()
        maximizing_player = True if self.player == 1 else False
        action, eval_ = self.negamax(env, env.turn, self.depth, -INFINITY, INFINITY, maximizing_player)
        # end_time = time.time()
        #print(f' player {self.player}, action : {best_action}, evaluation : {eval_}, computation time : {end_time - start_time} s ')
        return action
        
# EnvManager Class
class Connect4EnvManager(Connect4):
    def __init__(self, device):
        super().__init__()
        self.device = device
        
    def get_state(self):
        return self.board.reshape(1,1,6,7)
    
    def take_action(self, action, player): 
        self.action_played(action)
        end_game = self.check_end_game()
        if end_game == 'player1_win' :
            reward = 1 if player == 1 else -1
            return torch.tensor([reward], device=self.device)
        elif end_game == 'player2_win' :
            reward = 1 if player == 2 else -1
            return torch.tensor([reward], device=self.device) # Only wining mooves reward -1000 should not happen at the end of the agent turn (kept to debug)
        for action in self.get_valid_action(): # Check if next moove lead to an immediat loose 
            line = self.action_played(action)
            end_game = self.check_end_game()
            self.turn -= 1 
            self.board[line][action] = 0
            if end_game == 'player1_win' :
                reward = 1 if player == 1 else -1
                return  torch.tensor([reward], device=self.device)
            elif end_game == 'player2_win' :
                reward = 1 if player == 2 else -1
                return torch.tensor([reward], device=self.device)
        reward = 0
        return torch.tensor([reward], device=self.device) 
    
# Epsilon Greedy Strategy class    
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
    
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay) # Exponential decay

# Experience Class
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)  

# Replay memory class
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        
    def push(self, experience): # Pile like behaviour 
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

# Agent class DQN
class Agent():
    def __init__(self, strategy, device):
        self.current_step = 0
        self.strategy = strategy
        self.device = device
        
    def select_action(self, state, valid_action, policy_net):
        state = state.to(device)
        valid_action = valid_action.to(device)
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random(): # Explore
            action = random.choice(valid_action)
            return torch.tensor([action]).to(self.device) # explore      
        else: # exploit
            with torch.no_grad(): # Turn off gradient tracking use for inferring and not training 
                action = policy_net(state).argmax(dim=1).to(self.device)
                if action in valid_action :
                    return policy_net(state).argmax(dim=1).to(self.device) 
                else :
                    action = random.choice(valid_action)
                    return torch.tensor([action]).to(self.device) # select a valid action

# Qvalues class
class QValues():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    
    @staticmethod # dont' need to create an instance of the class before to use theses methods 
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))
        # gather create a new tensor, collect the elements along axis 1 and get the output corresponding to the actions 
    
    @staticmethod        
    def get_next(target_net, next_states):                
        # batch_size = next_states.shape[0]
        # values = torch.zeros(batch_size).to(QValues.device)
        # .detach creates a tensor that shares storage with tensor that does not require grad. 
        # It detaches the output from the computational graph. So no gradient will be backpropagated along this variable.
        values = target_net(next_states).max(dim=1)[0].detach() 
        return values

# class Deep Q linear neural network 
class DQANN(nn.Module):
    def __init__(self):
        super().__init__()
         
        self.fc1 = nn.Linear(in_features=42, out_features=84)   
        self.fc2 = nn.Linear(in_features=84, out_features=168)
        self.fc3 = nn.Linear(in_features=168, out_features=84)
        self.fc4 = nn.Linear(in_features=84, out_features=42)
        self.out = nn.Linear(in_features=42, out_features=7)            

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = F.relu(self.fc4(t))
        t = self.out(t)
        return t
    
# class Deep Q convolutional neural netwotk,
# Also check Sequential model to create NN 
class DQCNN(nn.Module) :
    def __init__(self):
        super().__init__()
         
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4, padding=1)  
        self.conv1_bn = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, padding=1)
        self.conv2_bn = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)


        self.fc1 = nn.Linear(in_features=64*4*5, out_features=1024)
        #self.fc1_bn = nn.BatchNorm1d(200) # BatchNorm can't be used with batch_size = 1 which happen with inference on the policy net
        self.out = nn.Linear(in_features=1024, out_features=7)            

    def forward(self, t):
    
        t = t.reshape(-1,1,6,7)
        t = F.relu(self.conv1_bn(self.conv1(t)))
        t = F.relu(self.conv2_bn(self.conv2(t)))
        t = F.relu(self.conv3_bn(self.conv3(t)))
        t = F.relu(self.conv4_bn(self.conv4(t)))
        
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = self.out(t)
        
        return t


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)


def play(player, turn, env, color, myfont, screen):
    if player == 'player' : # If the player is human 
        for event in pygame.event.get():
           if event.type == pygame.MOUSEBUTTONDOWN:
               pygame.draw.rect(screen, BLACK, (0,0, WIDTH, SQUARESIZE))
               # Ask for Player 1 Input
               if env.turn %2 == turn:
                   posx = event.pos[0]
                   col = int(math.floor(posx/SQUARESIZE))
                    
                   if col in env.get_valid_action().tolist():
                       action = torch.tensor(col)
                       env.action_played(action)
                       env.draw_board(screen)
                    
                       if env.check_end_game() != 'False':
                           label = myfont.render(f'Player {turn} wins!!', 1, color)
                           screen.blit(label, (40,10))
                           env.done = True
                   else:
                       continue          
    else:
        action = player.play(env)
        env.action_played(action)
        env.draw_board(screen)
        if env.check_end_game() != 'False':
            label = myfont.render(f'Player {turn} wins!!', 1, color)
            screen.blit(label, (40,10))
            env.done = True

# if __name__ == '__main__' :
#     pygame.init()
#     pygame.font.init()
#     screen = pygame.display.set_mode(SIZE)
#     con = Connect4()
#     con.draw_board(screen)
#     pygame.display.update()
#     myfont = pygame.font.SysFont("monospace", 75)
    
#     player1 = IaNegamax(4, 1)
#     player2 = IaMinimax(5, 2)
    
#     while not con.done:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 sys.exit()
            
#             if player1 == 'player' or player2 == 'player' :
#                 if event.type == pygame.MOUSEMOTION:
#                     pygame.draw.rect(screen, BLACK, (0,0, WIDTH, SQUARESIZE))
#                     posx = event.pos[0]
#                     if con.turn %2 == 0:
#                         pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
#                     else: 
#                         pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
#                 pygame.display.update()
                
#         if con.turn %2 == 0 :
#             play(player1, 0, con, RED, myfont, screen)
#             time.sleep(1)
#         else :   
#             play(player2, 1, con, YELLOW, myfont, screen)
#             time.sleep(1)
                
#         if con.done:
#             pygame.time.wait(5000)
#             pygame.quit()
#             sys.exit()
   
# pygame.init()
# screen = pygame.display.set_mode(SIZE)
# pygame.display.update()

# def test_ia(Ia1,Ia2,nb_game):
#     env = Connect4EnvManager("cpu")
#     win = [0,0]
#     start_time = time.time()
#     for episode in range(1,nb_game+1):
#         env.reset()
#         # Ia1.transposition_table.reset_hash_table()

#         for step in count():
#             end = False
#             action = Ia1.play(env)
#             env.action_played(action)
#             end_game = env.check_end_game()
#             if end_game == 'player1_win':
#                 win[0] += 1
#                 break
#             elif end_game == 'draw':
#                 break
#             action = Ia2.play(env)
#             env.action_played(action)
#             end_game = env.check_end_game()
#             if end_game == 'player2_win':
#                 win[1] += 1
#                 break
#             elif end_game == 'draw':
#                 break

#         if episode % 100 == 0 :
#             end_time = time.time()
#             print(f'Ia1 win : {win[0]},  Ia2 win : {win[1]}, draw : {100-win[0]-win[1]}, time taken {end_time-start_time} s')
#             start_time = end_time
#             win = [0,0]

# test_ia(IaMinimax(2,1),IaMinimax(1,2),1000)

batch_size = 250
gamma = 0.90
eps_start = 1
eps_end = 0.01
eps_decay = 0.0001

target_update = 10
memory_size = 20_000
lr = 0.1
num_episodes = 100_000 # run for more episodes for better results

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, device)
memory = ReplayMemory(memory_size)
# player2 = IaRandom()
player1 = IaRandom()
player2 = IaMinimax(1, 2)
player3 = IaMinimax(2, 2)
player4 = IaMinimax(3, 2)
player = [player1,player2,player3,player4]
player_name = [" Random", "Minmax lv 1 ", "Minmax lv 2", "Minmax lv 3"]
em = Connect4EnvManager(device)
n = 0

# policy_net = DQCNN().to(device) 
policy_net = torch.load('./target_net').to(device)
target_net = DQCNN().to(device)
target_net.load_state_dict(policy_net.state_dict()) # Update the weight of the network with the weight of the policy network 
target_net.eval() # Set the network into eval mode 
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

win = [0,0] # agent, Ia
start_time = time.time()
for episode in range(1,num_episodes+1):
    
    em.reset()
    state = em.get_state().to(device)
    
    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         pygame.quit()
    #         sys.exit()
    # playerx = np.random.choice(player,p=[0.4,0.3,0.2,0.1])
    playerx = player[n]
    
    for timestep in count():
        end = False

        action = agent.select_action(state, em.get_valid_action(), policy_net) # Select the action, from the policy network 
        
        reward = em.take_action(action,1)
        next_state = em.get_state().to(device)
        memory.push(Experience(state, action, next_state, reward)) # Push the experience in replay memory 
        state = next_state

        end_game = em.check_end_game()
        if end_game == 'player1_win':
            win[0] +=1
            end = True
        elif end_game == 'draw':
            end = True

        if not end :
            action = playerx.play(em)
            em.action_played(action)
            #em.draw_board(screen)

            end_game = em.check_end_game()
            if end_game == 'player2_win':
                win[1] +=1
                end = True
            elif end_game == 'draw':
                end = True
        
        if memory.can_provide_sample(batch_size): # Check if we can get a sample from the replay memory 
                                                  #(if the memory is fill with at least batch_size nb of elements)
            experiences = memory.sample(batch_size) 
            states, actions, rewards, next_states = extract_tensors(experiences) # Extract these tensors
            
            current_q_values = QValues.get_current(policy_net, states, actions) # Get the Q value as a tensor 
            next_q_values = QValues.get_next(target_net, next_states) # Get next max Q value possible for the next state using the target net
            target_q_values = (next_q_values * gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1)) #MSE loss
            # loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1)) # Use MSE to calculate the loss or Huber loss less sensible to outlier in data 
            optimizer.zero_grad() # Avoid gradient accumulating 
            loss.backward() # Compute the gradient of the loss 
            optimizer.step() # Update the weight and biais 
            
        if end :
            break
        

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict()) # Updating the taget net 
        
    if episode %100 == 0:
        end_time = time.time()
        print(f'episode {episode}, IA {player_name[n]} win : {win[1]}, agent win : {win[0]}, draw : {100-win[0]-win[1]} time taken {end_time - start_time} s ')
        start_time = end_time
        if win[0] - win [1] > 90 : # At least 95 % victory
            n += 1
            agent.current_step = 0 
        win = [0,0]

        if episode %1000 == 0:
            torch.save(target_net,'./target_net') 
            pass

            if episode % 20_000 == 0 :
                agent.current_step = 0 

torch.save(target_net,'./target_net')      
# em.close()  

print('END')