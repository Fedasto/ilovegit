#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:54:21 2022

@author: federicoastori
"""
import numpy as np
import matplotlib.pyplot as plt



#if "setup_text_plots" not in globals():
#    from astroML.plotting import setup_text_plots
#setup_text_plots(fontsize=14, usetex=False) #False if I'don't want to use latex
%config InlineBackend.figure_format='retina' # very useful command for high-res images

N=100000 # number of picks
N_doors=3

#Simulate three doors, one car, and two goats. 

door_1=int(np.random.random_integers(0,1,1)) #0=CAR 1=GOAT

if door_1==0:
    door_2=1
    door_3=1
elif door_1==1:
    door_2=int(np.random.random_integers(0,1,1))
    if door_2==0:
        door_3=1
    elif door_2==1:
        door_3=0
        
doors=[door_1,door_2,door_3]

player_1=(np.random.random_integers(1,3,N))


losing_door=[]
winning_door=234

for i in range(N_doors):
    if doors[i]==0:
        winning_door=i+1
    else:
        losing_door.append(i+1)
        

opened_door=int(np.random.random_integers(0,1,1)) # delle loosing doors

for u in range(N_doors):
    while opened_door==player_1[u]:
        opened_door=int(np.random.random_integers(0,1,1))

not_opened_door=234

if losing_door[0]==opened_door:
     not_opened_door=losing_door[1]
else:
     not_opened_door=losing_door[0]

remaining_doors=[winning_door, not_opened_door]

conservative=player_1
switcher=np.empty(len(player_1))
for u in range(N_doors):
    if conservative[u]==remaining_doors[0]:
        switcher[u]=remaining_doors[1]
    else:
        switcher[u]=remaining_doors[0]
        
newcamer=(np.random.random_integers(0,1,N))

pick=np.empty([N,N_doors])

for temp in range(N):
    pick[temp]=[switcher[temp],conservative[temp],newcamer[temp]]

#Record who wins.

winner=np.empty([N,3])

for i in range(N):
    for j in range(N_doors):
        if pick[i][j]==1:
            winner[i][j]=j
        else:
            winner[i][j]='nan'
        
print('the winner is player #',winner) 

n_win_switcher=len(np.where(winner==0)[0]) #how many times player switcher wins
n_win_conservative=len(np.where(winner==1)[0]) #how many times player switcher wins
n_win_newcamer=len(np.where(winner==2)[0]) #how many times player switcher wins


print('player switcher win # =', n_win_switcher ,'\n','player conservative win # =' , n_win_conservative, '\n', 'player newcamer win #=', n_win_newcamer)

hist_win=[n_win_switcher/N, n_win_conservative/N, n_win_newcamer/N] #array of victories per player normalized to N

ax=plt.subplot(111) #plot th histogram
plt.bar(['switcher','conservative','newcamer'],hist_win, fill=True, color='red', alpha=0.5) 
plt.xlabel('players')
plt.ylabel('# of wins normalized')
plt.title('Histogram of winning')
plt.ylim(0,1)
plt.show()






