#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:28:59 2022

@author: federicoastori
"""

import numpy as np
import random
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 8]

#N=1000
#n_door=3
#n_win_door=1
#n_open=1



def Selectrandom(names):
    return (random.choice(names))

def TheGoatGame(N,n_door,n_win_door, n_open):

    win=[]
    n_win_conservative=0
    n_win_switcher=0
    n_win_newcamer=0
    n_loss=0
    
    
    for temp in range(N):
        door=np.linspace(0,0,n_door,dtype=int) # GOAT
        
        winning_door=np.random.random_integers(0,n_door-1,n_win_door) 
        
        door[winning_door]=1 # CAR
        
        
        player_1=np.random.random_integers(0,n_door-1,n_win_door) # choosen door by the player 1
        #player_2=np.random.random_integers(0,n_door-1,n_win_door) # choosen door by the player 2
        
        #prob_win_player_1=n_win_door/n_door # the P that the first playe win is 1/3
        #prob_win_player_2=n_win_door/n_door # the P that the second player win is 1/3
        
        
        loosing_door=[]
        # Here, I find the doors that contains a goat
        for i in range(n_door):
            if door[i]==0:
                loosing_door.append(i)
        
        opened_door=[]
        for u in range(n_open):
            opened_door.append(Selectrandom(loosing_door))    # index of the opened door in the loosing_door list
        remaining_door=[]
        
        
        for j in range(n_door):
            if j!=opened_door[0]:
                remaining_door.append(door[j])
                
        conservative=player_1
        switcher=player_1
        newcamer=Selectrandom(remaining_door)
        
        while switcher==player_1:
            switcher=Selectrandom(remaining_door) 
        
    
        winner=[]
        
        if conservative==winning_door:
            winner.append('conservative')
            n_win_conservative+=1
        else:
            winner.append('switcher')
            n_win_switcher+=1
        
        if newcamer==switcher:
            winner.append('newcamer')
            n_win_newcamer+=1
        
        
    plt.bar(['conservative','switcher','newcamer'], [n_win_conservative/N,n_win_switcher/N,n_win_newcamer/N],alpha=0.5, color='red')
    plt.ylabel('normalized # of wins')
    plt.hlines(0.5,-0.5,2.5, label=r'P=1/2', colors='black', linestyles='dotted')
    plt.hlines(0.6666,-0.5,2.5, label=r'P=2/3',colors='black', linestyles='dotted')
    plt.hlines(0.333,-0.5,2.5,label=r'P=1/3',colors='black', linestyles='dotted')
    plt.legend(loc='best')
    plt.title('rule: %1.0f %1.0f %1.0f' %  (N,n_door,n_win_door))
    
TheGoatGame(1000, 3, 1, 1)