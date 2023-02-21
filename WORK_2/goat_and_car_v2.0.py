# -*- coding: utf-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt

'''if "setup_text_plots" not in globals():
    from astroML.plotting import setup_text_plots
setup_text_plots(fontsize=14, usetex=True)
%config InlineBackend.figure_format='retina' # very useful command for high-res images'''

N=10000
n_door=3
n_win_door=1
n_open=1

def Selectrandom(names):

        return (random.choice(names))

#def TheGoatGame(N,n_door,n_win_door, n_open):
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
    player_2=np.random.random_integers(0,n_door-1,n_win_door) # choosen door by the player 2
    
    prob_win_player_1=n_win_door/n_door # the P that the first playe win is 1/3
    prob_win_player_2=n_win_door/n_door # the P that the second player win is 1/3
    
    
    loosing_door=[]
    # Here, I find the doors that contains a goat
    for i in range(n_door):
        if door[i]==0:
            loosing_door.append(i)
    
    
    opened_door=Selectrandom(loosing_door)    # index of the opened door in the loosing_door list
    remaining_door=[]
    
    j=0
    
    for j in range(n_door):
        if j!=opened_door:
            remaining_door.append(door[j])
            
    conservative=player_1
    switcher=player_2
    newcamer=Selectrandom(remaining_door )
    
    while switcher==player_2:
        switcher=Selectrandom(remaining_door) 
    
    winner=[] 
    
    
    if conservative==winning_door:
        winner.append('conservative')
        n_win_conservative+=1
    elif switcher==winning_door:
        winner.append('switcher')
        n_win_switcher+=1
    elif newcamer==winning_door:
        winner.append('newcamer')
        n_win_newcamer+=1
    else:
        winner.append('loosers')
        n_loss+=1
    
    
    
    win.append(winner)
    
result=('P(goat and car)=%1.3f \n P(car)=%1.3f' % (len(loosing_door)/n_door * len(winning_door)/(n_door-1), len(winning_door)/n_door))   
plt.bar(['conservative','switcher','newcamer'], [n_win_conservative/N,n_win_switcher/N,n_win_newcamer/N],alpha=0.5, label=result)
plt.ylabel('normalized # of wins')
plt.legend()
plt.title('rule: %1.0f %1.0f %1.0f' %  (N,n_door,winning_door))


print('P(goat and car)=%1.3f' % (len(loosing_door)/n_door * len(winning_door)/(n_door-1)))
#return(print(result))

#TheGoatGame(100,3,1,1)
#TheGoatGame(100, 100, 1,98)

    















    