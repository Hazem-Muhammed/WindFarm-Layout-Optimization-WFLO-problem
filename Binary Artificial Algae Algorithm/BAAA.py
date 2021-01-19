# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 08:14:23 2021

@author: ElMousel
"""

import numpy as np
import math 
import random 
import copy
import matplotlib.pyplot as plt
import time
import sys
np.seterr('raise')
#import time
#start_time = time.time()
vf = 16 #wind speed
length = 2000 #width and length
k = 0.094 #wake decay coff
p = 1.225 #air dinsity
Cef = 0.4 #eff
Ct= 0.88 #thrust coff
start_time = time.time()
np.seterr(divide="ignore")
global  D , H ,N_r,BestFitIter,time_recorded
def read_data():
    global  D , H, N_r
    D = 63.6
    H = 60
    N_r = int(math.floor(length / (5*D)))
def Grid(D):
   grid_layout = np.zeros((N_r,N_r))
   return grid_layout
#    
def Cost(N_turbines):
#    A = np.pi * ((D/2)**2)
    #c= (N_turbines)*((1170*pr)+(1.5 * ( (0.016) * (D**2.8) * ((H/D)**(1.7)) * ((pr/A)**(0.6)))))
    c = (N_turbines * ((2./3)+((1./3) * (np.exp((-0.00174*((N_turbines)**2)))))))
    return c
def Wake_effect(grid_layout):
    A = (3.14) * ((D/2)**2) 
    P_matrix = np.zeros((N_r,N_r))
    Vdf = np.zeros((N_r,N_r))
    Dwk = np.zeros((N_r,N_r))
    for i in range(N_r):
        for j in range(N_r):
            if j==0 and grid_layout[i][j] == 1:
                P_matrix [i][j]=0.5* p * A * Cef * (vf**3)# * (10**-6) #megawatt
            else:    
                if(grid_layout[i][j] == 1):
                    c_j = j
                    c_empty = 0
                    while c_j > 0:
                        if grid_layout[i][c_j-1] == 0 :
                            c_empty +=1
                        else:
                            s= 5.*D + ((5.*D)*c_empty)
                            Dwk[i][j]=D +  (2 * k * s)
                            Vdf[i][j]=Vdf[i][j]+math.pow((vf * ((1 - (math.sqrt(1-Ct))) * ((D/(Dwk[i][j]))**2))),2)
                            c_empty +=1

                        c_j=c_j-1

                    P_matrix [i][j]=0.5 * p * A * Cef * ((vf-(math.sqrt(Vdf[i][j])))**3)        
                        
                else:
                    Vdf[i][j] = 0
                    P_matrix[i][j] = 0           
    return P_matrix 
def Power(P_matrix):
    p_total = np.sum(P_matrix ) *0.001
    return p_total
    
def Obj_func (grid_layout):
    cou = 0
    for i in range (N_r):
        for j in range (N_r):
            if grid_layout[i][j]==1:
                cou=cou+1 
    P_matrix = Wake_effect(grid_layout)
    pow = Power(P_matrix)
    co = Cost(cou)
    obj = co/pow
    return obj
def get_S(x):
    S=0
    try:
        S=round(1/(1+(math.exp(-(x)))),10)
    except OverflowError:
        S = 0   
    return S
def update_S_grid(x):
    grid_rand = copy.deepcopy(x)
    grid_sol = Grid(D)
    grid_s = Grid(D)
    for i in range(N_r):
        for j in range(N_r):
            grid_s[i][j] = get_S(grid_rand[i][j])
            grid_sol[i][j] = update_cell(grid_s[i][j])
            
    return grid_sol
def update_cell(y):
    if random.uniform(0,1)>= y:
        return 0
    else: 
       return 1
def initial_x():
    
    grid_eq = Grid(D)
    for i in range(N_r):
        for j in range(N_r):
            grid_eq[i][j] = (random.uniform(0,1))
    return grid_eq

def initial_sol(x):
    grid_rand = copy.deepcopy(x)
    grid_s = Grid(D)
    grid_sol = Grid(D)
    for i in range(N_r):
        for j in range(N_r):
            grid_s[i][j] = get_S(grid_rand[i][j])
    for i in range(N_r):
        for j in range(N_r):
            grid_sol[i][j] = update_cell(grid_s[i][j])
    return grid_sol
def helical_movement(xi,xj,ta,delta):
    s = xi + ((xi - xj)*(delta-ta)*(random.uniform(-1,1)))
    return s
    
def eval_fit(x,b,w):
    
    x_obj = Obj_func(x)
    b_obj = Obj_func(b)
    w_obj = Obj_func(w)
    return 1-((x_obj-w_obj)/(b_obj-w_obj))
def Selection(G):
  fitness_arr=[]
  sorted_chr=[]
  for i in range(len(G)) :
      fitness_arr.append(Obj_func(G[i]))
  sort=sorted(fitness_arr)
  
  for i in range (len(G)):
      for j in range(len(G)):
          if sort[i]==Obj_func(G[j]):
              sorted_chr.append(G[j])
              break
   
  
  return sorted_chr
def tournement_selection(source):

    index1= random.randint(0,len(source)-1)
    index2= random.randint(0,len(source)-1)
    while index1 == index2:
      index2= random.randint(0,len(source)-1)  
    if Obj_func(source[index1]) < Obj_func(source[index2]):
        choice = index1
    else:
        choice = index2
    return choice
def random_pair(size):
    pairs = []
    x1 = random.randint(0,size-1)
    pairs.append(x1)
    x2 = random.randint(0,size-1)
    while x2 ==x1:
        x2 = random.randint(0,size-1)
    pairs.append(x2)
    x3 = random.randint(0,size-1)
    while x3 ==x1 or x3 ==x2  :
        x3 = random.randint(0,size-1)
    pairs.append(x3)
    return pairs
   
def BAAA(imax,e,delta,m,Ap):
    global BestFitIter,time_recorded
    read_data()
    Gbest=[]
    Gworst=[]
    bestsol=[]
    BestFitIter=[]
    time_recorded =[]
    best_so_far_sol=[]
    best_so_far_fitness = 1
    for z in range(10):
#        N_turbines = random.randint((N_r**2)//2,N_r**2)
        Generation=[]
        pp_G = []
        f=[]
        G = []
        A_star=[]
        K=[]
        for i in range(m):
            pp = initial_x()
            pp_G.append(pp)
            initial_solution = initial_sol(pp)
            Generation.append(initial_solution)
            G.append(1)
            A_star.append(0)
            K.append(0.5)
        sol = copy.deepcopy(Generation)
        Gbest=(Selection(sol))[0]
        Gworst= (Selection(sol))[len(sol)-1]
        for i in range(m):
            f.append(eval_fit(sol[i],Gbest,Gworst))
        for i in range(imax):
            Energy= 1
            for j in range(m):
                neighbour = tournement_selection(sol)
                starvation = True
                tau = (2*math.pi)*((0.75*G[j]/math.pi)**2.0/3)
                while Energy >= 0 :
                    pairs_x = random_pair(N_r)
                    for k in range(3):
                        yy =random.randint(0,N_r-1)
                        xii = pp_G[j][pairs_x[k]][yy]
                        xkk = pp_G[neighbour][pairs_x[k]][yy]
                        pp_G[j][pairs_x[k]][yy] = helical_movement(xii,xkk,tau,delta)
                    Energy = Energy -(e/2.)
                    sol_temp = update_S_grid(pp_G[j])
                    if Obj_func(sol_temp) < Obj_func(sol[j]):
                        sol[j] = sol_temp
                        starvation = False
                    else:
                        Energy = Energy -(e/2.)
                
                if starvation==True:
                    A_star[j]=A_star[j]+1
                
            s_i = G.index(min(G))
            b_i = G.index(max(G))
            starving_i = A_star.index(max(A_star))

            for l in range(m):
                f.append(eval_fit(sol[l],Gbest,Gworst))
                K[l]=G[l]/2.0
                G[l]=G[l]+ (G[l]*(f[l]/(K[l]-f[l])))

            x_s = random.randint(0,N_r-1)
            y_s = random.randint(0,N_r-1)
            pp_G[s_i][x_s][y_s]= pp_G[b_i][x_s][y_s]
            pp_G = np.array(pp_G)
            if Ap > random.uniform(0,1):
                pp_G[starving_i] = pp_G[starving_i] + (random.uniform(0,1)*(pp_G[b_i] - pp_G[starving_i]))
            pp_G = pp_G.tolist()
            for i in range(len(pp_G)-1):
                sol[i] = update_S_grid(pp_G[i])
#            sol = Selection(sol)
            Gbest=(Selection(sol))[0]
            Gworst= (Selection(sol))[len(sol)-1]
            if Obj_func(Gbest) < best_so_far_fitness:
                best_so_far_fitness = Obj_func(Gbest)
                best_so_far_sol = copy.deepcopy(Gbest)
            BestFitIter.append(best_so_far_fitness)
            time_recorded.append((time.time()-start_time)/60)
    bestsol=copy.deepcopy(best_so_far_sol)
#    plt.plot(BestFitIter)
    return bestsol
            
#best = BAAA(100,0.1,2,10,0.1)
#print"best fittness is: ", Obj_func(best)
#countt = 0
#for i in range (N_r):
#    for j in range (N_r):
#        if best[i][j]==1:
#            countt=countt+1 
#print 'best_so_far_N_turbines ', countt
#fig=plt.figure()
#sca=fig.add_subplot(1,1,1)
#for i in range(len(best)):
#    for j in range (len(best[i])):
#        if best[i][j]==1:
#            sca.scatter(i,j,marker='1',s=500,label="Grid Layout")
#
#plt.show()        
                    
def myscript(iteration_number,fit,N_t,T_p,T_c,best_sols,best_time):
    xfile_name = "Run%d.txt" % iteration_number
    with open(xfile_name, "w") as xf:
        xf.write("The best Fittness is "+repr(fit)+ ".\n")
        xf.write("****************************************\n")
        xf.write("The best Number of turbines is "+repr(N_t)+ ".\n")
        xf.write("****************************************\n")
        xf.write("The Total power is "+ repr(T_p)+ ".\n")
        xf.write("****************************************\n")
        xf.write("The Total cost is "+repr(T_c)+ ".\n")
        xf.write("****************************************\n")
        xf.write("The Recoreded best solutions are \n"+repr(best_sols)+ ".\n")
        xf.write("****************************************\n")
        xf.write("The time over iterations\n"+ repr(best_time)+ ".\n")
        xf.write("****************************************\n")
    return 0
def main(unused_command_line_args):
    for iii in xrange(10 ):
        best = BAAA(100,0.1,2,10,0.1)
        countt = 0
        for ii in range (N_r):
            for j in range (N_r):
                if best[ii][j]==1:
                    countt=countt+1 
        best_fit = Obj_func(best)
        p_m=Wake_effect(best)   
        Total_power = Power(p_m)
        Total_cost =Cost(countt)
        myscript(iii,best_fit,countt,Total_power,Total_cost,BestFitIter,time_recorded)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))               
            


          
            
                    
                
                
    