#GA Implementation
import numpy as np
import math 
import random 
import copy
import matplotlib.pyplot as plt
import time
import sys
vf = 16 #wind speed
length = 2000 #width and length
k = 0.094 #wake decay coff
p = 1.225 #air dinsity
Cef = 0.4 #eff
Ct= 0.88 #thrust coff
start_time = time.time()
global  pr ,D , H ,N_r,recorded_best_fitness,time_recorded
def read_data():
    global  pr ,D , H, N_r
    pr = 2 #megawatt
    D = 63.6
    H = 60
    N_r = int(math.floor(length / (5*D)))
def Grid(D):
   grid_layout = np.zeros((N_r,N_r))
   return grid_layout
#    
def Cost(N_turbines):
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
                            break
                        c_j=c_j-1
                    s= 5.*D + ((5.*D)*c_empty)
                    Dwk[i][j]=D +  (2 * k * s)
                    Vdf[i][j]=vf * ((1 - (math.sqrt(1-Ct))) * ((D/(Dwk[i][j]))**2))
                    P_matrix [i][j]=0.5 * p * A * Cef * ((vf-Vdf[i][j])**3)        
                        
                else:
                    Vdf[i][j] = 0
                    P_matrix[i][j] = 0           
    return P_matrix 
def Power(P_matrix):
    p_total = np.sum(P_matrix ) *0.001
    return p_total
    
def Obj_func (grid_layout ):
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

def random_sol(grid,N_turbines):
    no_ones =0
    while no_ones < N_turbines :
        for i in range(N_r):
            if no_ones == N_turbines :
                    break
            for j in range(N_r):
                if no_ones == N_turbines :
                    break
                if random.uniform(0,1) < 0.7 and grid[i][j]!=1:
                    grid[i][j]=1
                    no_ones +=1
                else:
                    grid[i][j]=0
                 
    return grid
def Selection(G):   #Ep======Elite presentage ##Mp===Worst presentage
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
  
def Crossover(sorted_chr,Cp):
    n_children_cross=int(Cp*len(sorted_chr))
    crossover = []
    sorted_temp = copy.deepcopy(sorted_chr)
    i=0
    while(i<n_children_cross):
        temp1 = copy.deepcopy(sorted_temp[i])
        temp2 = copy.deepcopy(sorted_temp[i+1])
        for j in range (len(sorted_chr[i])):
            if np.random.rand()<0.7:
                temp_row= copy.copy(temp1[j])
                temp1[j] = temp2[j]
                temp2[j] = temp_row
        if Obj_func(temp1) < Obj_func(temp2):
            crossover.append(temp1)
        else:
            crossover.append(temp2)
        i = i +1
    return crossover
                

                
        
        


def Mutation(sorted_chr,Mp):
     n_mutated=int(Mp*len(sorted_chr))
     i = len(sorted_chr)-1
     counter = 0
     mutated = []
     temp = copy.deepcopy(sorted_chr)
     
     while(counter < n_mutated):
         for j in range(N_r):
             for k in range(N_r):
                 xz = np.random.rand()
                 if xz < 0.5:
                     if temp[i][j][k] ==1:
#                         print temp[i][j][k] 
                         temp[i][j][k] =0
#                         print temp[i][j][k]    
                     else :
                        temp[i][j][k] = 1
                        
                     
         mutated.append(temp[i])
         i=i-1
         counter = counter+1               
     return mutated        
         
         
         
         
    
def GA(imax,N_chr,Mp,Cp,Ep):
    global recorded_best_fitness,time_recorded
    read_data()
    G=[]
    best_so_far_sol=[]
    best_so_far_fitness = 1000
    recorded_best_fitness = []
    time_recorded =[]
    for f in range(10):
        G=[]
        N_turbines = random.randint((N_r**2)//2,N_r**2)
        print "This is iteration No", f
        for i in range(N_chr):
            grid_layout=Grid(D)
            initial_solution = random_sol(grid_layout,N_turbines)
            G.append(initial_solution)
        for i in range (imax):
            sorting=Selection(G)
            Best_sols_ever=[]
            for j in range (int(Ep*len(G))):

                Best_sols_ever.append(sorting[j])
            Crossedover=Crossover(sorting,Cp)
            for q in range (len(Crossedover)):
                Best_sols_ever.append(Crossedover[q])

            Mutated=Mutation(sorting,Mp)
            for k in range (len(Mutated)):
                Best_sols_ever.append(Mutated[k])

            Best_sols_ever = Selection(Best_sols_ever)
            if Obj_func(Best_sols_ever[0]) < best_so_far_fitness:
                best_so_far_sol = Best_sols_ever[0]
                best_so_far_fitness = Obj_func(best_so_far_sol)   
                print "The Best So far solution",best_so_far_fitness
            recorded_best_fitness.append(best_so_far_fitness)
            time_recorded.append((time.time()-start_time)/60)
#    plt.plot(recorded_best_fitness)
    return best_so_far_sol    
            
        


     
#best = GA(100,10,0.2,0.6,0.2)
#countt = 0
#for i in range (N_r):
#    for j in range (N_r):
#        if best[i][j]==1:
#            countt=countt+1 
#print 'best_so_far_N_turbines ', countt
#best_fit = Obj_func(best)
#fig=plt.figure()
#sca=fig.add_subplot(1,1,1)
#for i in range(len(best)):
#    for j in range (len(best[i])):
#        if best[i][j]==1:
#            sca.scatter(i,j,marker='1',s=500,label="Grid Layout")
#
#plt.show()     
#p_m=Wake_effect(best)   
#Total_power = Power(p_m)
#print " The Total Power is : ",Total_power    
#Total_cost =Cost(countt)
#print"The Total Cost is : ",Total_cost


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
        best = GA(100,10,0.2,0.6,0.2)
        countt = 0
        for ii in range (N_r):
            for j in range (N_r):
                if best[ii][j]==1:
                    countt=countt+1 
        best_fit = Obj_func(best)
        p_m=Wake_effect(best)   
        Total_power = Power(p_m)
        Total_cost =Cost(countt)
        myscript(iii,best_fit,countt,Total_power,Total_cost,recorded_best_fitness,time_recorded)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
        
        
        