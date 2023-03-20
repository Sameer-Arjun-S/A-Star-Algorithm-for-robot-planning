#!/usr/bin/env python
# coding: utf-8

# In[1]:

#https://github.com/Rashmikapu/ENPM-661-A-Star-Algorithm-for-a-Mobile-Robot.git
#https://github.com/Sameer-Arjun-S/A-Star-Algorithm-for-robot-planning.git 
#Above is the link for Github repository

import cv2
import numpy as np
import heapq as hq
import copy
import math


# In[2]:


#Function to round the obtained values to the nearest 0.5 value
def round_nearest(a):
    return round(a * 2) / 2


#This function calculates the Eucilidean distance of any 2 nodes
def Euclidean_distance(node_1,node_2):
    #print(node_1,node_2)
    node1=copy.deepcopy(node_1)
    node2=copy.deepcopy(node_2)
    x2=node2[0]
    x1=node1[0]
    y1=node1[1]
    y2=node2[1]
    return (math.sqrt((y2-y1)**2+(x2-x1)**2))

#This function creates a threshold for the goal location as provided
def goal_thresh(start,goal):
    x1=start[0]
    y1=start[1]
    node_angle=start[2]
    x2=goal[0]
    y2=goal[1]
    goal_angle=goal[2]
    if((x1-x2)**2+(y1-y2)**2<=1.5**2 and node_angle==goal_angle):
        return True
    else:
        return False

#backtracking function, takes parent node from explored nodes and stores the coordinates of the path in a new list
def Backtrack( closed_list,initial_state, final_state, canvas):
    #final=[final_state[0],final_state[1]]
    #print(S)
   # print(S[tuple(final)])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # Creating video writer to generate a video.
    out = cv2.VideoWriter('A_star_rashmik.mp4',fourcc,1000,(canvas.shape[1],canvas.shape[0]))
    
    print("Total Number of Nodes Explored = ",len(closed_list)) 
    
    keys = closed_list.keys()    # Returns all the nodes that are explored
    path_stack = []    # Stack to store the path from start to goal
    
    
    start=[int(initial_state[1]),int(initial_state[0])]
    goal=[int(final_state[1]),int(final_state[0])]
    cv2.circle(canvas,tuple(start),3,(255,255,0),-1)           #draw green and red circles representing the start & goal
    cv2.circle(canvas,tuple(goal),3,(0,255,255),-1)
    # Visualizing the explored nodes
    keys = list(keys)
    for key in keys:
        canvas[int(key[0]),int(key[1])]=[255,255,255]
        cv2.imshow("A* Path Visualization",canvas)
        cv2.waitKey(1)
        out.write(canvas)

    parent_node = closed_list[tuple(final_state)]
    path_stack.append(final_state)    # Appending the final state because of the loop starting condition
    
    
    while(parent_node != initial_state):
        path_stack.append(parent_node)
        parent_node = closed_list[tuple(parent_node)]
    
    path_stack.append(initial_state)    # Appending the initial state because of the loop breaking condition
    print("\nOptimal Path: ")
    start_node = path_stack.pop()
    print(start_node)

    # Visualizing the optimal path
    while(len(path_stack) > 0):
        path_node = path_stack.pop()
        canvas[int(path_node[0]),int(path_node[1])]=[0,0,255]
        print(path_node)
        start_node = path_node.copy()
        out.write(canvas)
    
    out.release()
    
#This function checks if the node has been visited previously or not    
def check_duplicate(node,visited):
   # print(visited)
    if (visited[int(node[0]*2),int(node[1]*2),int(node[2]%30)]==1):
        return True
    
    else:
        visited[int(node[0]*2),int(node[1]*2),int(node[2]%30)]=1
        return False
    
    
#Function to create a movement of 60 degrees in the forward direction          
def forward_60( node,canvas,visited,L):
    #print(node)
    angle=node[2]-60
    if angle>360:
        angle=360-angle
    if angle<0:
        angle=360+angle
    next_node=[node[0]+L*np.cos(np.radians(node[2])),node[1]+L*np.sin(np.radians(angle)),angle]
    #print(canvas[current_node[0]][current_node[1]] )
    next_node[0]=round_nearest(next_node[0])
    next_node[1]=round_nearest(next_node[1])
    if(angle!=0):
        rounded_angle=360%angle
    else:
        rounded_angle=angle
    #next_node[2]=rounded_angle
    #node should not be in obstacle space
    #print(f"next_node:{next_node}")
    if((round(next_node[1]) > 0) and (round(next_node[0]) < 250) and (round(next_node[1]) < 600) and (round(next_node[0]) > 0)) :
        if (canvas[round(next_node[0]),round(next_node[1]),0]==0) and  (canvas[round(next_node[0]),round(next_node[1]),2]==0):
            return next_node
    else:
        return None
    
#Function to create a movement of 30 degrees in the forward direction        
def forward_30( node,canvas,visited,L):
    angle=node[2]-30
    if angle>360:
        angle=360-angle
    if angle<0:
        angle=360+angle
    next_node=[node[0]+L*np.cos(np.radians(node[2])),node[1]+L*np.sin(np.radians(angle)),angle]
    #print(canvas[current_node[0]][current_node[1]] )
    next_node[0]=round_nearest(next_node[0])
    next_node[1]=round_nearest(next_node[1])
    if(angle!=0):
        rounded_angle=360%angle
    else:
        rounded_angle=0
    #next_node[2]=rounded_angle

    #node should not be in obstacle space
    if((round(next_node[1]) > 0) and (round(next_node[0]) < 250) and (round(next_node[1]) < 600) and (round(next_node[0]) > 0)):
        if(canvas[round(next_node[0]),round(next_node[1]),0]==0) and  (canvas[round(next_node[0]),round(next_node[1]),2]==0):
            return next_node
    else:
        return None
       
#Function to create a movement in the forward direction
def forward( node,canvas,visited,L):
    angle=node[2]-0
    if angle>360:
        angle=360-angle
    if angle<0:
        angle=360+angle
    next_node=[node[0]+L*np.cos(np.radians(node[2])),node[1]+L*np.sin(np.radians(angle)),angle]
    #print(canvas[current_node[0]][current_node[1]] )
    next_node[0]=round_nearest(next_node[0])
    next_node[1]=round_nearest(next_node[1])
    if(angle!=0):
        rounded_angle=360%angle
    else:
        rounded_angle=angle
    
    #node should not be in obstacle space
    if((round(next_node[1]) > 0) and (round(next_node[0]) < 250) and (round(next_node[1]) < 600) and (round(next_node[0]) > 0)) :
        if (canvas[round(next_node[0]),round(next_node[1]),0]==0) and  (canvas[round(next_node[0]),round(next_node[1]),2]==0):
            return next_node
    else:
        return None
       
       
#Function to create a movement of 60 degrees in the backward direction        
def backward_60( node,canvas,visited,L):
    angle=node[2]+60
    if angle>360:
        angle=360-angle
    if angle<0:
        angle=360+angle
    next_node=[node[0]+L*np.cos(np.radians(node[2])),node[1]+L*np.sin(np.radians(angle)),angle]
    #print(canvas[current_node[0]][current_node[1]] )
    next_node[0]=round_nearest(next_node[0])
    next_node[1]=round_nearest(next_node[1])
    if(angle!=0):
        rounded_angle=360%angle
    else:
        rounded_angle=0
    
    #node should not be in obstacle space
    if((round(next_node[1]) > 0) and (round(next_node[0]) < 250) and (round(next_node[1]) < 600) and (round(next_node[0]) > 0)) :
        if (canvas[round(next_node[0]),round(next_node[1]),0]==0) and  (canvas[round(next_node[0]),round(next_node[1]),2]==0):
            return next_node
    else:
        return None
       
       
       
       
#Function to create a movement of 30 degrees in the backward direction       
def backward_30( node,canvas,visited,L):
    angle=node[2]-60
    if angle>360:
        angle=360-angle
    if angle<0:
        angle=360+angle
    next_node=[node[0]+L*np.cos(np.radians(node[2])),node[1]+L*np.sin(np.radians(angle)),angle]
    #print(canvas[current_node[0]][current_node[1]] )
    next_node[0]=round_nearest(next_node[0])
    next_node[1]=round_nearest(next_node[1])
    if(angle!=0):
        rounded_angle=360%angle
    else:
        rounded_angle=angle
    
    #node should not be in obstacle space
    if((round(next_node[1]) > 0) and (round(next_node[0]) < 250) and (round(next_node[1]) < 600) and (round(next_node[0]) > 0)) :
        if (canvas[round(next_node[0]),round(next_node[1]),0]==0) and  (canvas[round(next_node[0]),round(next_node[1]),2]==0):
            return next_node
    else:
        return None
       
       
       
       

    


# In[3]:


#Function to implement A star algorithm
def A_star(start_node, goal_node,canvas,L):
    S={}        #tuple(present):parent
    PQ=[]
    temp=0
    hq.heapify(PQ)
    c2g_initial_node=Euclidean_distance(start_node,goal_node)
    cost=c2g_initial_node
    
     #PQ= priority Queue. Elements: cost,c2c,c2g,parent,present
    hq.heappush(PQ,[cost,0,c2g_initial_node,start_node,start_node])  
    #print(f"Initial PQ:{PQ}")
    #visited=[]
    #since we are multiplying nearest 0.5 number with 2 and scaling down 0-360 angle to multiples of 30 
    visited=np.zeros((500,1200,12))     
    while(len(PQ)!=0):
        node=hq.heappop(PQ)
        S[tuple(node[4])]=node[3]
        cost=node[0]
        c2c=node[1]
        c2g=node[2]
        #print(node,S,cost,c2c,c2g)
        #val=goal_thresh(node[4],goal_node)
        if(goal_thresh(node[4],goal_node)):
            print(node[4])
            print("\n------\nGOAL REACHED\n--------\n")
            S[node[4][0],node[4][1]]=node[3]
            Backtrack(S,start_node,node[4],canvas)
            temp=1
            break;
        
        #actions
        
        next_node=[]
        next_node.append(forward_60(node[4],canvas,visited,L))
        #print(next_node)                 
        next_node.append(forward_30(node[4],canvas,visited, L))
        next_node.append(forward(node[4],canvas,visited, L))
        next_node.append(backward_30(node[4],canvas,visited, L))
        next_node.append(backward_60(node[4],canvas,visited, L))
        
        
        
        
        

        for count in range(5) :
            if(next_node[count]):
                print(f"NExt node:{next_node[count]}")
                if(tuple(next_node[count]) not in S):
                    current_cost=cost+L+Euclidean_distance(next_node[count],goal_node)
                    if( check_duplicate(next_node[count], visited)):
                    
                        for i in range(len(PQ)):
                            if(PQ[i][4]==[next_node[count][0],next_node[count][1],next_node[count][2]]):
                                if(PQ[i][0]>current_cost):
                                    PQ[i][3]=node[4]
                                    PQ[i][1]=cost+L
                                    PQ[i][0]=current_cost
                                    hq.heapify(PQ)
                                    #print("In for loop:"PQ)
                                break
            
                    else:
                    
                        PQ.append([current_cost,cost+L,Euclidean_distance(next_node[count],goal_node),node[4],next_node[count]])
                        hq.heapify(PQ)
                        #print(PQ)
        
    if temp==0 :
        print("Goal cannot be reached")


# In[ ]:


#This is main function of the program
if __name__ == '__main__': 
   canvas = np.zeros((250, 600, 3)) 
   rad=input("Enter Robot radius")
   cle=input("Enter Robot clearance") 
   start=[] 
   goal=[]
   
   
   bloat=int(float(rad)+float(cle))

   for j in range(100,151):
       for i,k in zip(range(101),range(150,250)):
           canvas[i][j]=[255,0,0]
           canvas[k][j]=[255,0,0]


   for j in range(600):
       for i,k in zip(range(bloat+1),range(250-bloat,250)):
           canvas[i][j]=[0,0,255]
           canvas[k][j]=[0,0,255]

   for i in range(250):
       for j,k in zip(range(bloat+1),range(600-bloat,600)):
           canvas[i][j]=[0,0,255]
           canvas[i][k]=[0,0,255]





   for i in range(canvas.shape[1]):
       for j in range(canvas.shape[0]):
       #First Rectangle Obstacle
           Rect_1 = (i>=100 and i<=150) and (j>=0 and j<=100)

       #second Rectangle Obstacle
           Rect_2 = (i>=100 and i<=150) and (j>=150 and j<=250)

           r=math.sqrt(3)
           #Hexagon Obstacle
           Edge_1 = i<=1.732*j+213.397
           Edge_2 = i<=364.952
           Edge_3 = i<=-1.732*j+646.41
           Edge_4 = i>=1.732*j-46.410
           Edge_5 = i>=235.048
           Edge_6 = i>=-1.732*j+386.602

       #Triangle Obstacle
           Side_1 = (i-460>=0)
           Side_2 = (j>2*i-895)
           Side_3 = (j<-2*i+1145)




       #bloating with clearance + radius (hexagon)
           x=math.ceil(2*bloat/r)+1.5
       #vertices= find_hexagon_vertices(125,300,75+x)


           edge1=i<=1.732*j+213.397+x

           edge2=i<=364.952+bloat 

           edge3= i<=-1.732*j+646.41+x

           edge4=i>=1.732*j-46.410-x

           edge5=i>=235.048-bloat

           edge6=i>=-1.732*j+386.602-x

       #obstacle bloating for triangle
           Side1 = (i-460+bloat>=0)
           Side2=  (j>2*i-895-x)
           Side3 = (j<-2*i+1145+x)


           if((edge1 and edge2 and edge3 and edge4 and edge5 and edge6) or (Side1 and Side2 and Side3)):
               canvas[j][i]=[0,0,255]



           if(Rect_1 or Rect_2 or (Edge_1 and Edge_2 and Edge_3 and Edge_4 and Edge_5 and Edge_6) or (Side_1 and Side_2 and Side_3)):
               canvas[j][i] = [255,0,0]
#bloating rectanges
   for j,l in zip(range(100-bloat,100),range(150,150+bloat+1)):
       for i,k in zip(range(101),range(150,250)): 
           canvas[i,j]=[0,0,255] 
           canvas[i,l]=[0,0,255] 
           canvas[k,j]=[0,0,255] 
           canvas[k,l]=[0,0,255]

   for j in range(100-bloat,150+bloat):
       for i,k in zip(range(100,100+bloat+1),range(150-bloat,150)):
           canvas[i,j]=[0,0,255]
           canvas[k,j]=[0,0,255]
           
           
           
           
           
   while(1):
       x1=input("Enter start point X coordinate: ")
       y1=input("Enter start point Y coordinate: ")
       
       theta1=input("Enter start point orientation (Enter a multiple of 30): ")
       
       x2=input("Enter goal point X coordinate: ") 
       y2=input("Enter goal point Y coordinate: ")
       theta2=input("Enter goal point orientation: (Enter a multiple of 30): ")
       L=int(input("Enter step size")) 
       if(canvas[250-int(y1),int(x1),0]!=0 or canvas[250-int(y1),int(x1),2]!=0):
           print("Start Node in obstacle space, try again")
       elif(canvas[250-int(y2),int(x2),0]!=0 or canvas[250-int(y2),int(x2),2]!=0):
           print("Goal Node in obstacle space, try again")
       elif(int(theta1)%30!=0 or int(theta2)%30!=0):
           print("Enter angles in multiples of 30")
       elif(int(L)<0 or int(L)>10):
           print("Enter step size in the range 0 to 10 units")
       elif([x1,y1,theta1]==[x2,y2,theta2]):
           print("Goal reached..")
           
       else:
           break
           
   start.append(250-int(y1)) 
   start.append(int(x1)) 
   start.append(int(theta1))
   goal.append(250-int(y2))
   goal.append(int(x2)) 
   goal.append(int(theta2))


   A_star(start,goal,canvas,L)
   cv2.imshow("canvas",canvas)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

