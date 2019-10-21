# Helper functions for the Probabilistic Robotic Class
#
import math
import numpy as np
import random

dirMat = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

def h2XY(h):
    global dirMat
    """
    Return the direction along the XY axes relative to
    the current heading (h).
    """
    row = math.floor((h + 1)/3.0)
    u_xy = dirMat[row, :]

    return u_xy 

def reachS(s, u, r):
    N_act = r.size() # Number of rotation to try
    S = np.zeros(N_act, 3)  # Output reachable set
    for i in range(N_act):
        rot = r[i]
        # Find in which direction I am going to move
        u_xy = u * h2XY(rot)
        S[i, :] = s + np.concatenate((u_xy, s[2] + rot))

    return S



def p_tran(s_new, s, a, pe):
    # Linear motion
    adv = a[0]
    # Rotative motion
    rot = a[1]

    p_vector = np.array([pe, pe, 1.0 - 2.0 * pe])
    
    # Consider the possibility to have an error
    # Pre-rotation of +1
    s_plus = s + np.array([0, 0, 1])
    # Pre-rotation of -1
    s_minus = s + np.array([0, 0, -1])
    # Matrix with all the possible initial states (considering errors)
    S = np.concatenate(([s], [s_plus], [s_minus]))

    # Distances from the final state
    delta_S = s_new - S


    p_tran = 0.0
    U = np.zeros((3,3))
    for (i in range(3)):
        u = adv * h2XY(S[i, 2]) # Linear motion
        U[i, 0:2] = u
        U[i, 2] = rot
        
        if (delta_S[i] == U[i]).all():
            p_tran = p_tran * p_vector[i]


    return p_tran


def trans(s, a, pe):
    """
    Transition function
    s' = f(s, a, pe)
    """
   
   ZZZ 
    p_next = np.zeros(4)
    for i in range(4):
        p_next[i] = p_tran(s + s + dirMat[i]) 

