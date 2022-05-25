# Libs
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

####################### TASK 1 ##############################
df = pd.read_csv('./data_opt.csv',header=None) # Load dataset
dataset = df.values # convert to ndarray

# function to compute D
def calc_distance_matrix(points):
    """
    calc_distance_matrix(ndarray)
    
    Function that calculates the Euclidean distance between each pair of a multidimensional dataset of points.
    """
    D = np.ndarray(shape=(len(points),len(points)), dtype=float) # initialize D matrix
    i = 1
    j = 1
    for point in points:
        p1 = point
        for point in points:
            p2 = point
            norm = (np.linalg.norm(p1-p2))
            D[i-1,j-1] = norm
            j = j + 1 if j < len(D) else 1
        i += 1
    return D

# calculate D for task 1
D = calc_distance_matrix(dataset)

# validate result
print('The expected result for D23 is: 5.8749')
print('The expected result for D45 is: 24.3769')
print('Calculated D23: ',D[1][2],'\nCalculated D45: ',D[3][4]) # in Python D45 is D34, e.g.
# answer questions
print('Max distance in D: ', np.amax(D), '\nMax distance location: ',np.where(D == np.amax(D)))

####################### TASK 2 ################################################################
### Functions

#  f_nm(y)

def f_nm(N,k,D,y,n,m):
    return np.linalg.norm( y[(m-1)*k:(m)*k] - y[(n-1)*k:(n)*k]  ) - D[m-1][n-1]

# f(y)

def f(N,k,D,y): 
    return sum([sum([ f_nm(N,k,D,y,n+1,m+1)**2 for n in range(N) if n>m ]) for m in range(N)] )

# gradient of f_nm(y)

def gradient_f_nm(N,k,y,n,m):
    y_m = y[(m-1)*k:(m)*k]
    y_n = y[(n-1)*k:(n)*k] 
    norm_2 = 1/np.linalg.norm( y[(m-1)*k:(m)*k] - y[(n-1)*k:(n)*k]  )
    
    if n > m: # for our problem, n is always greater than m
        return np.concatenate((np.concatenate((np.concatenate((np.concatenate((np.zeros((m-1)*k), norm_2*(y_m-y_n)) ),  np.zeros((n-1-m)*k)   )),   norm_2*(y_n-y_m)     )),   np.zeros((N-n)*k  )))
    elif n < m:
        return np.concatenate((np.concatenate((np.concatenate((np.concatenate((np.zeros((n-1)*k), norm_2*(y_n-y_m)) ),  np.zeros((m-1-n)*k)   )),   norm_2*(y_m-y_n)     )),   np.zeros((N-m)*k  )))
    else:
        return np.zeros(N*k)

# gradient of f(y)

def gradient_f(N,k,D,y):
    r = np.zeros(N*k)
    for m in range(N):
        for n in [z for z in range(N) if z > m ]:
            aux = gradient_f_nm(N,k,y,n+1,m+1)
            aux2 = f_nm(N,k,D,y,n+1,m+1)
            r = r + (2*aux2*aux)
    return r

# Matrix A

def build_matrix_A(N,k,y,lamb): 
    A = []
    for m in range(N):
        for n in [z for z in range(N) if z > m ]:
            A += [ gradient_f_nm(N,k,y,n+1,m+1) ]            
    aux_1 = np.diag([np.sqrt(lamb) for i in range(N*k)])
    for i in range(N*k):
        A += [ aux_1[i] ]
    return A

# Vector b

def build_vector_b(N,k,D,y,lamb): 
    b = []
    for m in range(N):
        for n in [z for z in range(N) if z > m ]:
            b += [  np.dot(gradient_f_nm(N,k,y,n+1,m+1), y) - f_nm(N,k,D,y,n+1,m+1) ]            
    for i in np.dot( np.diag([np.sqrt(lamb) for i in range(N*k)]), y ):
        b += [ i ]
    return b

####################### TASK 3 #####################################################
# LM algorithm

def LM(N,k,current_lbd,epslon,current_y,D):
    ''' Performs the Levenberg-Marquardt algorithm for a dataset of points of dimension N.p where D is the distance matrix between all the points (see calculate_distance_matrix()). Given a threshold epslon (for solution acceptance), an initial lambda and an initialization vector of dimension n.k, this function returns the solution vector y, the recorded gradient norms, lambdas and cost function values through the algorithm. age e.g.: y,g_norms,lambdas,cfvs = LM(N,k,current_lbd,epslon,current_y,D) '''
    start = time.time()
    g = gradient_f(N,k,D,current_y) # compute gradient of f
    g_norm = np.linalg.norm(g) # compute the norm of the gradient
    valid_steps = 0
    g_norms = [g_norm] # to record data
    lambs = [current_lbd] # to record data
    f_curr = f(N,k,D,current_y) # compute f for the initial vector
    fs = [f_curr]
    while g_norm > epslon: # repeat until the goal epslon is accompplished
        A = build_matrix_A(N,k,current_y,current_lbd) # compute matrix A
        b = build_vector_b(N,k,D,current_y,current_lbd) # compute vector b
        y_cand = np.linalg.lstsq(A, b, rcond = None)[0] # solve the least squares problem
        f_cand = f(N,k,D,y_cand) # compute f for y_cand, to check if is a valid or a null step
        if  f_cand < f_curr: # valid step
            current_y = y_cand # assume the new y
            current_lbd = 0.7 * current_lbd # decrease lambda, increasing the weight of the linearization process
            g = gradient_f(N,k,D,current_y) # compute the gradient of the new y
            g_norm = np.linalg.norm(g)
            g_norms += [g_norm]
            lambs += [current_lbd] 
            f_curr = f(N,k,D,current_y)
            fs += [f_curr]
            valid_steps += 1
        else: # null step, y stays unchangeg, candidate y is rejected
            current_lbd = 2 * current_lbd # lambda is increased (stating that we want to stay closer to current x)
    print("Execution time: " + str(time.time() - start))
    return current_y,g_norms,lambs,fs

# test for k in {2,3}
for k in [2,3]:
    # initialization
    lbd_0 = 1
    epslon = k*pow(10,-2)
    filename = 'yinit'+str(k)+'.csv'
    df = pd.read_csv(filename,header=None)
    y0 = np.squeeze(df.values) # initialization vector
    N = int(len(y0)/k)
    
    # run LM and collect data to plot
    out,g_norms,lambs,fs = LM(N,k,lbd_0,epslon,y0,D)
    
    # vector for plotting
    plot_vector = [ out[i*k:(i+1)*k]  for i in range(N)]

    # plots
    if k == 2:
        # plot data points
        plt.figure(figsize=(16,9))
        plt.grid()
        plt.plot([i[0] for i in plot_vector],[i[1] for i in plot_vector],"bo")
        # plot norms
        plt.figure(figsize=(16,9))
        plt.grid()
        plt.yscale("log")
        plt.plot(range(len(g_norms[1:])),g_norms[1:])
        # plot cost function
        plt.figure(figsize=(16,9))
        plt.grid()
        plt.yscale("log")
        plt.plot(range(len(fs[1:])),fs[1:])
        plt.show()
    elif k == 3:
        # plot data points
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter([i[0] for i in plot_vector],[i[1] for i in plot_vector],[i[2] for i in plot_vector], marker="o")
        plt.show()
        
####################### TASK 4 ####################################################################################

# plot function
def plot_LM(plot_vector,fs):
    """ Plots the solution and the cost function of the LM algorithm algorithm for for MDS to two(2) dimensions
    """
    # plot data points
    plt.figure(figsize=(16,9))
    plt.grid()
    plt.plot([i[0] for i in plot_vector],[i[1] for i in plot_vector],"bo")
    # plot cost function
    plt.figure(figsize=(16,9))
    plt.grid()
    plt.yscale("log")
    plt.plot(range(len(fs[1:])),fs[1:])
    plt.show()

# testing for random initializations:

# Load dataset
df = pd.read_csv('./dataProj.csv',header=None)
data = df.values # initialization vector

# initializing
k = 2 # given
N = len(data)
rang = N*k
# matrix D is constant and depends only on the points in dataProj.csv
D4 = np.ndarray(shape=(N,N), dtype=float) # construct D matrix
D4 = calc_distance_matrix(data)
lbd0 = 1 # given
epslon = k * pow(10,-4) # given
for i in range(5):
    y0 = np.random.rand(rang,1).squeeze() # random initialization points
    top = np.random.randint(0,20000) # random scaling
    bottom = -1 * top
    y0 = bottom + (y0 * (top - bottom))
    # run LM
    out,g_norms,lambs,fs = LM(N,k,lbd_0,epslon,y0,D4)
    # build and plot
    plot_vector = [ out[i*k:(i+1)*k]  for i in range(N)]
    plot_LM(plot_vector,fs)
lowest_cost_value = np.amin(fs)
print('Lowest cost value',lowest_cost_value)

#this is to analyze a larger number of solutions to study the topology of the cost function,
#it will take a few minutes to run:

# analyzing outputs

#solutions_found = []
#for i in range(30):
#    y0 = np.random.rand(rang,1).squeeze() # random initialization points
#    top = np.random.randint(0,20000) # random scaling
#    bottom = -1 * top
#    y0 = bottom + (y0 * (top - bottom))

#    out,g_norms,lambs,fs = LM(N,k,lbd_0,epslon,y0,D4)
#    solutions_found += [ out ]

#lowest_cost_value = np.amin(fs)
#print('Lowest cost value',lowest_cost_value)

#for j in solutions_found:
#    plot_vector = [ j[i*k:(i+1)*k]  for i in range(N)]
#    plt.plot([i[0] for i in plot_vector],[i[1] for i in plot_vector] )
#plt.show()

#for i in solutions_found[:4]:
#    for j in solutions_found[:4]:
#        if i[0] != j[0]:
#            cord = [i*(1-z) + j*z for z in np.arange(0,1.1,0.1)]
#            vector_fs = []
#            for z in cord:
#                a = f(N,k,D4,z)
#                vector_fs += [a]
#            plt.plot(vector_fs)
#plt.show()