#3.2.1 program
from scipy.integrate import RK45, solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil




def plot_points(patch_points, msg):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i,p in enumerate(patch_points):
        pp = propogate(p,dt)
        ax.plot3D(pp[:,0], pp[:,1], pp[:,2], label=f'patch point {i}')
        ax.plot3D(p[0], p[1], p[2], 'ro', markersize=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(msg)
    plt.legend()
    if len(sys.argv) > 1:
        plt.show()
    else:
        plt.close('all')

def rk4_step( f, t, y, h ):
    '''
    Calculate one RK4 step
    '''
    k1 = f( t, y )
    k2 = f( t + 0.5 * h, y + 0.5 * k1 * h )
    k3 = f( t + 0.5 * h, y + 0.5 * k2 * h )
    k4 = f( t +       h, y +       k3 * h )

    return y + h / 6.0 * ( k1 + 2 * k2 + 2 * k3 + k4 )


def get_acceleration(state):

    rho = state[:3]
    v = state[3:6]
    m_moon = 7.34767e22 #kg
    m_earth = 5.97219e24 #kg
    mu= m_moon/(m_earth + m_moon)
#    lstar = #distance between two primaries r12
    d= np.sqrt((rho[0]+mu)**2 + rho[1]**2 + rho[2]**2)#r13/lstar
    r= np.sqrt((rho[0]-1+mu)**2 + rho[1]**2 + rho[2]**2)#r23/lstar
    return np.reshape( np.array([
        2*v[1] + rho[0] - (1-mu)*(rho[0]+mu)/d**3 - mu*(rho[0]-1+mu)/r**3,
        -2*v[0] + rho[1] - (1-mu)*rho[1]/d**3 - mu*rho[1]/r**3,
        -rho[2]*(1-mu)/d**3 - mu*rho[2]/r**3
        ]) , (3,1))


def dynamics(t, state):
    '''
    circular restricted 3 body problem 
    given in EQ 2.23
    '''
    rho = state[:3]
    v = state[3:]
    m_moon = 7.34767e22 #kg
    m_earth = 5.97219e24 #kg
    mu= m_moon/(m_earth + m_moon)
#    lstar = #distance between two primaries r12
    d= np.sqrt((rho[0]+mu)**2 + rho[1]**2 + rho[2]**2)#r13/lstar
    r= np.sqrt((rho[0]-1+mu)**2 + rho[1]**2 + rho[2]**2)#r23/lstar
    return np.array([v[0],
                     v[1],
                     v[2],
        2*v[1] + rho[0] - (1-mu)*(rho[0]+mu)/d**3 - mu*(rho[0]-1+mu)/r**3,
        -2*v[0] + rho[1] - (1-mu)*rho[1]/d**3 - mu*rho[1]/r**3,
        -rho[2]*(1-mu)/d**3 - mu*rho[2]/r**3
        ])

def stm(state):
    rho = state[:3]
    mu= m_moon/(m_earth + m_moon)
    d= np.sqrt((rho[0]+mu)**2 + rho[1]**2 + rho[2]**2)#r13/lstar
    r= np.sqrt((rho[0]-1+mu)**2 + rho[1]**2 + rho[2]**2)#r23/lstar
    #d=sqrt((x+mu)**2 + y**2 + z^^2)
    #r=sqrt((x-1+mu)**2 + y**2 + z**2)
    x = state[0]
    y=state[1]
    z=state[2]
    xd=state[3]
    yd=state[4]
    zd=state[5]
    #dx../dx = 
    #1 - d (x+mu-mux -mu**2)/d**3 + d (-mux+mu+mu**2)/r**3
    dxdx1 = 1 - ((-mu+1)*(y**2+z**2-2*(x+mu)**2)) / (y**2+z**2 + (x+mu)**2)**(5/2) - (mu*(y**2+z**2-2*(x+mu-1)**2)) / (y**2+z**2 + (x+mu-1)**2)**(5/2) 
    dxdx2 = (3*y*(-mu+1)*(x+mu)) / (y**2+z**2+(x+mu)**2)**(5/2) + (3*y*(mu**2 + x*mu-mu)) / (y**2+z**2+(x+mu-1)**2)**(5/2)
    dxdx3 = (3*z*(-mu+1)*(x+mu)) / (y**2+z**2+(x+mu)**2)**(5/2) + (3*z*(mu**2 +x*mu - mu)) / (y**2+z**2+(x+mu-1)**2)**(5/2)
    dxdx4 = 0
    dxdx5 = 2
    dxdx6 = 0

    dydx1 = (3*y*(-mu+1)*(mu+x)) / (y**2+z**2+(mu+x)**2)**(5/2) + (3*y*mu*(mu+x-1)) / ((x-1+mu)**2+y**2+z**2)**(5/2)
    dydx2 = 1 - ((1-mu)*(mu**2+2*mu*x+x**2+z**2-2*y**2)) / (y**2+z**2 + (mu+x)**2)**(5/2) - (mu*(mu**2+2*mu*x-2*mu+x**2+z**2+1-2*y**2-2*x)) / (y**2+z**2 + (mu+x-1)**2)**(5/2)
    dydx3 = (3*y*mu*z) / (y**2+z**2+(mu+x-1)**2)**(5/2) + (3*y*z*(1-mu)) / (y**2+z**2+(mu+x)**2)**(5/2)
    dydx4 = -2
    dydx5 = 0
    dydx6 = 0

    dzdx1 = (3*z*(1-mu)*(x+mu)) / ((x+mu)**2+y**2+z**2)**(5/2) + (3*mu*z*(x+mu-1)) / ((x-1+mu)**2+y**2+z**2)**(5/2)
    dzdx2 = (3*z*y*(1-mu)) / ((x+mu)**2+y**2+z**2)**(5/2) + (3*mu*z*y) / ((x-1+mu)**2+y**2+z**2)**(5/2)
    dzdx3 = -(mu**2+2*mu*x+x**2+y**2-2*z**2)*(1-mu) / ((mu+x)**2+y**2+z**2)**(5/2) - mu*(-2*z**2+mu**2-2*mu+x**2+2*mu*x-2*x+y**2+1) / ((mu+x-1)**2+y**2+z**2)**(5/2)
    dzdx4=0
    dzdx5=0
    dzdx6=0


    stm = np.array([
        [0,0,0,1,0,0],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1],
        [dxdx1,dxdx2,dxdx3,dxdx4,dxdx5,dxdx6],
        [dydx1,dydx2,dydx3,dydx4,dydx5,dydx6],
        [dzdx1,dzdx2,dzdx3,dzdx4,dzdx5,dzdx6]
        ])
    return stm


def propogate(state,dt):
    '''
    propogate dynamics given initial state at inital time
    to a final time
    '''
    t = state[-2]
    tf = state[-1]
    state = state[:-2]
    states = np.zeros((int((tf-t)//dt),6))
    if states.size ==0:
        sys.exit(1)
    states[0] = np.copy(state) 
    i=1
    t+=dt 
    while t<tf:
        if i>=(states.shape[0] ):
            break
        states[i] = rk4_step(dynamics,t,states[i-1],dt )
        t+=dt
        i+=1
    return states 


def dp1dv0(state_guess, h ):
    t0 = state_guess[-2]
    tf = state_guess[-1]
    #state_guess = state_guess[:6]
    dp1dvx = ((propogate(state_guess + np.array([0,0,0,h,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,h,0,0,0,0]),dt))[-1] / (2*h))[:3]
    dp1dvy = ((propogate(state_guess + np.array([0,0,0,0,h,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,0,h,0,0,0]),dt))[-1] / (2*h))[:3]
    dp1dvz = ((propogate(state_guess + np.array([0,0,0,0,0,h,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,0,0,h,0,0]),dt))[-1] / (2*h))[:3]
    #dp1/dv0 
    return np.vstack((
        dp1dvx,
        dp1dvy,
        dp1dvz
        )).T
     

def dv1dp0(state_guess, h ):
    t0 = state_guess[-2]
    tf = state_guess[-1]
    #state_guess = state_guess[:6]
    dv1dx = ((propogate(state_guess + np.array([h,0,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([h,0,0,0, 0,0,0,0]),dt))[-1] / (2*h))[3:6]
    dv1dy = ((propogate(state_guess + np.array([0,h,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,h,0,0, 0,0,0,0]),dt))[-1] / (2*h))[3:6]
    dv1dz = ((propogate(state_guess + np.array([0,0,h,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,h,0, 0,0,0,0]),dt))[-1] / (2*h))[3:6]
    #dp1/dv0 
    return np.vstack((
        dv1dx,
        dv1dy,
        dv1dz
        )).T


def dv1dv0(state_guess, h ):
    t0 = state_guess[-2]
    tf = state_guess[-1]
    #state_guess = state_guess[:6]
    dv1dvx = ((propogate(state_guess + np.array([0,0,0,h,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,h,0,0,0,0]),dt))[-1] / (2*h))[3:6]
    dv1dvy = ((propogate(state_guess + np.array([0,0,0,0,h,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,0,h,0,0,0]),dt))[-1] / (2*h))[3:6]
    dv1dvz = ((propogate(state_guess + np.array([0,0,0,0,0,h,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,0,0,h,0,0]),dt))[-1] / (2*h))[3:6]
    #dp1/dv0 
    return np.vstack((
        dv1dvx,
        dv1dvy,
        dv1dvz
        )).T


def dp1dp0(state_guess, h ):
    t0 = state_guess[-2]
    tf = state_guess[-1]
    #state_guess = state_guess[:6]
    dp1dx = ((propogate(state_guess + np.array([h,0,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([h,0,0,0,0,0,0,0]),dt))[-1] / (2*h))[:3]
    dp1dy = ((propogate(state_guess + np.array([0,h,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,h,0,0,0,0,0,0]),dt))[-1] / (2*h))[:3]
    dp1dz = ((propogate(state_guess + np.array([0,0,h,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,h,0,0,0,0,0]),dt))[-1] / (2*h))[:3]
    #dp1/dv0 
    return np.vstack((
        dp1dx,
        dp1dy,
        dp1dz
        )).T


def level_1(initial_state, state_desired, epsilon):
    '''
    initial_state 8x1 (pos, vel, t0,tf).T
    '''
    pd = state_desired[:3]
    state_guess = np.copy(initial_state)
    t0 = initial_state[-2]
    tf_guess = initial_state[-1]

    p1 = propogate(state_guess, dt)
    residual = np.linalg.norm(p1[-1][:3] - pd)* lstar 
    

    
    #dp1/dv0
    #for first central finite difference
    h= .00001
    h_v = h* tstar / lstar

    ##Why do I even need this?
    #h_t = h/tstar
    #dp1dt1 = ((propogate(state_guess, 0,dt,tf_guess + h_t )[-1] - propogate(state_guess,0,dt,tf_guess -h_t ))[-1] / (2*h_t))[:3]



    i=0
    while residual > epsilon:
        print('residual', residual)
        rho0 = state_guess[:3]
        v0 = state_guess[3:6]
        rho1 = p1[-1][:3]
        v1 = p1[-1][3:6]
        
        #B = 3x4 -> 3x3 = dp1dv0, 3x1 = dp1/dt1
        #DX = -4x3 (3x4 * 4x3)**-1 * 3x1
        #DX = -4x3 (3x3) * 3x1
        #DX = -4x3 * 3x1
        #DX = 4x1
        B = dp1dv0(state_guess,h_v)
        #B01_v1 = np.column_stack((B, v1))
        #DX = np.matmul(-B01_v1.T, np.linalg.inv(np.matmul(B01_v1, B01_v1.T))) #np.zeros(4) #??
        
        #DX = np.matmul(DX, rho1 - pd)
        DX = -np.linalg.inv(B) @ (rho1 -pd )
        state_guess[3:6] += DX[:3] 
        #tf_guess += DX[3] #???
        #print(f'DX: {DX}')
        p1 = propogate(state_guess, dt)
         
        
        residual = np.linalg.norm( p1[-1][:3]- pd) * lstar


        print(f'level 1 patch point iteration {i}')
        i+=1


    return state_guess








def level_2(patch_points, epsilon):
#only one iteration
    #constraint F= v1 - state_desired[3:-2]
    #minimum norm soluion (3.12) DX = -DF.T * (DF* DF.T).inverse() * F
    #least squares solution (3.17) DX = -(DF.T*DF).inverse() * DF.T*F
    #DF = [0, dv1/dp0, dv1/dt0, (dv1/dp1 - dV1/dp1), (dv1/dt1 - dV1/dt1), -dV1dpf, -dV1/dtf, 0] 
    #where V is the velocity at state_desired, v is the propogated velocity at t=tf
    #jacobain equations for calculated v1
    #dv1dp0 = B.inverse()
    #dv1dt0 = -B.inverse() * v0     #assuming V0=v0
    #dv1dp1 = -B.inverse() * A
    #dv1dt1 = B.inverse() * A * v1 + a1


    #jacobain equations for desired V1
    #dV1dp0 = B.inverse()
    #dV1dt0 = -B.inverse() * v0
    #dV1dp1 = -B.inverse() * A
    #dV1dt1 = B.inverse() * A * V1 + A1
    
    h= .00001
    h_v = h* tstar / lstar
    h_p = h /lstar
    DF = np.zeros(((len(patch_points) - 2)*3, 4*len(patch_points)))
    F = np.zeros((3 * (len(patch_points) - 2), 1))
    
    for i in range(len(patch_points) - 2):
        #reference states from paper
        state0 = patch_points[i]
        statep = patch_points[i+1]
        statef = patch_points[i+2]

        #not sure if these are right
        #from collin york's paper 
        Aop = dp1dp0(state0, h_p)
        Bop = dp1dv0(state0, h_v)
        Cop = dv1dp0(state0, h_p)
        Dop = dv1dv0(state0, h_v)
        
        Apo = np.linalg.inv(Aop)
        Bpo = np.linalg.inv(Bop)
        Dpo = np.linalg.inv(Dop)


        Apf = dp1dp0(statep, h_p)
        Bpf = dp1dv0(statep, h_v)
        Cpf = dv1dp0(statep, h_p)
        Dpf = dv1dv0(statep, h_v)

        Afp = np.linalg.inv(Apf)
        Bfp = np.linalg.inv(Bpf)
        Dfp = np.linalg.inv(Dpf)

        #useful propogated states
        statepm = propogate(state0, dt)[-1]
        statefm = propogate(statep, dt)[-1]

        #useful accelerations 
        afm = get_acceleration(statefm)
        apm = get_acceleration(statepm)
        app = get_acceleration(statep)
        a0 = get_acceleration(state0)
        #these are from spreens paper 
        """
        dv0pdr0 = -Bpo_invApo
        dv0pdt0 = a0 - Dop@Bpo@np.reshape(state0[3:6], (3,1))
        dv0pdrp = Bop 
        dv0pdtp = -Bop @ np.reshape(statep[3:6], (3,1))

        dvpmdr0 = Bpo 
        dvpmdt0 = -Bpo @ np.reshape(state0[3:6], (3,1))
        dvpmdrp = -Bpo @ Aop 
        dvpmdtp = apm - Dpo@Bop@np.reshape(statep[3:6], (3,1))

        dvppdrp = -Bpf @ Afp 
        dvppdtp = app - Dpf @ Bfp @ np.reshape(statep[3:6],(3,1))
        dvppdrf = -Bpf 
        dvppdtf = -Bpf @ np.reshape(statefm[3:6], (3,1))

        dvfmdrp = Bfp
        dvfmdtp = -Bfp @ np.reshape(statep[3:6], (3,1))
        dvfmdrf = -Bfp@Apf
        dvfmdtf = afm - Dfp @ Bpf @ np.reshape(statefm[3:6], (3,1))






        """
        dvpmdp0 = Bop
        dvpmdt0 = -Bop @ np.reshape(state0[3:6], (3,1))
        dvpmdpp = -Bop @ Apo 
        dvpmdtp = Bop @ Apo @ np.reshape(statepm[3:6], (3,1)) + apm  

        dvppdpf = Bfp
        dvppdtf = -Bfp @ np.reshape(statefm[3:6], (3,1))
        dvppdpp = -Bfp @ Apf 
        dvppdtp = Bfp @ Apf @ np.reshape(statep[3:6], (3,1)) + app 
        
        #DF[i*3:3+ i*3, 4*i: 4*i + 12] = np.hstack((dvpmdr0, dvpmdt0, dvpmdrp-dvppdrp, dvpmdtp - dvppdtp, -dvppdrf, -dvppdtf ))
        DF[i*3:3+ i*3, 4*i: 4*i + 12] = np.hstack((dvpmdp0, dvpmdt0, dvpmdpp-dvppdpp, dvpmdtp-dvppdtp, -dvppdpf, -dvppdtf ))

        """
        print('DF shape', DF.shape)
        print(DF)
        print()
        print(DF.T)
        print()
        print(DF @ DF.T)
        """
        #minimum norm soluion (3.12) DX = -DF.T * (DF* DF.T).inverse() * F
        vp_prop = statepm[3:6]
        vf_prop = statefm[3:6]
        
        F[i*3:i*3 + 3] = np.reshape(vp_prop - statep[3:6], (3,1)) #, vf_prop - statef[3:6]])

    #print(f'DF {DF.shape}:{DF}')
    
    DX = -DF.T @ np.linalg.inv( DF @ DF.T) @ F
    print('DX SHAPE!!!!', DX.shape)
    for i in range(len(patch_points)):
        patch_points[i][:3] +=   (np.reshape( DX[i*4:i*4 + 3], 3))
        patch_points[i][-1] +=   DX[i*4 + 3][0] #adjust end time of patch point
        print(DX.shape, DX[i*4+3].shape,  patch_points[i][-1].shape)
        if i != (len(patch_points) -1):
            patch_points[i+1][-2] = patch_points[i][-1]#adjust start time of next patch point
        print('change in position (km)',  DX[i*4:i*4 + 3]*lstar)
        print('change in time (s)', DX[i*4 + 3]* tstar )
        assert (patch_points[i][-1] >=0) and (patch_points[i][-2] >=0)
        assert (patch_points[i][-2] != patch_points[i][-1])



def compute_residual(patch_points):
    n=len(patch_points) -1
    residual_vector = np.empty(n)
    for i in np.arange(n):
        state = patch_points[i]
        p1 = propogate(state,dt)[-1]
        residual_vector[i] = np.linalg.norm(p1[:3] - patch_points[i+1][ :3])
    return np.linalg.norm(residual_vector)

def compute_residual_v(patch_points):
    n=len(patch_points) -1
    residual_vector = np.empty(n)
    for i in np.arange(n):
        state = patch_points[i]
        p1 = propogate(state,dt)[-1]
        residual_vector[i] = np.linalg.norm(p1[3:6] - patch_points[i+1][ 3:6])
    print('vel res vec')
    for i,v in enumerate(residual_vector):
        print(f'point {i}', v)
    return np.linalg.norm(residual_vector)
if __name__ == "__main__":
    residuals = []
    residual_vs = []

    lstar = 238900 * 1.609
    G = 6.674e-20
    m_moon = 7.34767e22 #kg
    m_earth = 5.97219e24 #kg
    mstar = (m_earth + m_moon)
    tstar = np.sqrt(lstar**3 /(G*mstar) )
    #initial conditions given in 3.31 and 3.32
    p0 = np.array([192200, 192200, 0]) / lstar #km
    v0=np.array([-.5123, .1025, 0]) * tstar/ lstar #km
    tf = 4.3425 * 24 * 60 * 60/tstar #4.3425 days in seconds
    pd = np.array([-153760, 0, 0])/lstar #desired position
    tf *=.1
    dt =10.0/tstar#.5 

    #tolerance for answer
    epsilon = 3.844e-3 # km
    #constraint vector F=p1-p1d = 0
    v = .2 * tstar / lstar 
    p = 100000 / lstar

    patch1 = np.array([p0[0], p0[1], p0[2], v0[0], v0[1], v0[2], 0, tf])
    #patch2 = np.array([pd[0], pd[1], pd[2], v0[0], v0[1], v0[2], tf, tf*2  ])
    patch3 = np.array([pd[0] , pd[1], pd[2], v0[0] , v0[1], v0[2], tf, tf*2  ])
    patch2 = patch1+ patch3
    patch3[-2] = 2*tf
    patch3[-1] = 3*tf

    patch2[-2] = tf
    patch2[-1] = 2*tf
    patch3[3:6] *=-1
    #patch4[-2] = 3*tf 
    #patch4[-1] = 4*tf




    day = 1 * 24* 60 * 60 / tstar
    patch1 = [ .55 , -.225, 0, .000001,  .000001 , 0,     0,   day]
    patch2 = [ .75 ,  .005, 0, .000001,  .0000005, 0,   day, 2*day]
    patch3 = [ .975,  .085, 0, .000001, -.000001 , 0, 2*day, 3*day]
    patch4 = [1.2  , -.105, 0, .000001,  .000001 , 0, 3*day, 4*day]


    #patch_points = [patch1, patch2, patch3]
    patch_points = [patch1, patch2, patch3,  patch4]
   
    residual = 99999999
    iterations=0
    while residual > epsilon:
        plot_points(patch_points, f'Start of level 1:{residual}')    
    #first achieve position continuity
        for i in range(len(patch_points)-1):
            print(f'level 1 patch point {i+1}')
            patch_points[i] = level_1(patch_points[i], patch_points[i+1], epsilon)
        #plot_points(patch_points, f'Start of Level 2:{residual}')
        print('LEVEL 2!!!!!!!!!!!!!')    
    #Then achieve velocity continuity
        level_2(patch_points, epsilon)
        print('Global tlt iteration', iterations)
        iterations+=1
        residual = compute_residual(patch_points)
        residual_v = compute_residual_v(patch_points)
        residuals.append(residual)
        residual_vs.append(residual_v)
        print('global position residual, velocity residual:', residual, residual_v)
    
    plt.plot(np.arange(len(residuals))[1:], residuals[1:])
    plt.title('(global) residuals vs iteration')
    plt.ylabel('position residuals (km)')
    plt.xlabel('global iteration number')
    plt.show()
    
    plt.plot(np.arange(len(residual_vs)), residual_vs)
    plt.title('(global) residuals vs iteration')
    plt.ylabel('velocity residuals (km/s)')
    plt.xlabel('global iteration number')
    plt.show()
