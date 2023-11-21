#3.2.1 program
from scipy.integrate import RK45, solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import shutil




def plot_points(patch_points, msg):
    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    for i,p in enumerate(patch_points):
        plt.plot(p[0], p[1],  'ro', markersize=1)
        if i == len(patch_points)-1:
            break
        pTstate = propogate(p,dt, thrust=True)
        pT = np.copy(p)
        pT[0:9] = pTstate[-1]
        pf = propogate(pT, dt, thrust=False)
        #ax.plot3D(pp[:,0], pp[:,1], pp[:,2], label=f'patch point {i}')
        plt.plot(pTstate[:,0], pTstate[:,1], label=f'patch point {i}', color='r')
        plt.plot(pf[:,0], pf[:,1], label=f'patch point {i}', color='b')
        #ax.plot3D(p[0], p[1], p[2], 'ro', markersize=1)
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


def get_acceleration(state, thrust=True):

    rho = state[:3]
    v = state[3:6]
    m_moon = 7.34767e22 #kg
    m_earth = 5.97219e24 #kg
    mu= m_moon/(m_earth + m_moon)
#    lstar = #distance between two primaries r12
    d= np.sqrt((rho[0]+mu)**2 + rho[1]**2 + rho[2]**2)#r13/lstar
    r= np.sqrt((rho[0]-1+mu)**2 + rho[1]**2 + rho[2]**2)#r23/lstar
    a = np.reshape( np.array([
        2*v[1] + rho[0] - (1-mu)*(rho[0]+mu)/d**3 - mu*(rho[0]-1+mu)/r**3,
        -2*v[0] + rho[1] - (1-mu)*rho[1]/d**3 - mu*rho[1]/r**3,
        -rho[2]*(1-mu)/d**3 - mu*rho[2]/r**3
        ]) , (3,1))
    if thrust:
        a+= np.reshape(state[6:9], (3,1))
    return a

def dynamics(t, state):
    '''
    circular restricted 3 body problem with non impulsive thrust
    state vector: x, y, z, vx, vy, vz, Y, a, b, T0, Tf, t0, tf
    given in EQ 2.23
    '''
    rho = state[:3]
    v = state[3:6]
    T = state[6:]
    #thrust values

    
    m_moon = 7.34767e22 #kg
    m_earth = 5.97219e24 #kg
    mu= m_moon/(m_earth + m_moon)
#    lstar = #distance between two primaries r12
    d= np.sqrt((rho[0]+mu)**2 + rho[1]**2 + rho[2]**2)#r13/lstar
    r= np.sqrt((rho[0]-1+mu)**2 + rho[1]**2 + rho[2]**2)#r23/lstar
    a = np.array([
        2*v[1] + rho[0] - (1-mu)*(rho[0]+mu)/d**3 - mu*(rho[0]-1+mu)/r**3,
        -2*v[0] + rho[1] - (1-mu)*rho[1]/d**3 - mu*rho[1]/r**3,
        -rho[2]*(1-mu)/d**3 - mu*rho[2]/r**3
        ])
    a +=T #add thrust, T=[0,0,0] for non thrust arcs
    return np.array([v[0],
                     v[1],
                     v[2],
                     a[0],
                     a[1],
                     a[2],
                     0,
                     0,
                     0
        ])

def get_thrust(Y,A,B):
    '''
    Get thrust from azimuth, elevation and range 
    '''
    T = np.array([
                    np.sin(A)*np.cos(B),
                    np.cos(A)*np.cos(B),
                    np.sin(B)
                    ])
    T*=Y
    return T

def get_aby(T):
    Y = np.linalg.norm(T)
    t = T/Y
    A = np.arctan(t[0] / t[1])
    B = np.arctan(t[2] / t[1])
    return A,B,Y


def propogate(state,dt, thrust=False):
    '''
    propogate dynamics given initial state at inital time
    to a final time
    '''
    #thrust times
#    TF = state[-3]
#    T0 = state[-4]
#    Y  = state[6]
#    A  = state[7]
#    B  = state[8]
#    
#    ZeroThrust = np.zeros(3)
#    ThrustVector = np.array([
#                    np.sin(A)*np.cose(B),
#                    np.cos(A)*np.cos(B),
#                    np.sin(B)
#                    ])
#    T*=Y
    if not thrust:
        t = state[-3] #tT
        tf = state[-1] #ti+1
        print(f'no thrust PROPOGATE tf {tf} t {t} dt {dt}')
        states = np.zeros((int((tf-t)//dt),9))
        state = state[:-3] #exclude times from state vector
        states[0] = np.copy(state) 
        states[0,6:] = np.zeros(3)
    else:
        t = state[-2] #t0
        tf = state[-3] #tT
        print(f'thrust PROPOGATE tf {tf} t {t} dt {dt}')
        states = np.zeros((int((tf-t)//dt),9))
        state = state[:-3] #exclude times from state vector
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

#B01
def dp1dv0(state_guess, h ):
    t0 = state_guess[-2]
    tf = state_guess[-1]
    #state_guess = state_guess[:6]
    dp1dvx = ((propogate(state_guess + np.array([0,0,0,h,0,0,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,h,0,0,0,0,0,0,0,0]),dt))[-1] / (2*h))[:3]
    dp1dvy = ((propogate(state_guess + np.array([0,0,0,0,h,0,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,0,h,0,0,0,0,0,0,0]),dt))[-1] / (2*h))[:3]
    dp1dvz = ((propogate(state_guess + np.array([0,0,0,0,0,h,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,0,0,h,0,0,0,0,0,0]),dt))[-1] / (2*h))[:3]
    #dp1/dv0 
    return np.vstack((
        dp1dvx,
        dp1dvy,
        dp1dvz
        )).T
     
#C01
def dv1dp0(state_guess, h ):
    t0 = state_guess[-2]
    tf = state_guess[-1]
    #state_guess = state_guess[:6]
    dv1dx = ((propogate(state_guess + np.array([h,0,0,0,0,0,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([h,0,0,0,0,0,0,0,0,0,0,0]),dt))[-1] / (2*h))[3:6]
    dv1dy = ((propogate(state_guess + np.array([0,h,0,0,0,0,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,h,0,0,0,0,0,0,0,0,0,0]),dt))[-1] / (2*h))[3:6]
    dv1dz = ((propogate(state_guess + np.array([0,0,h,0,0,0,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,h,0,0,0,0,0,0,0,0,0]),dt))[-1] / (2*h))[3:6]
    #dp1/dv0 
    return np.vstack((
        dv1dx,
        dv1dy,
        dv1dz
        )).T

#D01
def dv1dv0(state_guess, h ):
    t0 = state_guess[-2]
    tf = state_guess[-1]
    #state_guess = state_guess[:6]
    dv1dvx = ((propogate(state_guess + np.array([0,0,0,h,0,0,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,h,0,0,0,0,0,0,0,0]),dt))[-1] / (2*h))[3:6]
    dv1dvy = ((propogate(state_guess + np.array([0,0,0,0,h,0,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,0,h,0,0,0,0,0,0,0]),dt))[-1] / (2*h))[3:6]
    dv1dvz = ((propogate(state_guess + np.array([0,0,0,0,0,h,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,0,0,h,0,0,0,0,0,0]),dt))[-1] / (2*h))[3:6]
    return np.vstack((
        dv1dvx,
        dv1dvy,
        dv1dvz
        )).T

#A01
def dp1dp0(state_guess, h ):
    t0 = state_guess[-2]
    tf = state_guess[-1]
    #state_guess = state_guess[:6]
    dp1dx = ((propogate(state_guess + np.array([h,0,0,0,0,0,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([h,0,0,0,0,0,0,0,0,0,0,0]),dt))[-1] / (2*h))[:3]
    dp1dy = ((propogate(state_guess + np.array([0,h,0,0,0,0,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,h,0,0,0,0,0,0,0,0,0,0]),dt))[-1] / (2*h))[:3]
    dp1dz = ((propogate(state_guess + np.array([0,0,h,0,0,0,0,0,0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,h,0,0,0,0,0,0,0,0,0]),dt))[-1] / (2*h))[:3]
    return np.vstack((
        dp1dx,
        dp1dy,
        dp1dz
        )).T


def dp1dY0(state_guess,h):
    #just the burn
    t0 = state_guess[-2]
    tf = state_guess[-3]
    #state_guess = state_guess[:6]
    A,B,Y = get_aby(state_guess[6:9])
    Yp = Y+h 
    Ym = Y-h 
    Tp = get_thrust(Yp,A,B)
    Tm = get_thrust(Ym,A,B)
    statep = np.copy(state_guess)
    statem = np.copy(state_guess)
    statep[6:9] = Tp 
    statem[6:9] = Tm
    #state_guess = state_guess[:6]
    dp1dY0 = ((propogate(statep,dt)[-1] - propogate(statem,dt))[-1] / (2*h))[:3]
    return np.reshape(dp1dY0, (3,1))

def dv1dY0(state_guess,h):
    t0 = state_guess[-2]
    tf = state_guess[-3]
    #state_guess = state_guess[:6]
    A,B,Y = get_aby(state_guess[6:9])
    Yp = Y+h 
    Ym = Y-h 
    Tp = get_thrust(Yp,A,B)
    Tm = get_thrust(Ym,A,B)
    statep = np.copy(state_guess)
    statem = np.copy(state_guess)
    statep[6:9] = Tp 
    statem[6:9] = Tm
    #state_guess = state_guess[:6]
    dv1dY0 = ((propogate(statep,dt)[-1] - propogate(statem,dt))[-1] / (2*h))[3:6]
    return np.reshape(dv1dY0, (3,1))


def dp1dA0(state_guess,h):
    t0 = state_guess[-2]
    tf = state_guess[-3]
    #bro this is stupid, find a better way...
    A,B,Y = get_aby(state_guess[6:9])
    Ap = A+h 
    Am = A-h 
    Tp = get_thrust(Y,Ap,B)
    Tm = get_thrust(Y,Am,B)
    statep = np.copy(state_guess)
    statem = np.copy(state_guess)
    statep[6:9] = Tp 
    statem[6:9] = Tm
    #state_guess = state_guess[:6]
    dp1dA0 = ((propogate(statep,dt)[-1] - propogate(statem,dt))[-1] / (2*h))[:3]
    return np.reshape(dp1dA0, (3,1))


def dv1dA0(state_guess,h):
    t0 = state_guess[-2]
    tf = state_guess[-3]
    #bro this is stupid, find a better way...
    A,B,Y = get_aby(state_guess[6:9])
    Ap = A+h 
    Am = A-h 
    Tp = get_thrust(Y,Ap,B)
    Tm = get_thrust(Y,Am,B)
    statep = np.copy(state_guess)
    statem = np.copy(state_guess)
    statep[6:9] = Tp 
    statem[6:9] = Tm
    #state_guess = state_guess[:6]
    dv1dA0 = ((propogate(statep,dt)[-1] - propogate(statem,dt))[-1] / (2*h))[3:6]
    return np.reshape(dv1dA0, (3,1))


def dp1dB0(state_guess,h):
    t0 = state_guess[-2]
    tf = state_guess[-3]
    #bro this is stupid, find a better way...
    A,B,Y = get_aby(state_guess[6:9])
    Bp = B+h 
    Bm = B-h 
    Tp = get_thrust(Y,A,Bp)
    Tm = get_thrust(Y,A,Bm)
    statep = np.copy(state_guess)
    statem = np.copy(state_guess)
    statep[6:9] = Tp 
    statem[6:9] = Tm
    #state_guess = state_guess[:6]
    dp1dA0 = ((propogate(statep,dt)[-1] - propogate(statem,dt))[-1] / (2*h))[:3]
    return np.reshape(dp1dA0, (3,1))

def dv1dB0(state_guess,h):
    t0 = state_guess[-2]
    tf = state_guess[-3]
    #bro this is stupid, find a better way...
    A,B,Y = get_aby(state_guess[6:9])
    Bp = B+h 
    Bm = B-h 
    Tp = get_thrust(Y,A,Bp)
    Tm = get_thrust(Y,A,Bm)
    statep = np.copy(state_guess)
    statem = np.copy(state_guess)
    statep[6:9] = Tp 
    statem[6:9] = Tm
    #state_guess = state_guess[:6]
    dv1dA0 = ((propogate(statep,dt)[-1] - propogate(statem,dt))[-1] / (2*h))[3:6]
    return np.reshape(dv1dA0, (3,1))


def level_1(initial_state, state_desired, epsilon):
    '''
    initial_state 8x1 (pos, vel, t0,tf).T
    '''
    pfp = state_desired[:3]
    state_guess = np.copy(initial_state)
    #thrust arc
    stateTm = propogate(state_guess, dt, thrust=True)[-1]
    ptm = stateTm[:3]
    Tm = np.copy(state_guess)
    Tm[:9] = stateTm
    #coast arc
    pfm = propogate(Tm, dt, thrust=False)[-1,:3]
   

    

    
    #dp1/dv0
    #for first central finite difference
    h= .00001
    h_v = h* tstar / lstar
    h_p = h/ lstar 
    h_f = h * lstar * mstar / (tstar**2)


    #design vector [Y,A,B,v,t]
    A,B,Y = get_aby(initial_state[6:9])
    X = np.reshape(np.array([Y,A,B, initial_state[3], initial_state[4], initial_state[5], initial_state[-3] ]), (7,1))
    i=0
    
    while residual > epsilon:
        print('residual', residual)
        #update propogated position
        stateTm = propogate(state_guess, dt, thrust=True)[-1]
        pTm = stateTm[:3]
        Tm = np.copy(state_guess)
        Tm[:9] = stateTm 
        pfm = propogate(Tm, dt, thrust=False)[-1,:3]
        
        F = np.reshape(pfm - pfp, (3,1))
        print('stateTm shape', stateTm.shape)
        apTm = get_acceleration(stateTm, thrust=True) #ingoing with thrust
        apTp = get_acceleration(stateTm, thrust=False) #outgoing without thrust
        
        Abpf = dp1dp0(Tm,h_p) #just the coast
        Bbpf = dp1dv0(Tm,h_v) #just the coast
        Bpf  = dp1dv0(state_guess, h_v)
        Fpf  = dp1dY0(state_guess,h_f)#just the burn
        Gpf  = dp1dA0(state_guess,h) #just burn, i dont think h needs to be nondimensionalized for radians
        Hpf  = dp1dB0(state_guess,h) #just the burn i dont think we need to nondimensionalize angles
        Dpf  = dv1dv0(state_guess, h_v)
        Jpf  = dv1dY0(state_guess,h_f)#just the burn
        Kpf  = dv1dA0(state_guess,h)  #just the burn
        Lpf  = dv1dB0(state_guess,h) #just the burn
        
        print('shapes', (Abpf@Fpf + Bbpf@Jpf).shape,
                        (Abpf@Gpf + Bbpf@Kpf).shape,
                        (Abpf@Hpf + Bbpf@Lpf).shape,
                        (Abpf@Bpf  +Bbpf@Dpf).shape,
                        (Bbpf@ np.reshape(apTm - apTp, (3,1))).shape ) 

        DF = np.hstack((Abpf@Fpf + Bbpf@Jpf,
                        Abpf@Gpf + Bbpf@Kpf,
                        Abpf@Hpf + Bbpf@Lpf,
                        Abpf@Bpf  +Bbpf@Dpf,
                        Bbpf@ np.reshape(apTm - apTp, (3,1)))) 
        
        DX = -DF.T@np.linalg.inv(DF@DF.T)@F
        print('DX SHAPE', DX.shape)
        print('X SHAPE', X.shape)
        X+=DX
        #update state_guess with X... bro this is lame
        state_guess[3:6] = np.reshape(X[3:6], 3)
        state_guess[0:3] = np.reshape(X[:3], 3)
        state_guess[-3]  = X[6]
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
        #Cop = dv1dp0(state0, h_p)
        #Dop = dv1dv0(state0, h_v)
        
        Apo = np.linalg.inv(Aop)
        Bpo = np.linalg.inv(Bop)
        #Dpo = np.linalg.inv(Dop)


        Apf = dp1dp0(statep, h_p)
        Bpf = dp1dv0(statep, h_v)
        #Cpf = dv1dp0(statep, h_p)
        #Dpf = dv1dv0(statep, h_v)

        Afp = np.linalg.inv(Apf)
        Bfp = np.linalg.inv(Bpf)
        #Dfp = np.linalg.inv(Dpf)

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
    n=len(patch_points) -2
    residual_vector = np.empty(n)
    for i in np.arange(n):
        state = patch_points[i+1]
        p1 = propogate(state,dt)[-1]
        residual_vector[i] = np.linalg.norm(p1[3:6] - patch_points[i+2][ 3:6])
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
    dt =100.0/tstar#.5 

    #tolerance for answer
    epsilon = 3.844e-9 # km





    day = 1 * 24* 60 * 60 / tstar
    day*=1.5
    T = day/5
    Thrust = 1
    patch1 = [ .55 , -.225, 0,  .75 ,  .85 , 0,Thrust,0,0, T,     0,   day]
    patch2 = [ .75 ,  .005, 0,  .751,  .5  , 0,Thrust,0,0, T+day,   day, 2*day]
    patch3 = [ .975,  .085, 0, 1.0  , -.01 , 0,Thrust,0,0, T+2*day, 2*day, 3*day]
    patch4 = [1.2  , -.105, 0,  .1  , -.5  , 0,Thrust,0,0, T+3*day, 3*day, 4*day]


    #patch_points = [patch1, patch2, patch3]
    patch_points = [patch1, patch2, patch3,  patch4]
   
    residual = 99999999
    iterations=0
    while residual > epsilon:
        residual = compute_residual(patch_points)
        #residual_v = compute_residual_v(patch_points)
        residuals.append(residual)
        #residual_vs.append(residual_v)
        #print('global position residual, velocity residual:', residual, residual_v)
        plot_points(patch_points, f'Start of level 1:{residual}')    
    #first achieve position continuity
        for i in range(len(patch_points)-1):
            print(f'level 1 patch point {i+1}')
            patch_points[i] = level_1(patch_points[i], patch_points[i+1], epsilon)
        #plot_points(patch_points, f'Start of Level 2:{residual}')
        print('LEVEL 2!!!!!!!!!!!!!')    
    #Then achieve velocity continuity
        sys.exit(0)
        level_2(patch_points, epsilon)
        print('Global tlt iteration', iterations)
        iterations+=1
    
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
