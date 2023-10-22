#3.2.1 program
from scipy.integrate import RK45, solve_ivp
import numpy as np 
import matplotlib.pyplot as plt

def plot_trajectory(statvector):
    
    p0 = np.array([192200, 192200, 0])#km
    pd = np.array([-153760, 0, 0]) #desired position
    plt.plot(statvector[:,0], statvector[:,1])
    plt.plot([p0[0]], [p0[1]],marker='o',markersize=20, markerfacecolor='green')
    plt.plot([pd[0]], [pd[1]],marker='o',markersize=20, markerfacecolor='red')

    plt.show()

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
    return np.array([
        2*v[1] + rho[0] - (1-mu)*(rho[0]+mu)/d**3 - mu*(rho[0]-1+mu)/r**3,
        -2*v[0] + rho[1] - (1-mu)*rho[1]/d**3 - mu*rho[1]/r**3,
        -rho[2]*(1-mu)/d**3 - mu*rho[2]/r**3
        ])


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
#    lstar = 238900 * 1.609
#    G = 6.674e-20
#    m_moon = 7.34767e22 #kg
#    m_earth = 5.97219e24 #kg
#    mstar = (m_earth + m_moon)
#    tstar = np.sqrt(lstar**3 /(G*mstar) )
#    h = 5/tstar
#    x = solve_ivp(dynamics, (t,t+ h), state ).y
#    print('ivp_solution',x)
#    return x[:]
    #while t<tf:
#    solution= RK45(dynamics, t, state,tf ) 
#    while True:
#        solution.step()
#        if solution.status == 'finished':
#            break
#        t_values = solution.t
#        y_values=solution.y
#    y_values=np.array(y_values)
#    print('y value shape',y_values.shape)
#    return y_values 
    t = state[-2]
    tf = state[-1]
    state = state[:-2]
    print('propogate')
    print(tf, dt)
    states = np.zeros((int(tf//dt),6))
    states[0] = np.copy(state) 
    i=1
    t+=dt 
    while t<tf:
    #for i in range(int(t),int(tf),int(dt)):
        if i>=(int(tf//dt)):
            break
        states[i] = rk4_step(dynamics,t,states[i-1],dt )
        t+=dt
        i+=1
#    print(states)
    return states 


def dp1dv0(state_guess, h ):
    t0 = state_guess[-2]
    tf = state_guess[-1]
    #state_guess = state_guess[:6]
    dp1dvx = ((propogate(state_guess + np.array([0,0,0,h, 0,0, 0, 0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,h, 0,0,0,0]),dt))[-1] / (2*h))[:3]
    dp1dvy = ((propogate(state_guess + np.array([0,0,0,0, h,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,0, h,0,0,0]),dt))[-1] / (2*h))[:3]
    dp1dvz = ((propogate(state_guess + np.array([0,0,0,0, 0,h,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,0,0, 0,h,0,0]),dt))[-1] / (2*h))[:3]
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
    dv1dx = ((propogate(state_guess + np.array([h,0,0,0, 0,0, 0, 0]),dt)[-1] - propogate(state_guess - np.array([h,0,0,0, 0,0,0,0]),dt))[-1] / (2*h))[3:6]
    dv1dy = ((propogate(state_guess + np.array([0,h,0,0, 0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,h,0,0, 0,0,0,0]),dt))[-1] / (2*h))[3:6]
    dv1dz = ((propogate(state_guess + np.array([0,0,h,0, 0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,h,0, 0,0,0,0]),dt))[-1] / (2*h))[3:6]
    #dp1/dv0 
    return np.vstack((
        dv1dx,
        dv1dy,
        dv1dz
        )).T 




def dp1dp0(state_guess, h ):
    t0 = state_guess[-2]
    tf = state_guess[-1]
    #state_guess = state_guess[:6]
    dp1dx = ((propogate(state_guess + np.array([h,0,0,0, 0,0, 0, 0]),dt)[-1] - propogate(state_guess - np.array([h,0,0,0, 0,0,0,0]),dt))[-1] / (2*h))[:3]
    dp1dy = ((propogate(state_guess + np.array([0,h,0,0, 0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,h,0,0, 0,0,0,0]),dt))[-1] / (2*h))[:3]
    dp1dz = ((propogate(state_guess + np.array([0,0,h,0, 0,0,0,0]),dt)[-1] - propogate(state_guess - np.array([0,0,h,0, 0,0,0,0]),dt))[-1] / (2*h))[:3]
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
    plot_trajectory(p1*lstar)
    print(p1)
    residual = np.linalg.norm(p1[-1][:3] - pd)* lstar 
    

    
    #dp1/dv0
    #for first central finite difference
    h= .00001
    h_v = h* tstar / lstar

    ##Why do I even need this?
    #h_t = h/tstar
    #dp1dt1 = ((propogate(state_guess, 0,dt,tf_guess + h_t )[-1] - propogate(state_guess,0,dt,tf_guess -h_t ))[-1] / (2*h_t))[:3]



    print(f'p1:: {p1[-1][:3]}, pd: {pd}')
    i=0
    while residual > epsilon:
        print('residual', residual)
        rho0 = state_guess[:3]
        v0 = state_guess[3:6]
        print(f'v_guess  {v0}')
        print(f'tf guess difference {tf_guess-tf}') 
        rho1 = p1[-1][:3]
        v1 = p1[-1][3:6]
        
        #B = 3x4 -> 3x3 = dp1dv0, 3x1 = dp1/dt1
        #DX = -4x3 (3x4 * 4x3)**-1 * 3x1
        #DX = -4x3 (3x3) * 3x1
        #DX = -4x3 * 3x1
        #DX = 4x1
        print('TLT: ', state_guess)
        B = dp1dv0(state_guess,h)
        B01_v1 = np.column_stack((B, v1))
        DX = np.matmul(-B01_v1.T, np.linalg.inv(np.matmul(B01_v1, B01_v1.T))) #np.zeros(4) #??
        DX = np.matmul(DX, rho1 - pd)
        state_guess[3:6] += DX[:3] 
        tf_guess += DX[3] #???
        #print(f'DX: {DX}')
        p1 = propogate(state_guess, dt)
         
        #plot_trajectory(p1*lstar) 
        print(f'p1: {p1[-1][:3]}, pd: {pd}, dp: {rho1-pd}')
        
        residual = np.linalg.norm( p1[-1][:3]- pd) * lstar


        print(f'iteration {i}')
        i+=1


    return state_guess









def level_2(state0, statep,statef, epsilon):
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

    Bpo = dv1dp0(state0, h_p)
    Bpo_inv = np.linalg.inv(Bpo) 
    Apo = dp1dp0(state0, h_p)
    #Apo_inv = np.linalg.inv(Apo)

    Bpf = dv1dp0(statep, h_p)
    Bpf_inv = np.linalg.inv(Bpf)
    Apf = dp1dp0(statep, h_p)
    #Apf_inv np.linalg.inv(Apf)
    #X += DX
    binvA_po = Bpo_inv*Apo
    binvA_pf = Bpf_inv*Apf

    ap = get_acceleration(statep)

    dvpdp0 = Bpo_inv 
    dvpdt0 = -Bpo_inv*state0[3:6]
    dvpdpp = -binvA_po
    dvpdtp = binvA_po*statep[3:6] + ap
    

    dvpdpf = Bpf_inv 
    dvpdtf = -Bpf_inv*statef[3:6]
    dvpdpf = -binvA_pf
    dvpdtf = binvA_pf*statep[3:6] + ap


    DF = np.hstack((dvpdp0, dvpdt0, dvpdpp-dvpdpp, dvpdtp-dvpdtp, -dvpdpf, -dvpdtf))
    #minimum norm soluion (3.12) DX = -DF.T * (DF* DF.T).inverse() * F
    #TODO F and X
    vp_prop = propogate(state0, dt)[-1, 3:6]
    vf_prop = propogate(statep, dt)[-1, 3:6]
    
    F = np.array([vp_prop - statep[3:6], vf_prop - statef[3:6]])

    print(f'F shape {F.shape}, DF shape {DF.shape}')
    DX = -DF.T @ np.linalg.inv( np.matmul(DF,DF.T)) @ F.T
    statep[3:6] += DX
    return statep


def compute_residual(patch_points):
    n=len(patch_points) -1
    residual_vector = np.empty(n)
    for i in np.arange(n):
        state = patch_points[i]
        p1 = propogate(state[:-2],state[-2],dt,state[-1])
        residual_vector[i] = np.linalg.norm(p1[:3] - patch_points[i+1, :3])
    return np.linalg.norm(residual_vector)

if __name__ == "__main__":
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
    
    dt =100.0/tstar#.5 

    #tolerance for answer
    epsilon = 3.844e-3 # km
    #constraint vector F=p1-p1d = 0
    patch1 = np.array([p0[0], p0[1], p0[2], v0[0], v0[1], v0[2], 0, tf])
    patch2 = np.array([pd[0], pd[1], pd[2], v0[0], v0[1], v0[2], tf, tf*2  ])
    patch3 = patch1+patch2
    patch3[-2] = 2*tf
    patch3[-1] = 3*tf
    patch_points = [patch1, patch2, patch3]
   
    residual = 99999999
    i=0
    while residual > epsilon:
    #first achieve position continuity
        for i in range(len(patch_points)-1):
            print('patch_points' , patch_points[i].shape, patch_points[i+1].shape)
            patch_points[i+1] = level_1(patch_points[i], patch_points[i+1], epsilon)
        print('LEVEL 2!!!!!!!!!!!!!')    
    #Then achieve velocity continuity
        for i in range(len(patch_points) - 2):
            patch_points[i+1] = level_2(patch_points[i], patch_points[i+1],patch_points[i+2], epsilon)
        print('Global tlt iteration', i)
        i+=1
        residual = compute_residual(patch_points)
        print('residual:', residual)
