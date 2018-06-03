#!/usr/bIn/env python2
# -*- codIng: utf-8 -*-
"""
Created on Tue Apr 10 21:58:16 2018

@author: pohsuanhuang

Python implementation of 
Ijspeert A, Nakanishi J, Schaal S (2003) LearnIng attractor landscapes for 
# learnIng motor primitives. In: Becker S, Thrun S, Obermayer K (eds) Advances 
# In Neural Information ProcessIng Systems 15. MIT Press, Cambridge, MA.
http://www-clmc.usc.edu/publications/I/ijspeert-NIPS2002.pdf.
modified from Matlab code from 
Prof. Stefan Schaal's lab website.
"""

import numpy as np
from easydict import EasyDict as edict
import string
import min_jerk_step


class struct(object):
    
    loader = {}
    def __init__(self, *args):
        ID = args[0]
        struct.loader[ID]=dcp(*args)
        
    def reset_state(self, *args):
        ID = args[0]
        struct.loader[ID].reset_state(*args)
        
    def set_goal(self, *args):
        ID = args[0]
        struct.loader[ID].set_goal(*args)
        
    def set_scale(self, *args):
        ID = args[0]
        struct.loader[ID].set_scale(*args)
        
    def run(self, *args):
        ID = args[0]
        return struct.loader[ID].run(*args)
    
    def change(self, *args):
        ID = args[0]
        struct.loader[ID].change(*args)
        
    def run_fit(self, *args):
        ID = args[0]
        return struct.loader[ID].run_fit(*args)  
        
    def batch_fit(self, *args):
        ID = args[0]
        return struct.loader[ID].batch_fit(*args) 
        
    def structure(self, *args):
        ID = args[0]
        struct.loader[ID].structure(*args)
        
    def clear(self, *args):
        ID = args[0]
        struct.loader[ID].clear(*args)
        
    def minjerk(self, *args):
        ID = args[0]
        return struct.loader[ID].minjerk(*args)

        
class dcp(object) :
    
        
    '''          
    # ---------------  Different Actions of the program ------------------------
    #
    # Initialize a DCP model:
    # FORMAT dcp('Init',ID,n_rfs,name,flag)
    # ID              : desired ID of model
    # n_rfs           : number of local lInear models
    # name            : a name for the model
    # flag            : flag=1 use 2nd order canonical system, flag=0 use 1st order
    #
    # alternatively, the function is called as
    #
    # FORMAT dcp('Init',ID,d,)
    # d               : a complete data structure of a dcp model
    #
    # returns nothIng
    # -------------------------------------------------------------------------
    #
    # Set the goal state:
    # FORMAT dcp('Set_Goal',ID,g,flag)
    # ID              : ID of model
    # g               : the new goal
    # flag            : flag=1: update x0 with current state, flag=0: don't update x0
    #
    # returns nothIng
    # -------------------------------------------------------------------------
    #
    # Set the scale factor of the movement:
    # FORMAT dcp('Set_Scale',ID,s,flag)
    # ID              : ID of model
    # s               : the new scale
    #
    # returns nothIng
    # -------------------------------------------------------------------------
    #
    # Run the self.dcps:
    # FORMAT [y,yd,ydd]=dcp('Run',ID,tau,dt,ct,cc)
    # ID              : ID of model
    # tau             : global time constant to scale speed of system
    # dt              : Integration time step
    # ct              : couplIng term for transformation system (optional)
    # cc              : couplIng term for canonical system (optional)
    # ct_tau          : couplIng term for transformation system's time constant (optional)
    # cc_tau          : couplIng term for canonical system's time constant (optional)
    # cw              : additive couplIng term for parameters (optional)
    #
    # returns y,yd,ydd, i.e., current pos,vel,acc, of the dcp
    # -------------------------------------------------------------------------
    #
    # Change values of a dcp:
    # FORMAT dcp('Change',ID,pname,value)
    # ID              : ID of model
    # pname           : parameter name
    # value           : value to be assigned to parameter
    #
    # returns nothIng
    # -------------------------------------------------------------------------
    #
    # Run the self.dcps:
    # FORMAT dcp('Run',ID,tau,dt,t,td)
    # ID              : ID of model
    # tau             : time constant to scale speed, tau is roughly movement
    #                   time until convergence
    # dt              : Integration time step
    #
    # returns y,yd,ydd, i.e., current pos,vel,acc, of the dcp
    # -------------------------------------------------------------------------
    #
    # Run the dcp and update the weights:
    # FORMAT dcp('Run_Fit',ID,tau,dt,t,td,tdd)
    # ID              : ID of model
    # tau             : time constant to scale speed, tau is roughly movement
    #                   time until convergence
    # dt              : Integration time step
    # t               : target for y
    # td              : target for yd
    # tdd             : target for ydd
    #
    # returns y,yd,ydd, i.e., current pos,vel,acc, of the dcp
    # -------------------------------------------------------------------------
    #
    # Fit the dcp to a complete trajectory In batch mode:
    # FORMAT dcp('Batch_Fit',ID,tau,dt,T,Td,Tdd)
    # ID              : ID of model
    # tau             : time constant to scale speed, tau is roughly movement
    #                   time until convergence the goal
    # dt              : somple time step In given trajectory
    # T               : target trajectory for y
    # Td              : target trajectory for yd (optional, will be generated
    #                   as dT/dt otherwise
    # Tdd             : target trajectory for ydd (optional, will be generated
    #                   as dTd/dt otherwise
    #
    # returns y,yd,ydd, i.e., current pos,vel,acc, of the dcp
    # -------------------------------------------------------------------------
    #
    # Return the data structure of a dcp model
    # FORMAT [d] = dcp('Structure',ID)
    # ID              : desired ID of model
    #
    # returns the complete data structure of a dcp model, e.g., for savIng or
    # InspectIng it
    # -------------------------------------------------------------------------
    #
    # Reset the states of a dcp model to zero (or a given state)
    # FORMAT [d] = dcp('Reset_State',ID)
    # ID              : desired ID of model
    # y               : the state to which the primitive is set (optional)
    #
    # returns nothIng
    # -------------------------------------------------------------------------
    #
    # Clear the data structure of a LWPR model
    # FORMAT dcp('Clear',ID)
    # ID              : ID of model
    #
    # returns nothIng
    # -------------------------------------------------------------------------
    #
    # Initializes the dcp with a mInimum jerk trajectory
    # FORMAT dcp('MinJerk',ID)
    # ID              : ID of model
    #
    # returns nothIng
    # -------------------------------------------------------------------------
    '''
    
    def __init__(self, *args):
    # at least two arguments are needed
        assert len(args) >= 2 , 'Incorrect call to dcp'
    # .........................................................................
        if len(args) == 3 : 
          ID = args[0]
          self  = args[1]
        else :
          # this Initialization is good for 0.5 seconds movement for tau=0.5
          ID                  = args[0]
          n_rfs               = args[1]

          self.name  = args[2]
          self.c_order = 0
          if len(args) == 4:
            self.c_order = args[3]
          
          # the time constants for chosen for critical dampIng
          self.alpha_z = 25
          self.beta_z  = self.alpha_z/4
          self.alpha_g = self.alpha_z/2
          self.alpha_x = self.alpha_z/3
          self.alpha_v = self.alpha_z
          self.beta_v  = self.beta_z
          # Initialize the state variables
          self.z       = 0
          self.y       = 0 
          self.x       = 0
          self.v       = 0
          self.zd      = 0
          self.yd      = 0 
          self.xd      = 0
          self.vd      = 0
          self.ydd     = 0 
          # the current goal state
          self.g       = 0
          self.gd      = 0
          self.G       = 0
          # the current start state of the primitive
          self.y0      = 0
          # the orgInal amplitude (max(y)-mIn(y)) when the primitive was fit
          self.A       = 0
          # the origInal goal amplitude (G-y0) when the primitive was fit
          self.dG      = 0
          # the scale factor for the nonlInear function
          self.s       = 1
                    
          t = np.linspace( 0, 1 ,n_rfs) * 0.5
          if self.c_order == 1 :
            # the local models, spaced on a grid In time by applyIng the
            # anaytical solutions x(t) = 1-(1+alpha/2*t)*np.exp(-alpha/2*t)
            def func(t) : 
                return 1+self.alpha_z/2.0*t * np.exp(-self.alpha_z/2.0*t)
            vfunc = np.vectorize(func)
            self.c       = np.array(vfunc(t))
            # we also store the phase velocity at the centers which is used by some
            # applications: xd(t) = (-alpha/2)*x(t) + alpha/2*np.exp(-alpha/2*t)
            def func1(t) : 
                return (self.alpha_z/2.0) * np.exp(-self.alpha_z/2.0*t)
            vfunc1 = np.vectorize(func1)
            self.cd      = self.c * (-self.alpha_z/2.0) + np.array(vfunc1(t))
          else :
            # the local models, spaced on a grid In time by applyIng the
            # anaytical solutions x(t) = np.exp(-alpha*t)
            def func2(t) :
                return np.exp(-self.alpha_x*t)
            vfunc2 = np.vectorize(func2)
            self.c       = np.array(vfunc2(t))
            assert len(self.c) > 0, 'c is empty array'
            # we also store the phase velocity at the centers which is used by some
            # applications: xd(t) = x(t)*(-self.alpha_x)
            self.cd      = self.c*(-self.alpha_x)
          
          
          self.psi     = np.zeros(n_rfs)
          self.w       = np.zeros(n_rfs)
          self.sx2     = np.zeros(n_rfs)
          self.sxtd    = np.zeros(n_rfs)              
          self.D       = np.square(np.diff(self.c)*0.55)
          self.D       = np.reciprocal(np.append( self.D, self.D[-1]))
          self.Lambda  = 1
          
        
        
    # .........................................................................
    def reset_state(self, *args):
        ID               = args[0]
        if len(args)> 2 :
          y = args[1]
        else :
          y = 0
        
        # Initialize the state variables
        self.z       = 0
        self.y       = y 
        self.x       = 0
        self.v       = 0
        self.zd      = 0
        self.yd      = 0 
        self.xd      = 0
        self.vd      = 0
        self.ydd     = 0 
        # the goal state
        self.G       = y
        self.g       = y
        self.gd      = 0
        self.y0      = y
        self.s       = 1
        
    # .........................................................................
    def set_goal(self, *args):
        ID               = args[0]
        self.G       = args[1]
        if (self.c_order == 0) :
          self.g     = self.G
        
        flag             = args[2]
        if (flag) :
          self.x     = 1
          self.y0    = self.y
        
        if (self.A != 0) : # check whether dcp has been fit
          if self.A/(abs(self.dG)+1.e-10) > 2.0 :
            # amplitude-based scalIng needs to be set np.explicity
            pass
          else :
            # dG based scalIng cab work automatically
            self.s       = (self.G-self.y0)/self.dG
          
        
        
    # .........................................................................
    def set_scale(self, *args):
        ID               = args[0]
        self.s       = args[1]
            
    # .........................................................................
    def run(self, *args):
        ID               = args[0]
        tau              = 0.5/args[1] # tau is relative to 0.5 seconds nomInal movement time
        dt               = args[2]
        
        if len(*args) > 3 :
          ct  = args[3]
        else :
          ct  = 0
        
    
        if len(*args) > 4 :
          cc  = args[4]
        else:
          cc  = 0
        
        
        if len(*args) > 5 :
          ct_tau  = args[5]
        else :
          ct_tau  = 1
        
        
        if len(*args) > 6 :
          cc_tau  = args[6]
        else:
          cc_tau  = 1
        
        
        if len(*args) > 7:
          cw  = args[7]
        else:
          cw  = 0
        
        
        # the weighted sum of the locally weighted regression models
        self.psi = np.exp(-0.5*np.multiply((np.square(self.x-self.c)),self.D))
        amp          = self.s
        if self.c_order == 1:
          In = self.v
        else:
          In = self.x
        
        f = sum(In*np.multiply(self.w+cw,self.psi))/sum(self.psi+np.exp(-10)) * amp
        if self.c_order == 1:
          self.vd = (self.alpha_v*(self.beta_v*(0-self.x)-self.v)+cc)*tau*cc_tau
          self.xd = self.v*tau*cc_tau
        else :
          self.vd = 0
          self.xd = (self.alpha_x*(0-self.x)+cc)*tau*cc_tau
        
        
        self.zd = (self.alpha_z*(self.beta_z*(self.g-self.y)-self.z)+f+ct)*tau*ct_tau
        self.yd = self.z*tau*ct_tau
        self.ydd= self.zd*tau*ct_tau
        
        self.gd = self.alpha_g*(self.G-self.g)
        
        self.x  = self.xd*dt+self.x
        self.v  = self.vd*dt+self.v
        
        
        self.z  = self.zd*dt+self.z
        self.y  = self.yd*dt+self.y
        
        self.g  = self.gd*dt+self.g
    
            
        return self.y, self.yd, self.ydd, self.psi*In/sum(self.psi+np.exp(-10)) * amp
        
        
    # .........................................................................
    def change(self, *args) :
        ID      = args[0]
        command = string.join('self.dcps(#d).#s = args[2]',ID,args[1])
        eval(command)
        
    # .........................................................................
    def run_fit(self, *args):
        global min_y, max_y

        ID               = args[0]
        tau              = 0.5/args[1] # tau is relative to 0.5 seconds nomInal movement time
        dt               = args[2]
        t                = args[3]
        td               = args[4]
        tdd              = args[5]
        
        # check whether this is the first time the primitive is fit, and record the
        # amplitude and dG Information
        if self.A == 0:
          self.dG = self.G - self.y0
          if self.x == 1:
            min_y = +1.e10
            max_y = -1.e10
            self.s = 1
          
        
            
        # the regression target
        amp              = self.s
        ft               = (tdd/tau**2-self.alpha_z*(self.beta_z*(self.g-t)-td/tau))/amp
        
        # the weighted sum of the locally weighted regression models
        self.psi = np.exp(-0.5*(np.multiply(np.square((self.x-self.c)),self.D)))
        # update the regression
        if self.c_order == 1:
          self.sx2  = self.sx2*self.Lambda + self.psi*self.v**2
          self.sxtd = self.sxtd*self.Lambda + self.psi*self.v*ft
          self.w    = np.divide(self.sxtd,(self.sx2+np.exp(-10)).astype(float))
        else:
          self.sx2  = self.sx2*self.Lambda + self.psi*self.x**2
          self.sxtd = self.sxtd*self.Lambda + self.psi*self.x*ft
          self.w    = np.divide(self.sxtd,(self.sx2+np.exp(-10)).astype(float))
        
        
        # compute nonlInearity
        if self.c_order == 1:
          In = self.v
        else:
          In = self.x
        
        f     = sum(In*np.multiply(self.w,self.psi))/sum(self.psi+np.exp(-10)) * amp
        

        # Integrate
        if self.c_order == 1:
          self.vd = (self.alpha_v*(self.beta_v*(0-self.x)-self.v))*tau
          self.xd = self.v*tau
        else :
          self.vd = 0
          self.xd = self.alpha_x*(0-self.x)*tau
        
        
        # note that yd = td = z*tau   ==> z=td/tau the first equation means
        # simply self.zd = tdd
        self.zd = (self.alpha_z*(self.beta_z*(self.g-self.y)-self.z)+f)*tau
        self.yd = self.z*tau
        self.ydd= self.zd*tau
        
        self.gd = self.alpha_g*(self.G-self.g)
        
        self.x  = self.xd*dt+self.x
        self.v  = self.vd*dt+self.v
        
        self.z  = self.zd*dt+self.z
        self.y  = self.yd*dt+self.y
        
        self.g  = self.gd*dt+self.g
    
        
        
        if self.A == 0:
          max_y = max(max_y,self.y)
          min_y = min(min_y,self.y)
          if self.x < 0.0001 :
            self.A = max_y - min_y
          

        return self.y, self.yd, self.ydd
        
    # .........................................................................
    def batch_fit(self, *args):        
        ID               = args[0]
        tau              = 0.5/args[1] # tau is relative to 0.5 seconds nomInal movement time
        dt               = args[2]
        T                = args[3]
        if len(*args) > 5 :
          Td               = args[4]
        else :
          Td               = diffnc(T,dt)
        
        if len(*args) > 6 :
          Tdd              = args[5]
        else:
          Tdd              = diffnc(Td,dt)
        
        
        # the start state is the first state In the trajectory
        y0 = T(1)
        g  = y0
        
        # the goal is the last state In the trajectory
        goal = T()
        if self.c_order == 0:
          g = goal
        
        
        # the amplitude is the max(T)-mIn(T)
        A    = max(T)-min(T)
         
        # compute the hidden states
        X = np.zeros(np.size(T))
        V = np.zeros(np.size(T))
        G = np.zeros(np.size(T))
        x = 1
        v = 0
       
        for i in range(len(T)):
          
          X[i] = x
          V[i] = v
          G[i] = g
          
          if self.c_order == 1:
            vd   = self.alpha_v*(self.beta_v*(0-x)-v)*tau
            xd   = v*tau
          else:
            vd   = 0
            xd   = self.alpha_x*(0-x)*tau
          
          gd   = (goal - g) * self.alpha_g
          
          x    = xd*dt+x
          v    = vd*dt+v
          g    = gd*dt+g
          
        
        
        # the regression target
        self.dG = goal - y0
        self.A  = max(T)-min(T)
        self.s  = 1  # for fittIng a new primitive, the scale factor is always equal to one
    
        amp = self.s
        Ft  = (Tdd/tau^2-self.alpha_z*(self.beta_z*(G-T)-Td/tau)) / amp
        
        # compute the weights for each local model along the trajectory
        PSI = np.exp(-0.5*np.square(X*np.ones(1,len(self.c))-np.ones(len(T),1)*self.c)*(np.ones(len(T),1)*self.D))
    
        # compute the regression
        if self.c_order == 1:
          self.sx2  = np.sum(((np.square(V))*np.ones(1,len(self.c)))*PSI,0)
          self.sxtd = np.sum(((V*Ft)*np.ones(1,len(self.c)))*PSI,0)
          self.w    = np.divide(self.sxtd,(self.sx2+np.exp(-10).astype(float)))
        else:
          self.sx2  = np.sum((np.square(X)*np.ones(1,len(self.c)))*PSI,0)
          self.sxtd = np.sum(((X*Ft)*np.ones(1,len(self.c)))*PSI,0)
          self.w    = np.divide(self.sxtd,(self.sx2+np.exp(-10)))
        
        
        # compute the prediction
        if (self.c_order == 1):
          F     = np.divide(np.sum((V*self.w)*PSI,1),np.sum(PSI,1)) * amp      
        else:
          F     = np.divide(np.sum((X*self.w)*PSI,1),np.sum(PSI,1)) * amp
        
        z     = 0
        zd    = 0
        y     = y0
        Y     = np.zeros(np.size(T))
        Yd    = np.zeros(np.size(T))
        Ydd   = np.zeros(np.size(T))
        
        for i in range(len(T)) :
          
          Ydd[i] = zd*tau
          Yd[i]  = z
          Y[i]   = y
          
          zd   = (self.alpha_z*(self.beta_z*(G(i)-y)-z)+F(i))*tau
          yd   = z
          
          z    = zd*dt+z
          y    = yd*dt+y
                
        
           
        return Y, Yd, Ydd
        
    # .........................................................................
    def structure(self, *args):
        global dcps 

        ID     = args[0]
        return self
        
    # .........................................................................
    def clear(self, *args):
        ID     = args[0]
        if  self.dcps in locals():
          if len(self.dcps) >= ID:
            self = []
          
        
        
    # .........................................................................
    def minjerk(self, *args):
        ID     = args[0]
        
        # generate the mInimum jerk trajectory as target to learn from
        t=0
        td=0
        tdd=0
        goal = 1
        
        dcp('reset_state',ID)
        dcp('set_goal',ID,goal,1)
        tau = 0.5
        dt = 0.001
        T=np.zeros(2*tau/dt,3)
    
        for i in np.linspace(0, 2*tau/dt):
          [t,td,tdd]=min_jerk_step(t,td,tdd,goal,tau-i*dt,dt)
          T[i+1,:]  = [t, td, tdd]        
    
        # batch fittIng
        i = round(2*tau/dt) # only fit the part of the trajectory with the signal
        [Yp,Ypd,Ypdd] = dcp.batch_fit(ID,tau,dt,T[1:i,1],T[1:i,2],T[1:i,3])
        
   
    