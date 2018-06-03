#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 23:56:12 2018

@author: pohsuanhuang
"""
import numpy as np
import string
from dcp import struct
from min_jerk_step import min_jerk_step
import matplotlib.pyplot as plt
# function to learn a DCP in incremental mode using a minimum jerk trajectory as
# template

# general parameters
dt        = 0.001
goal      = 1
tau       = 0.5
n_rfs     = 10
ID        = 1

#dcp.clear(ID)

DMPS = struct(ID, n_rfs,'minJerk_dcp',1)
# initialize some variables for plotting
Z=np.zeros((int(2*tau*(1./dt)),2))
T=np.zeros((int(2*tau*(1./dt)),3))
Y=T
PSI=np.zeros((int(np.floor(2*tau*(1./dt))),n_rfs))
W=PSI
X=Z
V=Z

# one sweep through the training data is all that is needed
n_fits = 1

for r in range(n_fits):

  DMPS.reset_state(ID)
  DMPS.set_goal(ID,goal, 1)
  t=0
  td=0
  tdd=0

  for i in range(int(2*tau/dt)):

    # the target trajectory computed by minimum jerk
    if (tau-i*dt > 0):
      [t,td,tdd]=min_jerk_step(t,td,tdd,goal,tau-i*dt,dt)
    

    if r == n_fits : # on the last run, only prediciton is tested
      [y,yd,ydd]=DMPS.run(ID,tau,dt)
    else : # fit the desired trajectory
      [y,yd,ydd]=DMPS.run_fit(ID,tau,dt,t,td,tdd)
    

    Z[i,:]   = np.array([DMPS.loader[ID].z, DMPS.loader[ID].zd])
    T[i,:]   = np.array([t, td, tdd])
    Y[i,:]   = np.array([y, yd, ydd])
    V[i,:]   = np.array([DMPS.loader[ID].v, DMPS.loader[ID].vd])
    X[i,:]   = np.array([DMPS.loader[ID].x,DMPS.loader[ID].xd])
    PSI[i,:] = np.array(DMPS.loader[ID].psi)
    W[i,:] = np.array(DMPS.loader[ID].w)

  


  #plotting
  time = np.arange(0,tau*2, dt)

  plt.figure(1,figsize =(10,10) )
  

  # plot position, velocity, acceleration vs. target
  plt.subplot(431)
  plt.plot(time,Y[:,0], time, T[:,0],'r--')
  plt.title('y vs.t')

  plt.subplot(432)
  plt.plot(time,Y[:,1], time, T[:,1],'r--')
  plt.title('yd vs. td')


  plt.subplot(433)
  plt.plot(time,Y[:,2], time, T[:,2],'r--')
  plt.title('ydd vs. tdd')
  

  # plot internal states
  plt.subplot(434)
  plt.plot(time,Z[:,0])
  plt.title('z')
  

  plt.subplot(435)
  plt.plot(time,Z[:,1])
  plt.title('zd')
 
  plt.subplot(436)
  plt.plot(time,PSI)
  plt.title('Weighting Kernels')

  plt.subplot(437)
  plt.plot(time,V[:,0])
  plt.title('v')
  
  plt.subplot(438)
  plt.plot(time,V[:,1])
  plt.title('vd')
 
  plt.subplot(439)
  plt.plot(time,W)
  plt.title('Linear Model Weights over Time')
  
  plt.subplot(4,3,10)
  plt.plot(time,X[:,0])
  plt.title('x')
  
  plt.subplot(4,3,11)
  plt.plot(time,X[:,1])
  plt.title('xd')
  
  plt.subplot(4,3,12)
  plt.plot(W[-1,:])
  plt.title('Weights')
  plt.xlabel('tau=%f'.format(tau))

  plt.show()