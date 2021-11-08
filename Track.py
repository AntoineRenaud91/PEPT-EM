import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from mpl_toolkits.mplot3d import Axes3D, proj3d
from tqdm import tqdm
from EM_Tools import *

npart = 5 # number of particle to track
L,Traj = Gen_MixedData(npart) # generate a random mixed data sample

order = 1	# order of the tracking
nT = 2000 # number of frame
nperframe = L[:,0].size//nT # number of LoRs per frame
n_initsetp = 50 # number of steps for the first frame
nstep = 20 # number of steps for the next frames
nshow = 50 # update the plot every nshow steps
eps	= 1e-4 # weight of the outlier set
nsave=50

######## Initialisation #######
x0,s0 =	Initial_centroid(L[:nperframe])
x_m=x0[np.newaxis,:]+np.random.normal(0,s0,(npart,3)) # Particles positions
if order>0:
	x_m=np.concatenate((x_m,np.zeros((npart,3*order))),axis=-1) # Particles velocities, accelerations ...
sigma_m = np.ones(npart)*s0 # variances of the cluster
rho_m = np.ones(npart)/npart # densities of the cluster
d2=None # centroid to LoRs distance matrix
fig,P=InitPlot(x_m[:,:3],x0=Traj[0][:,1:],s=sigma_m)
for i in tqdm(np.arange(n_initsetp)): # first frame loop 
	x_m,sigma_m,rho_m,d2=EM_Single_Step(x_m,sigma_m,L[:nperframe],eps=eps,parallel=True,d2=d2)	
	P=UpdatePlot(fig,P,x_m,sigma_m) 
######## Trajectories ########
t_track=np.array([np.mean(L[:nperframe,0])])
x_track=np.copy(x_m)[np.newaxis,:,:]
s_track=np.copy(sigma_m)[np.newaxis,:]
r_track=np.copy(rho_m)[np.newaxis,:]
print('\n')
######## Main Loop #######
for iT in tqdm(np.arange(nT-1)+1):
	d2=None
	for i in np.arange(nstep): # next frames loop
		x_m,sigma_m,rho_m,d2=EM_Single_Step(x_m,sigma_m,L[iT*nperframe:(iT+1)*nperframe],eps=eps,parallel=True,d2=d2)
	t_track = 	np.append(t_track,np.mean(L[iT*nperframe:(iT+1)*nperframe,0]))
	x_track	=	np.concatenate((x_track,x_m[np.newaxis,:,:]),axis=0)
	s_track	=	np.concatenate((s_track,sigma_m[np.newaxis,:]),axis=0)
	r_track =	np.concatenate((r_track,rho_m[np.newaxis,:]),axis=0)
	if iT%nsave==0: # update plot
		ittraj=np.int_((Traj.shape[0]-1)*iT/nT)
		P=UpdatePlot(fig,P,x_m,sigma_m,x0=Traj[ittraj][:,1:])
plt.close('all')
plt.ioff()
