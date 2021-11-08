def Gen_MixedData(npart):
	import numpy as np
	from tqdm import tqdm
	npart_max=10
	file= './data_gate/'
	fid = ['Particle','Coincidences']
	if npart>npart_max:
		npart=npart_max
	index=np.random.choice(npart_max, npart)
	for ipart in tqdm(np.arange(npart)):
		data=np.load(file+fid[0]+'%d'%ipart+fid[1]+'.npz')	
		traj = data['traj']
		lors = data['lors']
		if ipart == 0:
			LoRs=lors
			Traj=traj[:,np.newaxis,:]
		else:
			LoRs=np.concatenate((lors,LoRs),axis=0)
			Traj=np.concatenate((Traj,traj[:,np.newaxis,:]),axis=1)
	LoRs[:,4:]-= LoRs[:,1:4]
	LoRs[:,4:]/= np.linalg.norm(LoRs[:,4:],axis=-1)[:,np.newaxis]
	LoRs = LoRs[LoRs[:,0].argsort()]
	LoRs[:,0]*=1000
	Traj[:,:,0]-=Traj[0,0,0]
	Traj=Traj[Traj[:,0,0]<10000]
	return LoRs,Traj


def axisEqual3D(ax):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	import numpy as np
	extents		=	np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
	sz			= 	extents[:,1] - extents[:,0]
	centers		=	np.mean(extents, axis=1)
	maxsize		=	max(abs(sz))
	r			=	maxsize/2
	for ctr, dim in zip(centers, 'xyz'):
		getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def InitPlot(x,L=None,x0=None,s=None,lim=None):
	import matplotlib.pyplot as plt
	import numpy as np
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.cm as cm
	cmap = cm.get_cmap('nipy_spectral')
	plt.ion()
	fig 		=	plt.figure(figsize=[10,10])
	ax			=	fig.add_subplot(111, projection='3d')
	if L is not None:
		if len(L)==2:
			x1,x2=L
			for i in np.arange(x1.shape[0]):
				ax.plot([x1[i,0],x2[i,0]],[x1[i,1],x2[i,1]],[x1[i,2],x2[i,2]],'k-',linewidth=0.3,alpha=0.1)
		elif len(L)==3:
			x1,x2,w=L
			colors=cmap(np.linspace(0,1,w.shape[1]+1)[1:-1])
			colorsx=np.concatenate((np.copy(colors),np.array([[0.,0.,0.,1.]])),axis=0)
			colorsx[:,-1]/=5
			rgba_colors=np.minimum(1,np.sum(colorsx[np.newaxis,:,:]*w[:,:,np.newaxis],axis=1))
			for i in np.arange(x1.shape[0]):
				ax.plot([x1[i,0],x2[i,0]],[x1[i,1],x2[i,1]],[x1[i,2],x2[i,2]],'-',color=rgba_colors[i],linewidth=0.4)
	colors=cmap(np.linspace(0,1,x.shape[0]+2)[1:-1])
	P=[]
	if s is not None:
		for ip in np.arange(x.shape[0]):
			pp,=ax.plot([x[ip,0]],[x[ip,1]],[x[ip,2]],'o',color=colors[ip],markersize=np.maximum(s[ip],5),alpha=np.maximum(0.1,np.minimum(1,5/s[ip])),markeredgewidth=1,markeredgecolor='k')
			P.append(pp)
	else:
		for ip in np.arange(x.shape[0]):
			pp,=ax.plot([x[ip,0]],[x[ip,1]],[x[ip,2]],'o',color=colors[ip],markersize=5)
			P.append(pp)
	if x0 is not None:
		pp,=ax.plot(x0[:,0],x0[:,1],x0[:,2],'kx',markersize=5)
		P.append(pp)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	if lim is not None:
		xm,xM,ym,yM,zm,zM=lim
		ax.set_xlim(xm,xM)
		ax.set_ylim(ym,yM)
		ax.set_zlim(zm,zM)
	axisEqual3D(ax)
	fig.canvas.draw()
	fig.canvas.flush_events()
	return fig,P

def UpdatePlot(fig,P,x,s=None,x0=None):
	import matplotlib.pyplot as plt
	import numpy as np
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.cm as cm
	cmap = cm.get_cmap('nipy_spectral')
	plt.ion()
	colors=cmap(np.linspace(0,1,x.shape[0]+2)[1:-1])
	colors[:,-1]=np.maximum(0.1,np.minimum(1,5/s))
	if s is not None:
		for ip in np.arange(x.shape[0]):
			P[ip].set_data_3d([x[ip,0]], [x[ip,1]], [x[ip,2]])
			P[ip].set_markersize(np.maximum(s[ip],5))
			P[ip].set_markerfacecolor(colors[ip])
			P[ip].set_alpha(np.maximum(0.1,np.minimum(1,5/s[ip])))
	else:
		for ip in np.arange(x.shape[0]):
			P[ip].set_data_3d([x[ip,0]], [x[ip,1]], [x[ip,2]])
	if x0 is not None:
		P[-1].set_data_3d(x0[:,0],x0[:,1],x0[:,2])			
	fig.canvas.draw()
	fig.canvas.flush_events()
	return P

def dist_matrix(x,L):
	import numpy as np
	from numpy import newaxis as nx
	from scipy.special import factorial
	o = x.shape[-1]//3-1
	X = x[nx,:,:3]-L[:,nx,1:4]
	if o > 0:
		DT = L[:,0]-L[:,0].mean()
		for i in np.arange(o)+1:
			X += x[nx,:,3*i:3*(i+1)]*DT[:,nx,nx]**i/factorial(i)
	d2 = np.sum(X**2,axis=-1)-np.sum(X*L[:,nx,4:],axis=-1)**2
	return d2

def Latent_weights(d2,s,eps=0,r=None):
	import numpy as np
	from numpy import newaxis as nx
	if r is None:
		w = np.exp(-d2/2/s**2)/s**2+10**(-20)
		w /= np.sum(w,axis=-1)[:,nx]+eps
	else:
		w = np.exp(-d2/2/s**2)*r/s**2+10**(-20)
		w/= np.sum(w,axis=-1)[:,nx]+eps*(1-np.sum(r))
	return w

def Centroid(L,w,o=0):
	import numpy as np
	from numpy import newaxis as nx
	from scipy.special import factorial
	m = (np.identity(3)[nx,:,:]-L[:,nx,4:]*L[:,4:,nx])*w[:,nx,nx]
	if o==0:
		M = np.sum(m,axis=0)
		V = np.sum(np.sum(m*L[:,nx,1:4],axis=-1),axis=0)
	else:
		DT = L[:,0]-L[:,0].mean()
		M = np.zeros((3*(o+1),3*(o+1)))
		V = np.zeros(3*(o+1))
		for i in np.arange(o+1):
			V[i*3:(i+1)*3] = np.sum(np.sum(m*L[:,nx,1:4]*DT[:,nx,nx]**i/factorial(i),axis=-1),axis=0)
			for j in np.arange(o-i+1)+i:
				M[i*3:(i+1)*3,j*3:(j+1)*3] = np.sum(m*DT[:,nx,nx]**(i+j)/(factorial(i)*factorial(j)),axis=0)
		for i in np.arange(o+1):
			for j in np.arange(i):
				M[i*3:(i+1)*3,j*3:(j+1)*3] = M[j*3:(j+1)*3,i*3:(i+1)*3]	
	x = np.matmul(np.linalg.inv(M),V)
	return x

def Centroid_Multi(L,w,o=0,parallel=False):
	import numpy as np
	if parallel:
		import joblib as jl
		x = np.asarray(jl.Parallel(n_jobs=-1)(jl.delayed(Centroid)(L,w[:,i],o=o) for i in np.arange(w.shape[1])))
	else:
		x = np.asarray([Centroid(L,w[:,i],o=o) for i in np.arange(w.shape[1])])
	return x

def Initial_centroid(L):
	import numpy as np
	from numpy import newaxis as nx
	w = np.ones(L.shape[0])
	x = Centroid(L,w)
	d2	= dist_matrix(x[nx,:],L)
	s=np.sqrt(np.mean(d2))
	return x,s

def EM_Single_Step(x,s,L,eps=0,parallel=False,r=None,d2=None):
	import numpy as np
	if d2 is None:
		d2 = dist_matrix(x,L)
	w=Latent_weights(d2,s,eps=eps,r=r)
	r = np.mean(w,axis=0)
	x = Centroid_Multi(L,w,o=x.shape[-1]//3-1,parallel=parallel)
	d2 = dist_matrix(x,L)
	s = np.sqrt(np.mean(d2*w,axis=0)/r/2)
	return x,s,r,d2



		





