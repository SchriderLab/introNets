import sys
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import dadi
import numpy
import scipy
import pyOpt
import dadiFunctions

def readLFromFSFile(fsFileName):
    with open(fsFileName) as fsFile:
        lines = []
        for line in fsFile:
            if not line.strip().startswith("#"):
                lines.append(line)
        ns = [int(x)-1 for x in lines[0].split()[:2]]
        L = sum([int(x) for x in lines[1].split()])
    return L, ns

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
except:
    raise ImportError('mpi4py is required for parallelization')

inFile, swarmSize, figFileName = sys.argv[1:]
swarmSize = int(swarmSize)

L, ns = readLFromFSFile(inFile)
data = dadi.Spectrum.from_file(inFile)
data = data.fold()

pts_l = [30,40,50]

# The Demographics1D and Demographics2D modules contain a few simple models,
# mostly as examples. We could use one of those.
func = dadiFunctions.IM

upper_bound = [100,10,100, 10, 20, 10,10]
lower_bound = [1e-1,1e-2,1, 1e-2, 0.1,0,0]

#params = [1.53508047e+00, 2.01954315e-01, 7.05672270e+00, 1.04123880e+00, 2.43171832e-01, 6.95840502e-01, 1.12649753e+00]
#p0 = params
#p1 = dadi.Misc.perturb_params(p0, fold=0.5, lower_bound=lower_bound, upper_bound=upper_bound)

p1=dadiFunctions.makeRandomParams(lower_bound,upper_bound)

func_ex = dadi.Numerics.make_extrap_func(func)

# Instantiate Optimization Problem 

#lamda definition of objective, stuffing constraint into return value
#objfunc = lambda x: (dadi.Inference._object_func(x, data, func_ex, pts_l, 
#                                   lower_bound=lower_bound,
#                                   upper_bound=upper_bound),(x[4]-x[3]),0)

def objfunc(x):
	f = dadi.Inference._object_func(x, data, func_ex, pts_l, 
	                                  lower_bound=lower_bound,
                                          upper_bound=upper_bound)
	g=[]
#	g = [0.0]*2
#	g[0] = x[3]-x[4]
#	g[1] = x[3]-x[2]
	fail = 0
	return f,g,fail
	
opt_prob = pyOpt.Optimization('dadi optimization',objfunc)
opt_prob.addVar('nu1_0','c',lower=lower_bound[0],upper=upper_bound[0],value=p1[0])
opt_prob.addVar('nu2_0','c',lower=lower_bound[1],upper=upper_bound[1],value=p1[1])
opt_prob.addVar('nu1','c',lower=lower_bound[2],upper=upper_bound[2],value=p1[2])
opt_prob.addVar('nu2','c',lower=lower_bound[3],upper=upper_bound[3],value=p1[3])
opt_prob.addVar('T','c',lower=lower_bound[4],upper=upper_bound[4],value=p1[4])
opt_prob.addVar('mSim_Sech','c',lower=lower_bound[5],upper=upper_bound[5],value=p1[5])
opt_prob.addVar('mSech_Sim','c',lower=lower_bound[6],upper=upper_bound[6],value=p1[6])
opt_prob.addObj('f')

if myrank == 0:
	print opt_prob


#optimize
psqp = pyOpt.ALPSO(pll_type='DPM')
psqp.setOption('printOuterIters',1)
#psqp.setOption('maxOuterIter',1)
#psqp.setOption('stopCriteria',0)
psqp.setOption('SwarmSize',swarmSize)
psqp(opt_prob)
print opt_prob.solution(0)

popt = numpy.zeros(len(p1))
for i in opt_prob._solutions[0]._variables:
    popt[i]= opt_prob._solutions[0]._variables[i].__dict__['value']

model = func_ex(popt, ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(model, data)
mu=3.5e-9
if myrank == 0:
	print 'Optimized log-likelihood:', ll_opt
	print 'AIC:', -2*ll_opt + 2*len(popt)
	#scaled estimates
	theta0 = dadi.Inference.optimal_sfs_scaling(model, data)
	Nref= theta0 / mu / L / 4

	

	print 'Nref:',Nref
	paramsTxt =['nu1_0','nu2_0','nu1','nu2','T','2Nref_m12','2Nref_m21']
	scaledParams = [Nref*popt[0],Nref*popt[1],Nref*popt[2],Nref*popt[3],2*Nref/15*popt[4],popt[5],
	                popt[6]]
	for i in range(len(paramsTxt)):
		print paramsTxt[i],':',str(scaledParams[i])
	print ""
	print repr(popt)

############### 
# Now refine the optimization using Local Optimizer
# Instantiate Optimizer (SLSQP) 
# Instantiate Optimizer (SLSQP)
slsqp = pyOpt.SLSQP()
# Solve Problem (With Parallel Gradient)
if myrank == 0:
	print 'going for second optimization'

slsqp(opt_prob.solution(0),sens_type='FD',sens_mode='pgc')
print opt_prob.solution(0).solution(0)
opt = numpy.zeros(len(p1))
for i in opt_prob._solutions[0]._solutions[0]._variables:
	popt[i]= opt_prob._solutions[0]._solutions[0]._variables[i].__dict__['value']
	# 
model = func_ex(popt, ns, pts_l)
ll_opt = dadi.Inference.ll_multinom(model, data)
if myrank == 0:	  
	print 'After Second Optimization'
	print 'Optimized log-likelihood:', ll_opt
	print 'AIC:', -2*ll_opt + 2*len(popt)

	#scaled estimates
	theta0 = dadi.Inference.optimal_sfs_scaling(model, data)
	print 'with u = %e' %(mu)
	Nref= theta0 / mu / L / 4
	

	print 'Nref:',Nref
	paramsTxt =['nu1_0','nu2_0','nu1','nu2','T','2Nref_m12','2Nref_m21']
	scaledParams = [Nref*popt[0],Nref*popt[1],Nref*popt[2],Nref*popt[3],2*Nref/15*popt[4],popt[5],
	                popt[6]]
	for i in range(len(paramsTxt)):
		print paramsTxt[i],':',str(scaledParams[i])
	print ""
	print repr(popt)

        fig = plt.figure()
        plt.clf()
        dadi.Plotting.plot_2d_comp_multinom(model, data, vmin=0.0001)
        plt.savefig(figFileName)
