#
#
# Code Usage Description:
# To run SGD and plot results, we have 4 options:
# 1) Call main(1): Produces a combined squared error plot for (i) Last Iterate Constant, (ii) Iterate Average Constant, and (iii) Last Iterate Decreasing
# 2) Call main(2): Produces a parameter estimate plot and a squared error plot for single configuration of Last Iterate Constant
# 3) Call main(3): Produces a parameter estimate plot and a squared error plot for single configuration of Iterate Average Constant
# 4) Call main(4): Produces a parameter estimate plot and a squared error plot for single configuration of Last Iterate Decreasing
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
#
# Configuration:
# To replicate the charts in the paper exactly, you will need to alter the Configuration labeled as such below to select the appropriate hyperparameters.
# 
#


from autograd import grad
import autograd.numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

def _robust_loss(psi, beta, nu, Y, Z):
    scaled_sq_errors = np.exp(-2*psi)  * (np.dot(Z, beta) - Y)**2
    if nu == np.inf:
        return scaled_sq_errors/2 + psi
    return (nu + 1)/2 * np.log(1 + scaled_sq_errors / nu) + psi


def make_sgd_robust_loss(Y, Z, nu):
    N = Y.size
    sgd_loss = lambda param, inds: np.mean(_robust_loss(param[0], param[1:], nu, Y[inds], Z[inds])) + np.sum(param**2)/(2*N)
    grad_sgd_loss = grad(sgd_loss)
    return sgd_loss, grad_sgd_loss


def generate_data(N, D, seed):
    rng = np.random.default_rng(seed)
    # generate multivariate t covariates with 10 degrees
    # of freedom and non-diagonal covariance 
    t_dof = 10
    locs = np.arange(D).reshape((D,1))
    cov = (t_dof - 2) / t_dof * np.exp(-(locs - locs.T)**2/4)
    Z = rng.multivariate_normal(np.zeros(D), cov, size=N)
    Z *= np.sqrt(t_dof / rng.chisquare(t_dof, size=(N, 1)))
    # generate responses using regression coefficients beta = (1, 2, ..., D)
    # and t-distributed noise 
    true_beta = np.arange(1, D+1)
    Y = Z.dot(true_beta) + rng.standard_t(t_dof, size=N)
    # for simplicity, center responses 
    Y = Y - np.mean(Y)
    return true_beta, Y, Z


def run_SGD(grad_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N):
    K = (epochs * N) // batchsize
    D = init_param.size
    paramiters = np.zeros((K+1,D))
    paramiters[0] = init_param
    for k in range(K):
        inds = np.random.choice(N, batchsize)
        stepsize = init_stepsize / (k+1)**stepsize_decayrate
        paramiters[k+1] = paramiters[k] - stepsize*grad_loss(paramiters[k], inds)
    return paramiters


def plot_iterates_and_squared_errors(paramiters_1, paramiters_avg, paramiters_3, chart_type, true_beta, opt_param, skip_epochs, epochs, N, batchsize, include_psi=True):
    # initial calculations on paramiters 
    D = true_beta.size
    param_names = [r'$\beta_{{{}}}$'.format(i) for i in range(D)]
    if include_psi:
        param_names = [r'$\psi$'] + param_names
    else:
        paramiters_1 = paramiters_1[:,1:]
        paramiters_avg = paramiters_avg[:,1:]
        paramiters_3 = paramiters_3[:,1:]
        opt_param = opt_param[1:]
    
    # Regression coefficient estimate plots:
    # Type 2: Coefficient estimate plot for Last Iterate Constant
    if chart_type == 2:
        skip_epochs = 0
        skip_iters = int(skip_epochs*N//batchsize)
        xs = np.linspace(skip_epochs, epochs, paramiters_1.shape[0] - skip_iters)
        plt.plot(xs, paramiters_1[skip_iters:]);
        plt.plot(np.array(D*[[xs[0], xs[-1]]]).T, np.array([true_beta,true_beta]), ':')
        plt.xlabel('epoch', fontsize = 14)
        plt.ylabel('parameter value', fontsize = 14)
        plt.legend(param_names, bbox_to_anchor=(0,1.02,1.0,0.2), loc='lower left',
                mode='expand', borderaxespad=0, ncol=4, frameon=False)
        sns.despine()
        plt.suptitle('Coefficient Estimates for Last Iterate Constant: '+ r'$\eta = $' + str(ETA), x=0.5, y=.9 )
        plt.show()

    # Type 3: Coefficient estimate plot for Iterate Average Constant
    if chart_type == 3:
        skip_epochs = 0
        skip_iters = int(skip_epochs*N//batchsize)
        xs = np.linspace(skip_epochs, epochs, paramiters_avg.shape[0] - skip_iters)
        plt.plot(xs, paramiters_avg[skip_iters:]);
        plt.plot(np.array(D*[[xs[0], xs[-1]]]).T, np.array([true_beta,true_beta]), ':')
        plt.xlabel('epoch', fontsize = 14)
        plt.ylabel('parameter value', fontsize = 14)
        plt.legend(param_names, bbox_to_anchor=(0,1.02,1,0.2), loc='lower left',
                mode='expand', borderaxespad=0, ncol=4, frameon=False)
        sns.despine()
        plt.title('Coefficient Estimates for Iterate Average Constant: '+ r'$\eta = $' + str(ETA), x=0.5, y=.95 )
        plt.show()    
    
    # Type 4: Coefficient estimate plot for Last Iterate Decreasing 
    if chart_type == 4:
        skip_epochs = 0
        skip_iters = int(skip_epochs*N//batchsize)
        xs = np.linspace(skip_epochs, epochs, paramiters_3.shape[0] - skip_iters)
        plt.plot(xs, paramiters_3[skip_iters:]);
        plt.plot(np.array(D*[[xs[0], xs[-1]]]).T, np.array([true_beta,true_beta]), ':')
        plt.xlabel('epoch', fontsize = 14)
        plt.ylabel('parameter value', fontsize = 14)
        plt.legend(param_names, bbox_to_anchor=(0,1.02,1,0.2), loc='lower left',
                mode='expand', borderaxespad=0, ncol=4, frameon=False)
        sns.despine()
        plt.title('Coefficient Estimates for Last Iterate Decreasing: ' + r'$\eta_0 = $' + str(ETA_0), x=0.5, y=.95 )
        plt.show()    
    

    # Squared error plots:
    # Type 1: combined plot for all three experiments
    if chart_type == 1:
        skip_epochs = 0
        skip_iters = int(skip_epochs*N//batchsize)
        xs = np.linspace(skip_epochs, epochs, paramiters_1.shape[0] - skip_iters)
        plt.plot(xs, np.linalg.norm(paramiters_1 - opt_param[np.newaxis,:], axis=1)**2, label='Last Iterate Constant')
        plt.plot(xs, np.linalg.norm(paramiters_avg - opt_param[np.newaxis,:], axis=1)**2, '-', label = 'Iterate Average Constant')
        plt.plot(xs, np.linalg.norm(paramiters_3 - opt_param[np.newaxis,:], axis=1)**2, '-', label='Last Iterate Decreasing')
        plt.xlabel('epochs', fontsize= 16)
        plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$', fontsize= 16)
        plt.yscale('log')
        plt.legend()
        sns.despine()
        plt.suptitle('Squared Error For ' +  str(epochs) + ' Epochs of Each Experiment', fontsize=15)
        plt.show()

    # Type 2: plot for Last Iterate Constant only
    elif chart_type == 2:
        plt.plot(xs, np.linalg.norm(paramiters_1 - opt_param[np.newaxis,:], axis=1)**2, label='Last Iterate Constant')
        plt.xlabel('epochs', fontsize= 16)
        plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$', fontsize= 16)
        plt.yscale('log')
        sns.despine()
        plt.suptitle('Squared Error For ' +  str(epochs) + ' Epochs of Last Iterate Constant', fontsize=15)
        plt.show()
        
    # Type 3: plot for Iterate Average constant
    elif chart_type == 3:
        plt.plot(xs, np.linalg.norm(paramiters_avg - opt_param[np.newaxis,:], axis=1)**2, label='Iterate Average Constant')
        plt.xlabel('epochs', fontsize= 16)
        plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$', fontsize= 16)
        plt.yscale('log')
        sns.despine()
        plt.suptitle('Squared Error For ' +  str(epochs) + ' Epochs of Iterate Average Constant', fontsize=15)
        plt.show()

    # Type 4: plot for Last Iterate Decreasing only
    elif chart_type == 4:
        plt.plot(xs, np.linalg.norm(paramiters_3 - opt_param[np.newaxis,:], axis = 1)**2, label='Last Iterate Decreasing')
        plt.xlabel('epochs', fontsize= 14)
        plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$', fontsize= 14)
        plt.yscale('log')
        sns.despine()
        plt.suptitle('Squared Error For ' +  str(epochs) +  ' Epochs of Last Iterate Decreasing', fontsize=15)
        plt.show()
    
       
    
#----------------------------------
# Configuration
#----------------------------------

seed = 5938 # set seed
ETA = 0.2 # stepsize
ETA_0 = 5 # it avg standard initial stepsize
epochs = 20 # number of epochs
N = 10000 # number of observations
D = 10 # number of dimensions
B = 10 # batch size 
sc_f = 0 # Scaling Factor for initialization distance (0 otherwise)
X_0 = sc_f * np.ones(D+1) # initialization value
NU = 5 # fixed for all experiments
ALPHA_0 = 0 # for Last Iterate Constant and Iterate Average Constant
ALPHA = 0.51 # for Last Iterate Decreasing 
a, b, c = generate_data(N, D, seed)
d, e = make_sgd_robust_loss(b, c, NU)
opt_param = sp.optimize.minimize(d, X_0, args = (np.arange(N),),).x




#-----------------------
# Last Iterate Constant
#-----------------------
p_1 = run_SGD(e, epochs, X_0, ETA, ALPHA_0, B, N)

#-------------------------
# Iterate Average Constant
#-------------------------
p_avg = np.array(
        [np.mean(p_1[k//2:k+1], axis=0) for k in range(p_1.shape[0])]
   )

#-------------------------
# Last Iterate Decreasing
#-------------------------
p_3 = run_SGD(e, epochs, X_0, ETA_0, ALPHA, B, N)




def main(chart_type):
    if chart_type == 1:    
        plot_iterates_and_squared_errors(p_1, p_avg, p_3, 1, a, opt_param, 0, epochs, N, B, True)
    elif chart_type == 2:
        plot_iterates_and_squared_errors(p_1, None, None, 2, a, opt_param, 0, epochs, N, B, True)
    elif chart_type == 3:
        plot_iterates_and_squared_errors(None, p_avg, None, 3, a, opt_param, 0, epochs, N, B, True)
    elif chart_type == 4:
        plot_iterates_and_squared_errors(None, None, p_3, 4, a, opt_param, 0, epochs, N, B, True)
        
main(4)









