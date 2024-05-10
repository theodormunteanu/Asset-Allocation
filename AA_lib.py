# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:00:19 2024

@author: Theodor Munteanu
"""

import numpy as np

import scipy.optimize as opt

import sys
import bond_price as bp

import pandas as pd
#%%

def sigma_problem(rets,vols,corr_matrix,sig_target,bnds = None,ret_phi = False):
    """
    Functionality:
        Given a list/np.ndarray of expected returns, I find the portfolio that maximizes 
        the return given a target volatility. (sig_target)
    
    Parameters:
        rets: Vector of expected returns 
        
        vols: volatilities 
        
        corr_matrix: correlation matrix of returns
        
        sig_target: target volatility 
        
        bnds: bounds within which the optimal solution must be located. 
    """
    cov_mat = np.array(vols,ndmin = 2).T*np.array(vols,ndmin = 2)*corr_matrix
    n = len(vols)
    EW = np.ones((1,n))/n
    ret_func = lambda x: np.dot(rets,x)
    vol_func = lambda x: np.sqrt(np.dot(x,np.dot(cov_mat,x.T)))
    cons = ({'type':'eq','fun':lambda x:sum(x)-1},
            {'type':'ineq','fun':lambda x:sig_target-vol_func(x)})
    sig_sol0 = opt.minimize(lambda x:-ret_func(x),EW,constraints = cons,bounds = bnds)
    
    return sig_sol0

def MV_problem(rets,vols,corr_matrix,bnds = None):
    """
    Returns:
        1. structure of the portfolio
        
        2. Return of the portfolio
        
        3. Volatility of the portfolio
    """
    cov_mat = np.array(vols,ndmin = 2).T*np.array(vols,ndmin = 2)*corr_matrix
    n = len(vols)
    EW = np.ones((1,n))/n
    ret_func = lambda x:np.dot(rets,x)
    vol_func = lambda x: np.sqrt(np.dot(x,np.dot(cov_mat,x.T)))
    cons = ({'type':'eq','fun':lambda x:sum(x)-1})
    sol = opt.minimize(lambda x:vol_func(x),EW,constraints = cons,bounds = bnds)
    return sol.x,np.dot(rets,sol.x),vol_func(sol.x)

def MV_problem2(cov_mat,bnds = None,constr = None,tol = 1e-8):
    """
    Given covariance matrix (input) and bounds (optional), the function returns the portfolio structure.
    """
    n = np.shape(cov_mat)[0]
    EW = np.ones((1,n))[0]/n
    vol_func = lambda x: np.sqrt(np.dot(x,np.dot(cov_mat,x.T)))
    if constr==None:
        cons = ({'type':'eq','fun':lambda x:sum(x)-1},)
    else:
        cons = constr+({'type':'eq','fun':lambda x:sum(x)-1},)
    sol = opt.minimize(lambda x:vol_func(x),EW,constraints = cons,bounds = bnds,tol = tol)
    return sol.x

def mu_problem(rets,vols,corr_matrix,mu_target,bnds = None,ret_phi = False):
    cov_mat = np.array(vols,ndmin = 2).T*np.array(vols,ndmin = 2)*corr_matrix
    n = len(vols)
    EW = np.ones((1,n))/n
    ret_func = lambda x: np.dot(rets,x)
    vol_func = lambda x: np.sqrt(np.dot(x,np.dot(cov_mat,x.T)))
    cons = ({'type':'eq','fun':lambda x:sum(x)-1},
            {'type':'ineq','fun':lambda x:ret_func(x)-mu_target})
    mu_sol0 = opt.minimize(lambda x:vol_func(x),EW,constraints = cons,bounds = bnds)
    if ret_phi == False:
        return mu_sol0
    else:
        phi = np.dot(rets,np.dot(np.linalg.inv(cov_mat),np.array(rets).T))/mu_target
        return mu_sol0, phi


def gamma_problem(rets,vols,corr_matrix,gamma,rf = 0.0,bnds = None):
    """
    Parameters:
        rets: list/numpy array. Means expected returns of the assets
        
        corr_matrix: correlation matrix
        
        vols: volatilities
    
    Returns:
        1. Structure of the portfolio
        2. Volatility of the portfolio
        3. Expected return
        4. Sharpe ratio
    """
    cov_mat = np.array(vols,ndmin = 2).T*np.array(vols,ndmin = 2)*corr_matrix
    n = len(vols)
    EW = np.ones((1,n))/n
    cons = ({'type':'eq','fun':lambda x:sum(x)-1})
    ret_func = lambda x: np.dot(rets,x)
    vol_func = lambda x: 1/2*np.dot(x,np.dot(cov_mat,x.T))
    sharpe_func = lambda x:np.dot(x,np.array(rets)-rf)
    gamma_func = lambda x: vol_func(x)-gamma*ret_func(x)
    sol = opt.minimize(gamma_func,EW[0],constraints = cons)
    return sol.x,np.sqrt(vol_func(sol.x)*2),ret_func(sol.x),sharpe_func(sol.x)



def beta_func(cov_mat,x,y):
    """
    beta function between a portfolio having structure x and another having structure y.
    
    Parameters:
        cov_mat: covariance matrix of returns
        
        x: structure of the first portfolio (vector)
        
        y: structure of the second portfolio (vector)
        

    """
    return np.dot(y,np.dot(cov_mat,x))/np.dot(x,np.dot(cov_mat,x))

def risk_contribs(cov_mat,x,adjusted = False,details = 'no'):
    """
    Risk contributions of the standard deviation measure 
    
    Parameters:
        cov_mat: 2D numpy array
        
    Returns:
        RC if details = no and adjusted = False
        
        marginal_risks,RCs, RC/sum(RC) if details = Yes
    """
    std_dev_func = lambda x: np.sqrt(np.dot(x,np.dot(cov_mat,x.T)))
    RC = x*np.dot(cov_mat,x.T)/std_dev_func(x)
    marginal_risks = np.dot(cov_mat,x.T)/std_dev_func(x)
    if adjusted==False:
        return RC if details == 'no' else marginal_risks,RC,RC/sum(RC)
    else:
        return RC/sum(RC) if details == 'no' else marginal_risks,RC,RC/sum(RC)


def perf_contribs(cov_mat,x,rets,r=0.0):
    """
    Performance contributions of a portfolio (x)
    
    Parameters:
        cov_mat: covariance matrix of returns
        
        x: structure of portfolio
        
        rets: vector of expected returns 
    """
    std_dev_func = lambda x: np.sqrt(np.dot(x,np.dot(cov_mat,x.T)))
    sharpe_func = lambda x: (np.dot(rets,x)-r)/std_dev_func(x)
    RCs = x*np.dot(cov_mat,x.T)/std_dev_func(x)
    PCs = sharpe_func(x)*RCs
    return PCs

def ERC_portf(cov_mat,details = 'no',sigs = 0,returns = 0):
    """
    If sigs are not provided, cov_mat is a covariance matrix
    
    If sigs are provided, cov_mat is a correlation matrix.
    
    Parameters:
        cov_mat: covariance matrix
    RETURNS:
        
        Structure of the portfolio (res)
        
        + Optionally (RC_func(res.))
    """
    n = np.shape(cov_mat)[0]
    EW = np.ones((1,n))[0]/n
    if sigs == 0:
        cov_mat2 = cov_mat
    else:
        cov_mat2 = np.array(sigs,ndmin = 2).T*np.array(sigs,ndmin = 2)*cov_mat
        
    RC_func = lambda x:risk_contribs(cov_mat2,x)[1]/sum(risk_contribs(cov_mat2,x)[1])
    
    func = lambda x: sum((RC_func(x)-EW)**2)
    cons = ({'type':'eq','fun':lambda x:sum(x)-1})
    res = opt.minimize(func,EW,constraints = cons)
    
    if details=='no':
        return res.x
    else:
        sig = np.sqrt(np.dot(res.x,np.dot(cov_mat,(res.x).T)))
        return res.x,RC_func(res.x),sig
    
def RB_portf(cov_mat,RBs):
    """
    Construction of a risk budgeted portfolio.
    
    Parameters:
        cov_mat: covariance matrix (of returns usually)
        
        RBs: risk budgets
    
    
    """
    RC_func = lambda x:risk_contribs(cov_mat,x)/sum(risk_contribs(cov_mat,x))
    func = lambda x: sum((RC_func(x)-RBs)**2)
    cons = ({'type':'eq','fun':lambda x:sum(x)-1})
    n = len(RBs)
    EW = np.ones((1,n))[0]/n
    res = opt.minimize(func,EW,constraints = cons)
    return res.x

def diversification_ratio(cov_mat,x,ret_vol = False):
    """
    Compute the DR of a portfolio x: 
    """
    if abs(sum(x) - 1.0)>0.01:
        raise ValueError('{0} must be a portfolio and must add up to 1'.format(x))
    sigs = np.sqrt(np.diag(cov_mat))
    vol = np.sqrt(np.dot(x,np.dot(cov_mat,x.T)))
    if ret_vol==False:
        return vol/np.dot(x,sigs)
    else:
        return vol/np.dot(x,sigs),vol

def MDP(cov_mat,bounds = None,sigs = None,rets = None,details = 'no'):
    """
    Maximum diversification portfolio.
    """
    cons = ({'type':'eq','fun':lambda x:sum(x)-1})
    
    n = np.shape(cov_mat)[0]
    
    EW = np.ones((1,n))[0]/n
    
    if sigs==None:
        cov_mat2 = cov_mat
    else:
        cov_mat2 = np.array(sigs,ndmin = 2).T*np.array(sigs,ndmin = 2)*cov_mat
    
    func = lambda x:diversification_ratio(cov_mat2,x)

    if bounds==None:
        result = opt.minimize(func,EW,constraints = cons)
    else:
        result = opt.minimize(func,EW,constraints = cons,bounds = bounds)
    if details == 'no':
        return result.x
    else:
        func2 = lambda x: np.sqrt(np.dot(x,np.dot(cov_mat,x.T)))
        return dict(zip(['Result','Volatility','Expected returns'],[result.x,func2(result.x),np.dot(rets,result.x)]))

def vol_func_TE(cov_mat,bench,x):
    """
    Tracking error volatility 
    """
    return np.sqrt(np.dot(x-bench,np.dot(cov_mat,(x-bench).T)))

def portf_decarb(cov_mat,bench,CIs,prc1 = 0.0,prc2 = 0.0,CMs = None,t = 0):
    """
    Given a portfolio of equities find the minimum variance portfolio when decarb 
    constraints are given. 
    
    Parameters:
        cov_mat: covariance matrix
        bench: array (benchmark)
        CIs: carbon intensities for each asset/sector.
        CMs:(Optional) Carbon Momentums for each asset/sector.
        prc1: constant percent reduction (usually 30% or 50%)
        prc2: reduction rate with time. (usually 7%)
        
    Returns:
        1. Structure of the portfolio
        2. Tracking Error vol
        3. Carbon intensity of the result
        4. Carbon reduction
        5. (optional) Carbon Momentum
    """
    n = len(bench)
    CI_bench = np.dot(bench,CIs)
    CI_func = lambda x:np.dot(CIs,x)
    x0 = np.ones((1,n))[0]/n
    cons = ({'type':'eq','fun':lambda x:sum(x)-1},
            {'type':'ineq','fun':lambda x:CI_bench*(1-prc1)*(1-prc2)**t - CI_func(x)})
    res = opt.minimize(lambda x:vol_func_TE(cov_mat,bench,x),x0,constraints = cons)
    TE_vol = vol_func_TE(cov_mat,bench,res.x)
    CI = np.dot(CIs,res.x)
    carb_red = 1-CI/CI_bench
    CM = np.dot(CMs,res.x)
    return res.x,TE_vol,CI,carb_red,CM

def cov_mat_bonds(cov_mat_ylds,bond_durs):
    """
    Find the covariance matrix between bond returns given yield data
    and modified durations.
    
    
    """
    cov_matrix_bonds = np.array(bond_durs,ndmin = 2).T*np.array(bond_durs,
                                                ndmin = 2)*cov_mat_ylds
    return cov_matrix_bonds

def MV_problem_bonds(cov_mat_ylds,bond_durs,bnds = None,details = 'no'):
    cov_mat_bonds = np.array(bond_durs,ndmin = 2).T*np.array(bond_durs,ndmin = 2)*cov_mat_ylds
    risk_func = lambda x: np.sqrt(np.dot(x,np.dot(cov_mat_bonds,x.T)))
    n = len(bond_durs)
    init_sol = np.ones((1,n))[0]/n
    cons = ({'type':'eq','fun':lambda x:sum(x)-1})
    if bnds == None:
        bnds = [(0,1)]*n
        sol = opt.minimize(risk_func,init_sol,bounds = bnds,constraints = cons)
    if details =='no':
        return sol.x
    else:
        return dict(zip(['Structure','Risk'],[sol.x, risk_func(sol.x)]))

def MV_problem_bonds2(coupons,Ts,freqs,ylds,cov_mat_ylds,bnds = None,details = 'no'):
    """
    coupons: interest rate of bonds 
    
    freqs: frequencies of bonds. 
    """
    bond_vals = [bp.bond_price_yield(1,coupons[i],Ts[i],ylds[i],freqs[i],details = 'yes')
                 for i in range(len(freqs))]
    df_bond_vals = pd.DataFrame(bond_vals,columns = ['Price','Duration','Convexity'])
    bond_durs = df_bond_vals['Duration']
    return MV_problem_bonds(cov_mat_ylds,bond_durs,bnds = bnds,details = details)

def MSR_bonds(cov_mat_ylds,bond_durs,chg_ylds,bnds = None,details = 'no'):
    """
    Portfolio of maximum sharpe ratio for bonds.
    
    Parameters:
        cov_mat_ylds: covariance matrix of yields
            
        bond_durs: bond durations
        
        chg_ylds: change of yields. 
        
        bnds: bounds for the structure of the portfolio.
    """
    n = len(bond_durs)
    cov_mat_bonds = np.array(bond_durs,ndmin = 2).T*np.array(bond_durs,ndmin = 2)*cov_mat_ylds
    risk_func = lambda x: np.sqrt(np.dot(x,np.dot(cov_mat_bonds,x.T)))
    exp_rets = -np.array(bond_durs)*chg_ylds
    ret_func = lambda x: np.dot(x,exp_rets)
    sharpe_func = lambda x:ret_func(x)/risk_func(x)
    init_sol = np.ones((1,n))[0]/n
    cons = ({'type':'eq','fun':lambda x:sum(x)-1})
    if bnds == None:
        bnds = [(0,1)]*n
    sol = opt.minimize(lambda x:-sharpe_func(x),init_sol,bounds = bnds,constraints = cons)
    if details == 'no':
        return sol.x
    else:
        return dict(zip(['Struct','Sharpe','Risk','Return'],
            [sol.x,sharpe_func(sol.x),risk_func(sol.x),ret_func(sol.x)]))

def risk_contribs_bonds(FVs,n_bonds,coupons,Ts,freqs,ylds,cov_ylds):
    """
    Parameters:
        FVs: face values
        ylds: bond yields
        cov_ylds: covariance matrix between yields.
        Ts: expiries of bonds
        
    Returns:
        risk contributions of bonds. 
    """
    if len(FVs)!=len(coupons) or len(FVs)!=len(freqs):
        raise ValueError("The frequencies, coupons and face values must have the same length")
    bond_vals = [bp.bond_price_yield(FVs[i],coupons[i],Ts[i],ylds[i],freqs[i],details = 'yes')
                 for i in range(len(freqs))]
    df_bond_vals = pd.DataFrame(bond_vals,columns = ['Price','Duration','Convexity'])
    bond_durs = df_bond_vals['Duration']
    bond_prices = df_bond_vals['Price']
    cov_mat_bonds = np.array(bond_durs,ndmin = 2).T*np.array(bond_durs,ndmin = 2)*cov_ylds
    risk_func = lambda x: np.sqrt(np.dot(x,np.dot(cov_mat_bonds,x.T)))
    w = np.array(n_bonds)*bond_prices/np.dot(n_bonds,bond_prices)
    risk_contribs = w*np.dot(cov_mat_bonds,w.T)/risk_func(w)
    return risk_contribs

def max_sharpe_ratio(cov_mat,returns,bnds = None):
    """
    Parameters:
        cov_mat: covariance matrix of the rreturns 
    
        returns: expected returns 
        
    Returns:
        Portfolio structure with the maximum sharpe ratio.
    
    """
    sig_func = lambda x:np.sqrt(np.dot(x,np.dot(cov_mat,x.T)))
    ret_func = lambda x: np.dot(returns,x)
    sharpe_func = lambda x: ret_func(x)/sig_func(x)
    n = len(returns)
    init_sol = np.array([1/n]*n)
    if bnds == None:
        bnds = [(0,1)]*n
    cons = ({'type':'eq','fun':lambda x:sum(x)-1},)
    sol = opt.minimize(lambda x:-sharpe_func(x),init_sol,bounds = bnds,constraints = cons)
    return sol.x,sharpe_func(sol.x)


def risk_contribs_bonds2(FVs,capital,w,coupons,Ts,freqs,ylds,cov_ylds):
    
    bond_vals = [bp.bond_price_yield(FVs[i],coupons[i],Ts[i],ylds[i],freqs[i],details = 'yes')
                 for i in range(len(freqs))]
    df_bond_vals = pd.DataFrame(bond_vals,columns = ['Price','Duration','Convexity'])
    bond_prices = df_bond_vals['Price']
    n_bonds = capital*w/bond_prices
    return risk_contribs_bonds(FVs,n_bonds,coupons,Ts,freqs,ylds,cov_ylds)

def PCA(cov_mat):
    eigvals,eigvecs = np.linalg.eig(cov_mat)
    sorted_eigvals = sorted(eigvals,reverse = True)
    pos = [eigvals.index(x) for x in sorted_eigvals]
    return eigvecs[:,pos]
    
def cov_mat_bonds2(cov_mat_ylds,coupons,Ts,freqs,ylds):
    """
    Inputs:
        cov_mat_ylds: covariance matrix of yield changes (ndarray)
        
        ylds: last available yields.
        
        coupons: list of coupons
        
        Ts: expiries
        
    Functionality:
        Returns the covariance matrix of the bond returns 
    """
    bond_vals = [bp.bond_price_yield(1,coupons[i],Ts[i],ylds[i],freqs[i],details = 'yes')
                 for i in range(len(freqs))]
    df_bond_vals = pd.DataFrame(bond_vals,columns = ['Price','Duration','Convexity'])
    bond_durs = df_bond_vals['Duration']
    cov_matrix_bonds = np.array(bond_durs,ndmin = 2).T*np.array(bond_durs,ndmin = 2)*cov_mat_ylds
    return cov_matrix_bonds

def EW_bonds(cov_mat_ylds,bond_durs,chg_ylds):
    """
    Find the risk, return and sharpe ratio of Equally Weighted bonds. 
    """
    n = len(bond_durs)
    cov_mat_bonds = np.array(bond_durs,ndmin = 2).T*np.array(bond_durs,ndmin = 2)*cov_mat_ylds
    risk_func = lambda x: np.sqrt(np.dot(x,np.dot(cov_mat_bonds,x.T)))
    exp_rets = -np.array(bond_durs)*chg_ylds
    ret_func = lambda x: np.dot(x,exp_rets)
    sharpe_func = lambda x:ret_func(x)/risk_func(x)
    EW_str = np.ones((1,n))[0]/n
    return dict(zip(['Risk','Return','Sharpe'],[risk_func(EW_str),
                                                ret_func(EW_str),
                                                sharpe_func(EW_str)]))

def MSR_bonds2(coupons,Ts,freqs,ylds,cov_ylds,chg_ylds,details = 'no'):
    """
    Given coupons, expiries, frequencies and yields 
    
    cov_ylds: covariance matrix of changes in yields (ndarray)
    
    chg_ylds: changes in yields (list or numpy.array)
    
    Find the Maximum Sharpe ratio portfolio.
    """
    bond_vals = [bp.bond_price_yield(1,coupons[i],Ts[i],ylds[i],freqs[i],details = 'yes')
                 for i in range(len(freqs))]
    df_bond_vals = pd.DataFrame(bond_vals,columns = ['Price','Duration','Convexity'])
    bond_durs = df_bond_vals['Duration']
    return MSR_bonds(cov_ylds,bond_durs,chg_ylds,details = details)
