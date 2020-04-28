import pandas as pd
import numpy as np
from scipy.optimize import minimize


def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio
    """
    return weights.T @ returns


def portfolio_vol(weights, cov):
    """
    Computes the volatility of a portfolio
    """
    return (weights.T @ cov @ weights)**0.5


def minimum_vol_weights(ret, er, cov):
    """
    Returns the portfolio weights which lead to the minimum volatility given a specific portfolio return
    """
    n = er.shape[0]
    x0 = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n

    # Constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: ret - portfolio_return(weights, er)}

    # Optimize
    weights = minimize(fun=portfolio_vol,
                       x0=x0,
                       args=(cov,),
                       method='SLSQP',
                       bounds=bounds,
                       constraints=(weights_sum_to_1, return_is_target),
                       options={'disp': False})

    return weights.x


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the Maximum Sharpe Ratio portfolio
    """
    n = er.shape[0]
    x0 = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n

    def negative_sharpe_ratio(weights, riskfree_rate, er, cov):
        return -(portfolio_return(weights, er)-riskfree_rate)/portfolio_vol(weights, cov)

    # Constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}

    # Optimize
    weights = minimize(fun=negative_sharpe_ratio,
                       x0=x0,
                       args=(riskfree_rate, er, cov),
                       method='SLSQP',
                       bounds=bounds,
                       constraints=(weights_sum_to_1,),
                       options={'disp': False})

    return weights.x


def gmv(cov):
    """
    Returns the weights of the Global Minimum Variance portfolio
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def efficient_frontier(er, cov, n_points=100, riskfree_rate=0, show_cml=True, show_ew=True, show_gmv=True, style='.-'):
    """
    Plots the efficient frontier
    """
    rets = np.linspace(er.min(), er.max(), n_points)
    weights = [minimum_vol_weights(r, er, cov) for r in rets]
    #rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]

    ef = pd.DataFrame({"Returns": rets,
                       "Volatility": vols})
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if show_cml:
        # MSR (or TAN) portfolio
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)

        ax.plot([0, vol_msr], [riskfree_rate, r_msr], color='red',
                marker='o', markersize=10, linestyle='dashed')

    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)

        ax.plot([vol_ew], [r_ew], color='green', marker='o', markersize=10)

    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)

        ax.plot([vol_gmv], [r_gmv], color='blue', marker='o', markersize=10)
