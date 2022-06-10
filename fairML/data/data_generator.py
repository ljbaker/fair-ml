import numpy as np
import pandas as pd

from itertools import combinations
from typing import Tuple
from scipy.stats import pearsonr
from scipy.stats import percentileofscore



np.random.seed(8675309)
mu_hat_x = 0
mu_hat_z = 10
sigma_hat_x = 1
sigma_hat_z = 10

def generate_corr_data_univariate(rho: float,
                                  N: int,
                                  seed: int=None,
                                  verbose: bool=True,
                                  reshape_dict: dict=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for simulating univariate correlated data. Given an approximate
    correlation coefficient, rho, simulate data of size N for two random Normal
    variables. Can also include dictionary for reshaping output data.

    Parameters
    ----------
    rho : float, range(-1,1)
        value of target approximate correlation for output vectors
        will have normal variance as a function of N
    N : int, range(2,inf)
        length of output
    seed : int
        optional argument to set random seed
    verbose : bool
        argument for printing steps
    reshape_dict : dict[**vargs]
        dictionary to reshape output with the following keys:
        'mu_hat_x' : output mean of X
        'sigma_hat_x' : output standard deviation of X
        'mu_hat_z' : output mean of z
        'sigma_hat_z' : output standard deviation of z

    Returns
    -------
    X, z : np.array
        numpy arrays of size N, approximately correlated at rho

    """
    if seed:
        np.random.seed(seed)

    X = np.random.normal(0, 1, size=N)
    y = np.random.normal(0, 1, size=N)

    a = rho/np.sqrt(1-rho**2)
    z = a*X+y

    if verbose:
        print(f'generating data with approxmate rho={rho}\n')
        print(f'seeded X mean: {X.mean(): .3f}, sd: {X.std(): .3f}')
        print(f'seeded y mean: {y.mean(): .3f}, sd: {y.std(): .3f}')
        print(f'simulated y (z) mean: {z.mean(): .3f}, sd: {z.std(): .3f}\n')

    # optional, if input moments differ from desired output moments
    if reshape_dict:
        rd = reshape_dict
        X = X*rd['sigma_hat_x']+rd['mu_hat_x']
        z = z*rd['sigma_hat_z']+rd['mu_hat_z']

    if verbose:
        print(f'adjusted X mean: {X.mean(): .3f}, sd: {X.std(): .3f}')
        print(f'adjusted simulated y (z) mean: {z.mean(): .3f}, sd: {z.std(): .3f}\n')
        print(f'correlation of z~X: {pearsonr(X, z)[0]:.3f}')

    return z, X


def generate_corr_data_multivariate(rmat: np.ndarray,
                                    N: int,
                                    seed: int=None,
                                    verbose: bool=True,
                                    colnames: list=None,
                                    reshape_mu: np.array=None,
                                    reshape_sigma: np.array=None,
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Function for simulating univariate correlated data. Given an approximate
    correlation coefficient, rho, simulate data of size N for two random Normal
    variables. Can also include dictionary for reshaping output data.

    Parameters
    ----------
    rho : float, range(-1,1)
        value of target approximate correlation for output vectors
        will have normal variance as a function of N
    N : int, range(2,inf)
        length of output
    seed : int
        optional argument to set random seed
    verbose : bool
        argument for printing steps
    reshape_mu : np.array
        array to reshape output means
    reshape_sigma : np.array
        array to reshape output standard deviations

    Returns
    -------
    output : pandas.DataFrame
        df shape (N,len(rmat)), with columns approximately correlated at rho
    """
    if seed:
        np.random.seed(seed)

    mu = np.zeros(len(rmat))

    mat = np.random.multivariate_normal(mu, cov=rmat,size=N)

    # optional, if input moments differ from desired output moments
    if reshape_sigma:
        mat = mat * reshape_sigma
    if reshape_mu:
        mat = mat + reshape_mu

    output = pd.DataFrame(mat, columns=colnames)

    if verbose:
        print(f'target rho \n{rmat}')
        print(f'achieved rho \n{pd.DataFrame(mat).corr()}')

    return output


def make_cov_array(n, var_range=(0,1)):
    """creates an array of correlation coefficients bound by var_range"""
    return np.random.uniform(var_range[0], var_range[1], len(list(combinations(range(n),2))))


def make_cov_matrix(n, cov_array):
    """
    generate a symetric covariance matrix given an array of
    correlations between columns, in combinatoric order of
    $_nC_2$

    e.g., three element matrix
    cov_array = cor(A,B), cor(A,C), cor(B,C)

    e.g., four element matrix
    cov_array = cor(A,B), cor(A,C), cor(A,D), cor(B,C), cor(B,D), cor(C,D)
    """
    M = np.ones([n,n])
    M[np.triu_indices(M.shape[0], k = 1)] = cov_array
    M.T[np.triu_indices(M.shape[0], k = 1)] = cov_array
    return M


def make_random_cov_matrix(n, var_range):
    """creates a random correlation matrix bound by var_range"""
    return make_cov_matrix(n, make_cov_array(n, var_range))


def generate_group_data(n_variables,
                        groups,
                        var_cov,
                        group_ES=None,
                        es_type='pct',
                        target_name='y',
                        size=10000,
                        random_seed=None):
        """
        Master function for producing randomly generated datasets with groupwise differences
        for use in testing various fairness systems.

        Note that this function will return values of only approximate group differences
        and covariance.

        Parameters
        ----------
        n_varibles : int
            number of independent variables to generate
            creates one vector for each IV
        groups : list(str)
            list of group names to divide data
            if only one group is provided, generates normally distributed data
        var_cov : np.ndarray or str
            covariance matrix for the desired multivariate normal distribution
            alternatively, can input string commans to generate covariances within range
            'low' : correlations between (0,.2)
            'med' : correlations between (.2,.6)
            'high' : correlations between (.6,1)
        group_ES : list
            effect size metrics for each group, relative to population mean
            see es_type for a description of values and expected behavior
        es_type : str
            type of effect size metrics entered in group_ES
            'pct' and 'prop' are most useful for segmenting groups for classification
            'd' is best used when segmenting groups on continuous metrics

            'pct' : percentile pass rates of each group (0, 100)
            'prop': proportional pass rates of each group (0,1)
            'd'   : effect size, d, difference between group and population mean
        target_name : str
            name for dependent variable, aka predictor
            defaults to 'y' by convention
        size : int
            number of rows of data to generate
        random_seed : int
            seed for random state

        Returns
        -------
        generated_df : pandas.DataFrame
            dataframe of generated data with covariance, var_cov,
            and group means segmented by group_ES
        """
        var_names = ['x'+str(i) for i in range(1,n_variables+1)]
        colnames = [target_name] + var_names

        if random_seed:
            np.random.seed(random_seed)

        mu = np.zeros(len(colnames))

        if isinstance(var_cov, str):
            #generate covariance matrix
            print('generating random covariance matrix\n')
            if var_cov == 'low':
                var_cov = make_random_cov_matrix(n=len(colnames), var_range=(0,.2))
            elif var_cov == 'med':
                var_cov = make_random_cov_matrix(n=len(colnames), var_range=(.2,.6))
            elif var_cov == 'high':
                var_cov = make_random_cov_matrix(n=len(colnames), var_range=(.6,1))

        mat = np.random.multivariate_normal(mu, cov=var_cov,size=size)

        output = pd.DataFrame(mat, columns=colnames)
        output['group'] = None

        batch_size = np.ceil(len(output)/len(groups))

        for b, chunk in output.groupby(np.arange(len(output)) // batch_size):
            b = int(b) #really weird numpy behavior leads to "-0"
            output.loc[chunk.index, 'group'] = groups[b]
            #calculate noncentrality parameter from type of effect size
            if es_type=='prop':
                # calculate from proprortional difference from 0
                prop = group_ES[b]
                # restrict to (0,100)
                pct = max(max(50 + prop*100, 100), 0)
            elif es_type == 'pct':
                # calculate from percentile
                pct = group_ES[b]
            else:
                # calculate from effect size d
                d = group_ES[b]
                # empirical percentile of score
                pct = percentileofscore(chunk.y, chunk.y.mean()+d)
            pct = max(min(pct, 100), 0) #restrict percentile range
            ncp = np.percentile(chunk.y, pct) - chunk.y.mean()

            output.loc[chunk.index, 'y'] = chunk['y'] + ncp

        output[target_name+'_b'] = np.array(output[target_name] > 0, dtype=int)
        output = output.join(pd.get_dummies(output.group,prefix='group', drop_first=False))
        return output
