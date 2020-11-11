import pandas as pd
import matplotlib.pyplot as plt
from asreview.state.utils import open_state
from scipy.stats import spearmanr


def probability_matrix_from_h5_state(state_fp):
    """Get the probability matrix from an .h5 state file.

    Arguments
    ----------
    state_fp: str
        Path to state file.

    Returns
    -------
    pandas.DataFrame:
        A dataframe of shape (num_papers, num_queries), with in (i,j) the probability
        that paper i was relevant according to the model at query j. Note that the row
        index starts at 0, but the column index starts at 1.
    """
    proba_dict = {}
    with open_state(state_fp, read_only=True) as state:
        queries = [int(num) for num in state.f['results'].keys()]
        total_queries = max(queries)

        for i in range(1, total_queries+1):
            proba_dict[i] = state.f[f'results/{i}/proba'][:]

    proba_matrix = pd.DataFrame.from_dict(proba_dict)
    return proba_matrix


def probability_plot(proba_matrix, row_nums, gap=1, **kwargs):
    """Plot the probability of a document being relevant, at different iterations of
    the model.

    Arguments
    ----------
    proba_matrix: pandas.DataFrame
        Probability matrix.
    row_nums: list
        List containing the row numbers you want to plot.
    gap: int
        Step size on the x-axis (the queries).

    Returns
    -------
    Plot of the probability of the documents in 'num_rows' of the probability matrix at
    query_{gap*i).
    """
    plt.plot(proba_matrix.iloc[row_nums, ::gap].T, **kwargs)
    plt.show()


def rank_plot(proba_matrix, row_nums, gap, **kwargs):
    """Plot the rank of the document, at different iterations of the model.

    Arguments
    ----------
    proba_matrix: pandas.DataFrame
        Probability matrix.
    row_nums: list
        List containing the row numbers you want to plot.
    gap: int
        Step size on the x-axis (the queries).

    Returns
    -------
    Plot of the rank of the documents in 'num_rows' of the probability matrix at
    query_{gap*i).
    """
    rank_matrix = proba_matrix.rank()
    plt.plot(rank_matrix.iloc[row_nums, ::gap].T, **kwargs)
    plt.show()


def rho_plot(proba_matrix, gap=1, **kwargs):
    """Plot the value of Spearman's rho, comparing different iterations of the active
    learning model.

    Arguments
    ----------
    proba_matrix: pandas.DataFrame
        Probability matrix.
    gap: int
        Calculate the rho of query_i and query_(i+gap), for i in
        range(gap, num_rows, gap).

    Returns
    -------
    Plot of value of rho. Note that per step on the x-axis, we take a step of 'gap'
    queries.
    """
    rho_list = [spearmanr(proba_matrix[i], proba_matrix[i-gap])
                for i in range(gap + 1, proba_matrix.shape[1], gap)]
    plt.plot(rho_list, **kwargs)
    plt.show()
