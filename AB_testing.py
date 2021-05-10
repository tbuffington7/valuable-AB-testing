import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from copy import deepcopy


class Advertisement:
    """
    Class for holding relevant attributes and functions for an advertisement to be A/B tested.

    Attributes:
    ----------
    alpha: float
        The \alpha parameter in the advertisement's beta distribution for conversion rate
    beta: float
        The \beta parameter in the advertisement's beta distribution for conversion rate
    value_function: func
        A function that takes in a conversion rate as an input and outputs a monetary value
    expected_value:
        The expected revenue of the advertisement computed by averaging over all of its potential conversion rates
    """

    def __init__(self, alpha, beta, conversion_rate_value):
        """
        Initializes the Advertisement class

        Parameters:
        ----------
        alpha: float
            The \alpha parameter in the advertisement's beta distribution for conversion rate
        beta: float
            The \beta parameter in the advertisement's beta distribution for conversion rate
        conversion_rate_value: float:
            The expected value (in thousands of dollars per %) of an ad based on its conversion rate
        """

        self.alpha = alpha
        self.beta = beta
        self.conversion_rate_value = conversion_rate_value
        self.expected_value = None

        # This updates self.expected_value
        self.calc_expected_value()

    def update_beliefs(self, test_sample_size, num_conversions):
        """
        Given A/B test results, this conducts Bayesian updating of the beta distribution
        defined over the conversion rate.

        Parameters
        ----------
        test_sample_size: int
            The number of ad interactions
        num_conversions: int
            The number of ad conversions
        """

        self.alpha += num_conversions
        self.beta += (test_sample_size - num_conversions)

        # Recalculate the expected revenue with the updated beliefs
        self.calc_expected_value()

    def get_test_data(self, test_sample_size):
        """
        Simulates the result of an A/B test of a given size. It draws a conversion rate from the beta distribution
        (our current beliefs about the underlying conversion rate). Given this conversion rate, it draws a number of
        conversions from a binomial distribution.

        Parameters
        ----------
        test_sample_size: int
            The number of ad interactions

        Returns
        -------
        num_conversions: int
            The simulated number of ad conversions

        """

        drawn_conversion_rate = np.random.beta(a=self.alpha, b=self.beta)
        num_conversions = np.random.binomial(n=test_sample_size, p=drawn_conversion_rate)
        return num_conversions

    def calc_conversion_value(self, conversion_rate):
        """
        Parameters
        ----------
        conversion_rate: float
            A conversion rate (as a decimal, not a percentage)

        Returns
        -------
        conversion_rate_value: float
            The monetary value of an advertisement with a specified conversion rate

        """
        conversion_rate_value = conversion_rate * self.conversion_rate_value * 100

        return conversion_rate_value

    def calc_expected_value(self):
        """
        Computes the mean expected value/revenue of a given advertisement.

        """

        mean_conversion_rate = stats.beta(self.alpha, self.beta).mean()
        self.expected_value = self.calc_conversion_value(mean_conversion_rate)

    def plot_conversion_dist(self, color):
        """
        A helpful function for making plots of the conversion rate distribution

        Parameters
        ----------
        color: str
            A color for the distribution
        """
        plt.rcParams['font.size'] = 16
        x = np.linspace(0, .2, 1000)
        plt.fill(x * 100, stats.beta(a=self.alpha, b=self.beta).pdf(x), alpha=0.4, color=color, edgecolor='k')
        plt.xticks(np.linspace(5, 20, 4))
        plt.xlim(0, 20)
        plt.xlabel('Conversion rate (%)')
        plt.ylabel("Probability density")

    def plot_value_dist(self, color):
        """
        A helpful function for making plots of the advertisement's monetary value

        Parameters
        ----------
        color: str
            A color for the distribution
        """
        x = np.linspace(0, .2, 1000)
        plt.fill(self.calc_conversion_value(x), stats.beta(a=self.alpha, b=self.beta).pdf(x), alpha=0.4, color=color,
                 edgecolor='k')
        plt.xlabel('Revenue (thousands of dollars)')
        plt.ylabel("Probability density")


def get_overall_value(A, B):
    """
    When faced with a decision between lottery A and lottery B, the overall value is the value of the greater lottery.

    Parameters
    ----------
    A: Advertisement object
        The first option in the A/B test
    B: Advertisement object
        The second option in the A/B test

    Returns
    -------
    overall_value: float
        The overall value calculated by choosing the advertisement (between A and B) that has a higher expected
        conversion rate.

    """
    overall_value = max(A.expected_value, B.expected_value)
    return overall_value


def simulate_test(A, B, test_sample_size=1000, verbose=False, update_objects=False):
    """
    Simulates an A/B test, and then returns the expected value/revenue of the lottery after
    updating beliefs with the test results.
    
    Parameters
    ----------
    A: Advertisement object
        The first option in the A/B test
    B: Advertisement object
        The second option in the A/B test
    test_sample_size: int
        The number of ad interactions
    verbose: bool
        Whether to print information about the test
    update_objects: bool
        Whether to update the inputted Advertisements in place    
    
    Returns
    -------
    overall_value: float
        The expected value of the option that appears better after obtaining A/B test results
    """

    # Simulate test results by drawing from our priors
    num_A_interactions = A.get_test_data(test_sample_size)
    num_B_interactions = B.get_test_data(test_sample_size)

    # These results update our beliefs (the beta distributions) of the advertisement objects
    if update_objects:
        # If we want to update the input objects, then just do shallow copies
        A_updated = A
        B_updated = B
    else:
        # Otherwise deep copy to avoid modifying the inputs to this function
        A_updated = deepcopy(A)
        B_updated = deepcopy(B)

    A_updated.update_beliefs(test_sample_size, num_A_interactions)
    B_updated.update_beliefs(test_sample_size, num_B_interactions)

    # After the test, we can pick the option that appears better
    overall_value = get_overall_value(A_updated, B_updated)

    if verbose:
        print(f"Advertisement A: {num_A_interactions} conversions out of {test_sample_size}")
        print(f"Advertisement B: {num_B_interactions} conversions out of {test_sample_size}")
        if A_updated.expected_value > B_updated.expected_value:
            print("A appears to be the better option")
        else:
            print("B appears to be the better option")
    return overall_value


def calc_voi(A, B, test_sample_size=1000, num_iter=5000):
    """
    Calculates the value of information (VoI) for a particular A/B test. This is done simulating the results from many
    A/B tests. Using those results to update our beliefs about the underlying conversion rates, and then choosing the
    better option.

    Parameters
    ----------
    A: Advertisement object
        The first option in the A/B test
    B: Advertisement object
        The second option in the A/B test
    test_sample_size: int
        The number of ad interactions for each advertisement in the test
    num_iter: int
        The number of A/B tests to simulate

    Returns
    -------
    value_of_information: float
        The monetary value of the potential A/B test
    """

    # This array holds monetary values of the lottery after getting simulated test results
    value_array = np.zeros(num_iter)
    for i in tqdm(range(num_iter)):
        value_array[i] = simulate_test(A, B, test_sample_size=test_sample_size)

    # The VoI is the difference between the average value of all the simulated values and the current expected value
    value_of_information = np.mean(value_array) - get_overall_value(A, B)

    # VOI cannot be negative. It is possible to get a small negative value here due to noise
    value_of_information = max(value_of_information, 0)

    return value_of_information


def calc_voc(A, B, num_steps=10000):
    """
    This function computes the value of clairvoyance (the value of a perfect test). This is useful for testing the code
    as a test with a very large number of samples should have a value very close to the VOC.

    Parameters
    ----------
    A: Advertisement object
        The first option in the A/B test
    B: Advertisement object
        The second option in the A/B test
    num_steps: int
        The number of steps to discretize the pdfs into for convolution

    Returns
    -------
    voc: float
        The value of clairvoyance (perfect information)
    """

    # Generating a grid of points over the possible conversion rates
    x = np.linspace(0, 1, num_steps)
    dx = x[1] - x[0]

    if B.expected_value >= A.expected_value:  # If they're equal it doesn't matter which one you pick
        better_in_prior = B
        worse_in_prior = A

    else:
        better_in_prior = A
        worse_in_prior = B

    # Discretizing both pdfs
    worse_array = stats.beta(a=worse_in_prior.alpha, b=worse_in_prior.beta).pdf(x) * dx
    better_array = -np.flip(stats.beta(a=better_in_prior.alpha, b=better_in_prior.beta).pdf(x) * dx)

    # Take the convolution, which gives the pdf for value(A)-value(B)
    convolved_pdf = -np.convolve(worse_array, better_array)

    # The value of the clairvoyant comes from the possibility that the ad we think is "worse" is actually better
    worse_minus_better = np.linspace(x[0] - x[-1], x[-1] - x[0], len(convolved_pdf))

    # If we learn that our current pick is the better option, then the information produced zero value (not negative)
    voc_array = A.calc_conversion_value(worse_minus_better)
    voc_array[voc_array < 0] = 0

    # Taking the expectation
    voc = voc_array @ convolved_pdf
    return voc
