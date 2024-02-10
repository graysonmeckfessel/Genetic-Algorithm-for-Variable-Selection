import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import GA
import scipy.stats as stats
# from sklearn.datasets import make_regression   
from math import log, sqrt

np.random.seed(12345)

@pytest.fixture
def population():
    num_individuals = 100 
    num_features = 20

    # Generate a random initial population
    initial_population = np.random.choice([True, False], size = (num_individuals, num_features))

    return initial_population

@pytest.fixture
def test_data():
    n_samples = 100
    n_features = 20
    X = pd.DataFrame(np.random.rand(n_samples, n_features), columns = [f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.rand(n_samples))
    return X, y

@pytest.fixture
def data():
    n_samples = 100
    n_features = 20
    X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.rand(n_samples))
    return X, y

@pytest.fixture
def parent_chromosomes():
    chromosome_length = 20

    parent1 = np.random.choice([True, False], size=chromosome_length)
    parent2 = np.random.choice([True, False], size=chromosome_length)

    return parent1, parent2

@pytest.fixture
def init_params():
    beta0, beta1, sigma2 = 1, 2, 0.5
    return [beta0, beta1, sigma2]

@pytest.fixture
def init_data():
    n_samples = 100
    n_features = 20
    X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.rand(n_samples))
    return X, y

@pytest.fixture
def ols_model():
    n_samples = 100
    n_features = 20
    X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.rand(n_samples))
    model = sm.OLS(y, X)
    fitted_model = model.fit()
    return fitted_model


# Helper Functions
def is_boolean_matrix(matrix):
    """Check if a numpy array contains only boolean values."""
    return np.isin(matrix, [True, False]).all()

def selection_chi_square_test(X, y, pop, method, probs, num_trials):
    # Perform a chi square test on the selection procedure for a given population and method
    # For the null hypothesis the data comes from a multinomial with the given probabilities

    # Generate parents based on the selection method for the given number of trials
    parents = []
    for _ in range(num_trials):
        parents1, parents2 = GA.selection(X, y, pop, "OLS", "AIC", method)
        parents.extend(parents1.tolist() + parents2.tolist())

    # Convert parents to a format suitable for chi-square test
    parent_keys = [''.join(['T' if gene else 'F' for gene in parent]) for parent in parents]
    population_keys = [''.join(['T' if gene else 'F' for gene in chromosome]) for chromosome in pop]
   
    
    # Count occurrences of each unique chromosome configuration
    observed_counts = {key: parent_keys.count(key) for key in set(parent_keys)}
    ordered_observed_counts = [observed_counts[k] for k in population_keys]

    
    # Calculate expected counts based on probabilities
    expected_counts = [probs[i] * len(parents) for i in range(len(probs))]

    # Perform chi-square test
    chi_square_stat, pvalue = stats.chisquare(ordered_observed_counts, f_exp = expected_counts)
    return chi_square_stat, pvalue

# Testing invalid paramter inputs
def test_invalid_parameters(init_data, init_params):
    """Test functions with invalid input parameters."""
    X, y = init_data
    params = init_params

    # Initialization function with invalid parameters
    with pytest.raises(ValueError):
        GA.initialization(-10, 5)   # Negative number of chromosomes
    with pytest.raises(ValueError):
        GA.initialization(10, -5)   # Negative chromosome length
    with pytest.raises(TypeError):
        GA.initialization("10", 5)  # Non-integer number of chromosomes
    with pytest.raises(TypeError):
        GA.initialization(10, "5")  # Non-integer chromosome length

    # Select function with invalid parameters
    with pytest.raises(ValueError):
        GA.select(X, y, -1) # Negative population size
    with pytest.raises(ValueError):
        GA.select(X, y, 10, max_iter = -100)    # Negative max_iter
    with pytest.raises(ValueError):
        GA.select(X, y, 10, max_iter = "100")     # Non-integer max_iter
    with pytest.raises(ValueError):
        GA.select(X, y, 10, selection_method = "unknown_method")  # Invalid selection method
    with pytest.raises(ValueError):
        GA.select(X, y, 10, crossover_method = "unknown_method")  # Invalid crossover method
    with pytest.raises(ValueError):
        GA.select(X, y, 10, mutation_rate = -0.01)  # Negative mutation rate
    with pytest.raises(ValueError):
        GA.select(X, y, 10, mutation_rate = 1.1)  # Mutation rate greater than 1

def test_initialization():
    population = GA.initialization(0, 0)
    assert population.size == 0, "Population size should be zero for zero inputs"

    P, C = 1000, 500
    population = GA.initialization(P, C)
    assert population.shape == (P, C), "Shape should match large input dimensions"
    assert is_boolean_matrix(population), "Population should contain only Boolean values even for large inputs"

    # Hypothesis test that the flattened Population (all chromosome loci combined),
    # is drawn from a binomial with success probabability 0.5
    alpha = 0.05
    P, C = 10, 20
    pop = (GA.initialization(P, C)).flatten()
    prob_pval = stats.binomtest(np.sum(pop), len(pop), 0.5).pvalue
    assert prob_pval > alpha


def test_fit_model(ols_model, init_data, init_params):
    """Test if the OLS generates the correct parameter estimates."""
    X, y = init_data
    params = init_params
    model = ols_model

    tolerance = 3 # This probably needs to be adjusted

    assert model.params.iloc[0] == pytest.approx(params[0], abs = tolerance)
    assert model.params.iloc[1] == pytest.approx(params[1], abs = tolerance)


def test_fitness_score(ols_model, init_data, init_params):
    """Test fitness score calculations."""
    X, y = init_data
    params = init_params
    model = sm.OLS(y, X).fit()

    neg_aic = GA.fitness_score(model, "AIC")
    neg_manual_aic = -(-2*model.llf + 2*X.shape[1])
    neg_bic = GA.fitness_score(model, "BIC")
    neg_manual_bic = -(-2*model.llf + np.log(X.shape[0])*X.shape[1])

    tolerance = 1 # This probably needs to be adjusted

    assert neg_aic == pytest.approx(neg_manual_aic, abs = tolerance)
    assert neg_bic == pytest.approx(neg_manual_bic, abs = tolerance)

def test_fitness(init_data, init_params, ols_model):
    """Test if the fitness function fits the correct model and returns the same fitness as the manual calculation."""
    X, y = init_data
    params = init_params
    model = sm.OLS(y, X).fit()
    chromosome = np.array([True] * X.shape[1])  
    fitness = GA.fitness(X, y, chromosome, "OLS", "AIC")
    neg_manual_aic = -(-2*model.llf + 2*X.shape[1])

    tolerance = 5 
    assert fitness == pytest.approx(neg_manual_aic, abs = tolerance)

def test_selection(init_data, init_params, ols_model):
    """Test selection function."""
    X, y = init_data
    num_trials, alpha = 10, 0.05
    P, C = 4, X.shape[1]
    population = np.random.choice([True, False], size=(P, C))

    fitness_val = np.array([GA.fitness(X, y, population[i], "OLS", "AIC") for i in range(P)])
    
    roulette_prob = fitness_val / np.sum(fitness_val)
    chi_square_stat, pval = selection_chi_square_test(X, y, population, "roulette_wheel", roulette_prob, num_trials)
    assert pval > alpha

    rank = stats.rankdata(fitness_val, "ordinal")
    rank_prob = (2*rank) / (P*(P+1))
    chi_square_stat, pval = selection_chi_square_test(X, y, population, "rank", rank_prob, num_trials)
    assert pval > alpha



def test_point_crossover_chromosome():
    parent1 = np.array([True, True, True, True, True])
    parent2 = np.array([False, False, False, False, False])

    children = GA.point_crossover_chromosome(parent1, parent2, 0)
    assert np.array_equal(children[0], np.array([False, False, False, False, False])) and  np.array_equal(children[1], np.array([True, True, True, True, True]))

    children = GA.point_crossover_chromosome(parent1, parent2, 1)
    assert np.array_equal(children[0], np.array([True, False, False, False, False])) and  np.array_equal(children[1], np.array([False, True, True, True, True]))

    children = GA.point_crossover_chromosome(parent1, parent2, 2)
    assert np.array_equal(children[0], np.array([True, True, False, False, False])) and  np.array_equal(children[1], np.array([False, False, True, True, True]))

    children = GA.point_crossover_chromosome(parent1, parent2, 3)
    assert np.array_equal(children[0], np.array([True, True, True, False, False])) and  np.array_equal(children[1], np.array([False, False, False, True, True]))

    children = GA.point_crossover_chromosome(parent1, parent2, 4)
    assert np.array_equal(children[0], np.array([True, True, True, True, False])) and  np.array_equal(children[1], np.array([False, False, False, False, True]))

def test_uniform_crossover_chromosome():
    # Perform a test of the null hypothesis the likelihood of heads or tails in each child is 0.5
    # Given our parents are each uniformly True or False, this will test if we have successfully fused them uniformly
    # This fails to capture the distribution of T/F within the sequence, so would miss the edge case we randomly point crossover at a random point
    # Perhaps this should include  longest string of heads/tails test stat /p value, and AR(1) test
    n, alpha = 1000, 0.05
    parent1, parent2 = np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

    children = GA.uniform_crossover_chromosome(parent1, parent2)
    observed_heads1, observed_heads2 = np.sum(children[0]), np.sum(children[1])
    child1_prob_pval = stats.binomtest(observed_heads1, n, 0.5).pvalue
    child2_prob_pval = stats.binomtest(observed_heads2, n, 0.5).pvalue

    assert child1_prob_pval > alpha and child2_prob_pval > alpha

def test_crossover_chromosome():
    alpha = 0.05
    n, C = 1000, 5
    parents1 = np.ones((n, C), dtype=bool)
    parents2 = np.zeros((n, C), dtype=bool)
    # We can interpret the chance a child remains all TRUE/ FALSE as a bernoulli trial with success probability 1/C
    # Thus we have the sum of bernoulli trials, which is binomial (n, 1/C)
    # While our trials are not independent, as each pair is tied, when working with proportion, dividing by 2n means we don't violate our distributional assumptions

    # We can construct Wald test statistics using var(X) = np(1-p) => se(p) = sqrt(np(1-p))
    # W ~ N(0,1), reject for pval < alpha
    offspring1  = GA.crossover(parents1, parents2, 'one-point')
    p_null1 = 1/C
    p_hat1 = np.sum(list(map(lambda child:  np.sum(child) in {0,C}, offspring1))) / (2*n)
    W1 = abs((p_hat1 - p_null1) /  sqrt(n*p_null1*(1-p_null1)))
    pval1 = stats.norm.sf(W1)


    # Observe that if the offspring has two changepoints, the series will start and end with the same value
    # No offspring can be unchanged, as we sample without replacement
    # but we can count the number of offspring that only have one changepoint, and this has probability 9/20
    # this is counted by counting the number of series that start and end with a different value
    offspring2  = GA.crossover(parents1, parents2, 'two-points')
    p_null2 = round(stats.hypergeom.pmf(1, C, 1, 2), 10) # for two point, the likelihood of getting one change is hypergeom
    one_change = list(map(lambda child:  child[0] != child[-1], offspring2))
    p_hat2 = np.sum(one_change) / (2*n)
    W2 = abs((p_hat2 - p_null2) /  sqrt(n*p_null2*(1-p_null2)))
    pval2 = stats.norm.sf(W2)

    assert pval1 > alpha and pval2 > alpha

def test_mutate_chromosome():
    rate = 0.01
    alpha = 0.0001 # expected fail rate 0.01%
    n = np.ceil(log(alpha) / log(1-rate)).astype('int') # choose n such that we have  (1-alpha)% likelihood of at least one mutation occuring
    chromosome = np.zeros(n, dtype=bool)
    assert np.sum(GA.mutate_chromosome(chromosome, rate)) > 0 # this will fail (alpha * 100) percent of the time

def test_mutation():
    rate = 0.01
    alpha = 0.0001 # expected fail rate 0.01%
    n = log(alpha) / log(1-rate) # choose n such that we have  (1-alpha)% likelihood of at least one mutation occuring
    C = 4
    P = np.ceil(n/C).astype('int')
    pre_mutation = np.zeros((P, C), dtype=bool)
    offspring = GA.mutation(pre_mutation, rate)
    assert offspring.shape == (P, C)
    assert np.sum(offspring.flatten()) > 0

def test_select():
    num_trials = 10
    n = 100
    for i in range(num_trials):
        beta = np.random.choice([1, 0], size=5)

        X = pd.DataFrame(np.hstack([np.ones((n, 1)), np.random.uniform(size=(n, len(beta)-1))]))
        y = X @ beta.T
        sol, fitness, best_fitness_list = GA.select(X, y)

        # Check if the fitness is above a threshold
        fitness_threshold = 0.1
        assert fitness > fitness_threshold


@pytest.fixture
def large_dataset():
    """Fixture for generating large datasets for performance testing."""
    n_samples = 100
    n_features = 20
    X = pd.DataFrame(np.random.rand(n_samples, n_features), columns = [f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.rand(n_samples))
    return pd.DataFrame(X), pd.Series(y)

# Selection Function Tests
@pytest.mark.parametrize("method", ["probability", "rank", "tournament"])
def test_selection_methods(test_data, method, population):
    """Test the selection function with different methods."""
    X, y = test_data
    
    reg_type = "OLS" 
    objective_criterion = "AIC"  
    method = "rank"  

    parents1, parents2 = GA.selection(X, y, population, reg_type, objective_criterion, method)

    P, C = population.shape
    assert parents1.shape == (P // 2, C), f"Parents1 shape incorrect for {method}"
    assert parents2.shape == (P // 2, C), f"Parents2 shape incorrect for {method}"

def test_selection_large_population(large_dataset, population):
    """Test selection function with a large population."""
    X, y = large_dataset
    P, C = population.shape

    reg_type = "OLS"
    objective_criterion = "AIC"
    method = "rank"

    parents1, parents2 = GA.selection(X, y, population, reg_type, objective_criterion, method)

    assert len(parents1) > 0, "Parents1 should not be empty"
    assert len(parents2) > 0, "Parents2 should not be empty"
    assert len(parents1) == P // 2, "Parents1 should be half of the original population size"
    assert len(parents2) == P // 2, "Parents2 should be half of the original population size"
    assert not np.array_equal(parents1, parents2), "Parents1 and Parents2 should not be identical"
    assert parents1.shape[1] == C, "Shape of Parents1 should match the number of features"
    assert parents2.shape[1] == C, "Shape of Parents2 should match the number of features"

def test_selection_probability(test_data, population):
    """Test the probability selection method."""
    X, y = test_data
    
    reg_type = "OLS" 
    objective_criterion = "AIC"  
    method = "rank"  

    parents1, parents2 = GA.selection(X, y, population, reg_type, objective_criterion, method)

    assert len(parents1) == len(parents2), "Number of parents should be equal"
    assert parents1.shape == parents2.shape, "Shapes of parent groups should be equal"

def test_selection_rank(test_data, population):
    """Test the rank selection method."""
    X, y = test_data

    reg_type = "OLS" 
    objective_criterion = "AIC"  
    method = "rank"  

    parents1, parents2 = GA.selection(X, y, population, reg_type, objective_criterion, method)

    assert len(parents1) == len(parents2), "Number of parents should be equal"
    assert parents1.shape == parents2.shape, "Shapes of parent groups should be equal"

def test_selection_tournament(test_data, population):
    """Test the tournament selection method."""
    X, y = test_data    
    
    reg_type = "OLS" 
    objective_criterion = "AIC"  
    method = "rank"  

    parents1, parents2 = GA.selection(X, y, population, reg_type, objective_criterion, method)

    assert len(parents1) == len(parents2), "Number of parents should be equal"
    assert parents1.shape == parents2.shape, "Shapes of parent groups should be equal"

def test_selection_similar_fitness(test_data, population):
    """Test selection function when all chromosomes have similar fitness."""
    X, y = test_data
    y[:] = 1  # Making y constant to simulate same fitness for all chromosomes
    reg_type = "OLS"
    objective_criterion = "AIC"
    method = "rank"

    parents1, parents2 = GA.selection(X, y, population, reg_type, objective_criterion, method)

    assert len(parents1) == len(population) // 2, "First half of parents should be returned"
    assert len(parents2) == len(population) // 2, "Second half of parents should be returned"

# Crossover Function Tests
@pytest.mark.parametrize("method", ["one-point", "two-points", "uniform"])
def test_crossover_methods(method):
    # Create parent chromosomes
    parent1 = np.array([True, False, False, True, True, False, True, False, True, False])
    parent2 = np.array([False, True, True, False, False, True, False, True, False, True])

    # Convert to 2D array if necessary
    parents1 = parent1[np.newaxis, :]
    parents2 = parent2[np.newaxis, :]

    offspring = GA.crossover(parents1, parents2, method)

    assert isinstance(offspring, np.ndarray), "Offspring must be an array"
    assert offspring.shape[0] == 2 * parents1.shape[0], "Incorrect number of offspring"
    assert offspring.shape[1] == parents1.shape[1], "Offspring chromosomes have incorrect length"

# Mutation Function Tests
def test_mutation_rate(population):
    """Test mutation function for different rates."""
    mutated_population = GA.mutation(population, 0.1)
    P, C = population.shape
    assert mutated_population.shape == (P, C), "Mutated population shape incorrect"

def test_zero_mutation_rate(population):
    """Test mutation function with zero mutation rate."""
    mutated_population = GA.mutation(population, 0.0)
    assert np.array_equal(mutated_population, population), "No mutation should occur with a zero mutation rate"

def test_full_mutation_rate(population):
    """Test mutation function with full mutation rate."""
    mutated_population = GA.mutation(population, 1.0)
    assert np.array_equal(mutated_population, ~population), "All genes should be mutated with a 100% mutation rate"

# Select Function Tests
def test_select_function(test_data):
    """Test the primary select function for different configurations."""
    X, y = test_data
    P = 10
    best_sol, best_fitness, best_fitness_list = GA.select(X, y, P)
    assert len(best_sol) == len(X.columns), "Best solution length incorrect"
    assert isinstance(best_fitness, float), "Best fitness should be a float"

@pytest.mark.parametrize("max_iter, selection_method, crossover_method, mutation_rate", [
    (50, "rank", "one-point", 0.01),
    (100, "roulette_wheel", "two-points", 0.05),
    (150, "tournament", "uniform", 0.1),
    (200, "rank", "uniform", 0.02),
    (250, "roulette_wheel", "one-point", 0.03),
    (300, "tournament", "two-points", 0.04),
])

def test_select_various_configs(test_data, max_iter, selection_method, crossover_method, mutation_rate):
    """Test the select function with various configurations."""
    X, y = test_data
    best_solution, best_fitness, best_fitness_list = GA.select(X, y, P = 20, max_iter = max_iter, selection_method = selection_method, crossover_method = crossover_method, mutation_rate = mutation_rate)
    assert isinstance(best_solution, np.ndarray), "Best solution should be a numpy array"
    assert isinstance(best_fitness, float), "Best fitness should be a float"
    assert best_solution.shape[0] == X.shape[1], "Best solution should have the same length as the number of features"

def test_select_stochastic_behavior(test_data):
    """Test the consistency of select function over multiple runs."""
    threshold = 1.0 # This is arbitrary and can be adjusted
    X, y = test_data
    P = 10
    results = [GA.select(X, y, P) for _ in range(10)]
    print(results)
    fitness_values = [result[1] for result in results]
    assert np.std(fitness_values) < threshold, "Stochastic behavior should be within an acceptable range"


# Additional Tests
def test_with_empty_dataset():
    """Test the select function with an empty dataset."""
    with pytest.raises(ValueError):
        GA.select(pd.DataFrame(), pd.Series(), 100)

def test_non_boolean_chromosome(large_dataset):
    """Test fitness function with non-Boolean chromosome."""
    X, y = large_dataset
    chromosome = np.random.rand(X.shape[1])  # Non-Boolean chromosome
    with pytest.raises(TypeError):
        GA.fitness(X, y, chromosome)

def test_full_algorithm_integration(test_data):
    """Integration test simulating a full run of the algorithm."""
    X, y = test_data

    reg_type = "OLS" 
    objective_criterion = "AIC"  
    method = "rank"  

    P = 10
    population = GA.initialization(P, len(X.columns))
    
    for _ in range(100):  # Simulate iterations
        parents1, parents2 = GA.selection(X, y, population, reg_type, objective_criterion, method)
        offspring = GA.crossover(parents1, parents2, "one-point")
        population = GA.mutation(offspring, 0.01)

    best_sol, best_fitness, best_fitness_list = GA.select(X, y, P)
    assert isinstance(best_sol, np.ndarray) and isinstance(best_fitness, float), "Integration test failed"