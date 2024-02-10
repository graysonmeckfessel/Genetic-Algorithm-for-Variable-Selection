import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata

def initialization(P, C):
    """Randomly initialize the population.

    Args:
        P (int): The number of chromosomes.
        C (int): The number of predictors.

    Returns:
        array: The initial population.
    """
    
    population = np.random.choice([True, False], size = (P, C))
    return population


def fit_model(X, y, reg_type, family = None, link = None, alpha = None, var_power = None):
    """Fit the model of specified regression type.
    
    Args:
        X (array_like): Design matrix.
        y (array_like): Dependent variable.
        reg_type (str): The regression type.
        family (str, optional): The distribution family for GLM.
        link (str, optional): The link function for GLM.
        alpha (float, optional): The ancillary parameter for the negative binomial distribution family.
        var_power (float, optional): The variance power for the Tweedie distribution family.

    Returns:
        'RegressionResultsWrapper' object: The results of the fitted model.
    """
    
    if reg_type == "OLS":
        return sm.OLS(y, X).fit()
    elif reg_type == "GLM":
        if link:
            # Check if the provided distribution family is valid and the link function is avaible 
            # in the family
            try:
                family_class = getattr(sm.families, family)
                link_class = getattr(sm.families.links, link)
            except Exception as error:
                print(error)
                return None
            
            if (family == "NegativeBinomial") and (alpha != None):

                # Check if alpha is valid
                if not isinstance(alpha, float) or alpha < 0.01 or alpha > 2:
                    raise ValueError("alpha should be a float between 0.01 and 2")
                instance = family_class(link = link_class(), alpha = alpha)

            elif (family == "Tweedie") and (var_power != None):

                # Check if var_power is valid
                if not isinstance(var_power, float) or var_power < 1 or var_power > 2:
                    raise ValueError("var_power should be a float between 1 and 2")
                instance = family_class(link = link_class(), var_power = var_power)
            
            else:
                instance = family_class(link = link_class())
                
        else:
            # Check if the provided distribution family is valid
            try:
                family_class = getattr(sm.families, family)
            except Exception as error:
                print(error)
                return None
            
            if (family == "NegativeBinomial") and (alpha != None):
                if not isinstance(alpha, float) or alpha < 0.01 or alpha > 2:
                    raise ValueError("alpha should be a float between 0.01 and 2")
                instance = family_class(alpha = alpha)

            elif (family == "Tweedie") and (var_power != None):
                if not isinstance(var_power, float) or var_power < 1 or var_power > 2:
                    raise ValueError("var_power should be a float between 1 and 2")
                instance = family_class(var_power = var_power)
            
            else:
                instance = family_class()
        
    return sm.GLM(y, X, family = instance).fit()


def fitness_score(model, objective_criterion):
    """Calculate the fitness of the model.
    
    Args:
        model ('RegressionResultsWrapper' object): The results of the fitted model.
        objective_criterion (str): The criterion for evaluating the model.

    Returns: 
        float: The fitness score.
    """

    # Negative AIC
    if objective_criterion == "AIC":
        return -model.aic
    
    # Negative BIC
    elif objective_criterion == "BIC":
        return -model.bic

    # Adjusted R^2
    elif objective_criterion == "adj_rsquared":
        return model.rsquared_adj
    

def fitness(X, y, chromosome, reg_type, objective_criterion, family = None, link = None,\
            alpha = None, var_power = None):
    """Calculate the fitness of the chromosome.
    
    Args:
        X (array_like): Design matrix of the data.
        y (array_like): Dependent variable of the data.
        chromosome (array): Chromosome on which we want to calculate the fitness value.
        reg_type (str): The regression type.
        objective_criterion (str): The objective criterion to select the variables.
        family (str, optional): The distribution family for GLM.
        link (str, optional): The link function for GLM.
        alpha (float, optional): The ancillary parameter for the negative binomial distribution family.
        var_power (float, optional): The variance power for the Tweedie distribution family.

    Returns: 
        float: The fitness score.
    """

    X = sm.add_constant(np.array(X)[:, chromosome])
    model = fit_model(X, y, reg_type, family, link, alpha, var_power)
    fitness = fitness_score(model, objective_criterion)
    return fitness


def selection(X, y, population, reg_type, objective_criterion, method, family = None, link = None, \
                alpha = None, var_power = None, tournament_size = None):
    """Select parents from population to produce offspring.

    Args:
        X (array_like): Design matrix of the data.
        y (array_like): Dependent variable of the data.
        population (array): The population.
        reg_type (str): The regression type.
        objective_criterion (str): The objective criterion to select the variables.
        method (str): The selection method.
        family (str, optional): The distribution family for GLM.
        link (str, optional): The link function for GLM.
        alpha (float, optional): The ancillary parameter for the negative binomial distribution family.
        var_power (float, optional): The variance power for the Tweedie distribution family.
        tournament_size (int, optional): The number of candidates in one tournament. Only used for 
                                         tournament selection method.
    
    Returns:
        array: Half of the selected parents.
        array: Half of the selected parents.
    """

    P = population.shape[0]
    fitness_val = np.array([fitness(X, y, population[i], reg_type, objective_criterion, family, link,\
                                    alpha, var_power) for i in range(P)]) 

    # Roulette wheel method
    if method == "roulette_wheel":
        
        # If there are negative probabilities, 
        # then adjust the fitness value by subtracting the minimum
        if any(fitness_val / np.sum(fitness_val)) < 0:
            fitness_val = fitness_val - min(fitness_val)
        prob = fitness_val / np.sum(fitness_val)
        index = np.random.choice(range(P), size = (int(P/2), 2), p = prob)
        parents1 = population[index[:,0]]
        parents2 = population[index[:,1]]
    
    # Rank method
    elif method == "rank":
        rank = rankdata(fitness_val, "ordinal")
        prob = (2*rank) / (P*(P+1))
        index = np.random.choice(range(P), size = P, p = prob)
        parents1 = population[index[:int(P/2)]]
        parents2 = population[index[int(P/2):]]
    
    # Tournament method
    elif method == "tournament":
        parents = np.zeros((P, population.shape[1]), dtype = bool)
        for i in range(P):
            tournament_idx = np.random.choice(range(P), size = tournament_size)
            parents[i] = population[tournament_idx[np.argmax(fitness_val[tournament_idx])]]
        index = np.random.choice(range(P), size = int(P/2), replace = False)
        parents1 = parents[index]
        parents2 = np.delete(parents, index, axis = 0)
    
    return parents1, parents2


def point_crossover_chromosome(parent1, parent2, position):
    """Implement one point crossover on a single chromosome.

    Args:
        parent1 (array): Parent chromosome participating in the crossover.
        parent2 (array): Parent chromosome participating in the crossover.
        position (int): The locus where the crossover occurs.

    Returns:
        array: The produced offspring chromosome.
        array: The produced offspring chromosome.
    """

    offspring1 = np.concatenate((parent1[:position], parent2[position:]))
    offspring2 = np.concatenate((parent2[:position], parent1[position:]))
    return offspring1, offspring2


def uniform_crossover_chromosome(parent1, parent2):
    """Implement uniform crossover on a single chromosome.

    Args:
        parent1 (array): Parent chromosome participating in the crossover.
        parent2 (array): Parent chromosome participating in the crossover.

    Returns:
        array: The produced offspring chromosome.
        array: The produced offspring chromosome.
    """

    flag = np.random.choice([True, False], size = len(parent1))
    offspring1 = np.where(flag, parent1, parent2)
    offspring2 = np.where(~flag, parent1, parent2)
    return offspring1, offspring2


def crossover(parents1, parents2, method):
    """Implement the crossover on a set of chromosomes.

    Args:
        parents1 (array): Parent chromosomes participating in the crossover.
        parents2 (array): Parent chromosomes participating in the crossover.
        method (str): The crossover method.

    Returns:
        array: The produced offspring chromosomes.
    """

    # One-point crossover
    if method == "one-point":
        
        # Randomly select crossover positions
        positions = np.random.choice(range(parents1.shape[1]), parents1.shape[0])
        offspring_list = list(map(lambda parent1, parent2, position: \
                                    point_crossover_chromosome(parent1, parent2, position), \
                                    parents1, parents2, positions))
        offspring = np.array(offspring_list).reshape(2*parents1.shape[0], parents1.shape[1])
    
    # Two-points crossover
    elif method == "two-points":
        
        # Randomly select crossover positions
        positions_list = [np.random.choice(range(parents1.shape[1]), size = 2, replace = False) \
                            for _ in range(parents1.shape[0])]
        positions = np.array(positions_list).reshape(parents1.shape[0], 2)
        positions1, positions2 = positions[:,0], positions[:,1]

        # The first one-point crossover
        firstcrossover_list = list(map(lambda parent1, parent2, position1: \
                                        point_crossover_chromosome(parent1, parent2, position1), \
                                        parents1, parents2, positions1))
        firstcrossover = np.array(firstcrossover_list)

        # The second one-point crossover
        offspring_list = list(map(lambda parent1, parent2, position2: \
                                    point_crossover_chromosome(parent1, parent2, position2), 
                                    firstcrossover[:,0], firstcrossover[:,1], positions2))
        offspring = np.array(offspring_list).reshape(2*parents1.shape[0], parents1.shape[1])
    
    # Uniform crossover
    elif method == "uniform":
        offspring_list = list(map(lambda parent1, parent2: \
                                    uniform_crossover_chromosome(parent1, parent2), \
                                    parents1, parents2))
        offspring = np.array(offspring_list).reshape(2*parents1.shape[0], parents1.shape[1])
    
    return offspring


def mutate_chromosome(chromosome, rate):
    """Implement the mutation on a single chromosome.

    Args:
        chromosome (array): The chromosome on which the mutation is implemented.
        rate (float): The probability of mutating.

    Returns:
        array: The mutated chromosome.
    """

    mutated_chromosome = list(map(lambda x: np.random.choice([not x, x], p = [rate, 1-rate]), \
                                    chromosome))
    return mutated_chromosome


def mutation(offspring, rate):
    """Implement the mutation on a set of chromosomes.
    Args:
        offspring (array): The offspring on which the mutation is implemented.
        rate (float): The probability of mutating.

    Returns:
        array: The mutated offspring.
    """

    offspring = np.array(list(map(lambda chromosome: mutate_chromosome(chromosome, rate), offspring)))
    return offspring


def select(X, y, P = None, reg_type = "OLS", family = "Gaussian", link = None, alpha = None, \
            var_power = None, max_iter = 100, objective_criterion = "AIC", selection_method = "rank", \
            tournament_size = None, crossover_method = "one-point", mutation_rate = 0.01):
    """Implement the variable selection using GA algorithm.

    Args:
        X (array_like): Design matrix of shape (n, k) where n is the number of observations and 
        k is the number of regressors.
        
        y (array_like): Dependent variable of shape (n, ) where n is the number of observations.
        
        P (None or int, optional): The size of the generation. It should be an even positive integer. 
        If P is None or not provided, it will be set as 2*k where k is the number of regressors.
        
        reg_type (str, optional): The regression type. Available options are "OLS" and "GLM". If it 
        is "OLS", the ordinary least square regression will be implemented. If it is "GLM", the 
        generalized least squares regression will be implemented. The default is "OLS".
        
        family (str, optional): The distribution family for generalized least squares. Available 
        options are "Binomial", "Gamma", "Gaussian", "InverseGaussian", "NegativeBinomial", "Poisson" 
        and "Tweedie". The default is "Gaussian".
        
        link (str, optional): The link function for generalized least squares. The list of available 
        link functions for each distribution family can be obtained by 
        >>> statsmodels.families.family.<familyname>.links. 
        If it is None or not provided, it will be set as the default link function of the distribution 
        family.
        
        alpha (float, optional): The ancillary parameter for the negative binomial distribution family. 
        If it is None or not provided, it will be set as the default value 1. Permissible values are 
        usually assumed to be between 0.01 and 2.

        var_power (float, optional): The variance power for the Teedie distribution family. It should 
        be between 1 and 2. If it is None or not provided, it will be set as 1.
        
        max_iter (int, optional): The maximum number of iterations. The default is 100.
        
        objective_criterion (str, optional): The objective criterion to select the variables. Available 
        options are "AIC", "BIC" and "adj_rsquared". The default is "AIC". And GLM doesn't support 
        "adj_rsquared".

        selection_method (str, optional): The selection method. Available options are "roulette_wheel", 
        "rank" and "tournament". The default is "rank".

        tournament_size (int, optional): The tournament size for the tournament selection method. It 
        should be an integer in the range (0, P). If it is None or not provided, it will be set as 
        20% of the generation size.

        crossover_method (str, optional): The crossover method. Available options are "one-point", 
        "two-points" and "uniform". The default is "one-point".

        mutation_rate (float, optional): The probability of mutating. It should be in the range [0, 1]. 
        The default is 0.01.

    Returns:
        boolean array: The best combination of the predictors
        float: The fitness value of the best solution
        list: The best fitness value of each generation

    Raises:
        ValueError: If P is not an even positive integer.
        ValueError: If reg_type is not "OLS" or "GLM".
        ValueError: If max_iter is not a positive integer.
        ValueError: If objective_criterion is not "AIC", "BIC" or "adj_rsquared".
        ValueError: If objective_criterion "adj_rsquared" is used for reg_type "GLM".
        ValueError: If selection_method is not "roulette_wheel", "rank" or "tournament".
        ValueError: If crossover_method is not "one-point", "two-points" or "uniform".
        ValueError: If tournament_size is not an integer between 0 and P.
        ValueError: If mutation_rate is not a float between 0 and 1.
    
    Examples:
        # Generate dataset
        >>> X = np.random.normal(size = (100, 10))
        >>> y = np.random.normal(size = 100)
        
        # Implement OLS
        >>> select(X, y, P = 50)

        # Implement GLM
        >>> select(X, y, P = 50, reg_type = "GLM", family = "Gaussian", link = "Identity", 
        objective_criterion = "BIC", selection_method = "tournament", tournament_size = 3, 
        crossover_method = "two-points")
    """

    C = X.shape[1]

    if P is None:
        P = 2 * C
    
    if (selection_method == "tournament") and (tournament_size == None):
        tournament_size = int(0.2 * P)
    
    # Check for P
    if not isinstance(P, int) or P <= 0:
        raise ValueError("P should be a positive integer")
    elif P % 2 != 0:
        raise ValueError("P shoule be even")
    
    # Check for reg_type
    if reg_type not in ["OLS", "GLM"]:
        raise ValueError("Invalid reg_type")
    
    # Check for max_iter
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter should be a positive integer")

    # Check for objective_criterion
    if objective_criterion not in ["AIC", "BIC", "adj_rsquared"]:
        raise ValueError("Invalid objective_criterion")
    if objective_criterion == "adj_rsquared" and reg_type == "GLM":
        raise ValueError("GLM doesn't support adj_rsquared")
    
    # Check for selection_method
    if selection_method not in ["roulette_wheel", "rank", "tournament"]:
        raise ValueError("Invalid selection_method")
    
    # Check for crossover_method
    if crossover_method not in ["one-point", "two-points", "uniform"]:
        raise ValueError("Invalid crossover_method")
    
    # Check for tournament_size
    if (tournament_size != None) and (not isinstance(tournament_size, int) or tournament_size <= 0 \
        or tournament_size >= P):
        raise ValueError("tournament_size should be an integer between 0 and P")
    
    # Check for mutation_rate
    if not (0 <= mutation_rate <= 1):
        raise ValueError("mutation_rate should be between 0 and 1")
    
    population = initialization(P, C)
    best_fitness_list = []
    for i in range(max_iter):
        parents1, parents2 = selection(X, y, population, reg_type = reg_type, \
                                        objective_criterion = objective_criterion, \
                                        method = selection_method, family = family, link = link, \
                                        tournament_size = tournament_size)
        offspring = crossover(parents1, parents2, method = crossover_method)
        offspring = mutation(offspring, rate = mutation_rate)
        population = offspring

        fitness_val = np.array([fitness(X, y, population[i], reg_type, objective_criterion, family, link) \
                                for i in range(P)])
        index = np.argmax(fitness_val)
        best_fitness = fitness_val[index]
        best_fitness_list.append(best_fitness)

    fitness_val = np.array([fitness(X, y, population[i], reg_type, objective_criterion, family, link) \
                            for i in range(P)])
    index = np.argmax(fitness_val)
    best_sol = population[index]
    best_fitness = fitness_val[index]
    return best_sol, best_fitness, best_fitness_list

