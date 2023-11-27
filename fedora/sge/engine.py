import random
import sys

from . import grammar, logger
from datetime import datetime
from tqdm import tqdm
from .operators.recombination import crossover
from .operators.mutation import mutate
from .operators.selection import tournament
from .parameters import (
    params,
    set_parameters,
    load_parameters
)


def generate_random_individual():
    genotype = [[] for key in grammar.get_non_terminals()]
    tree_depth = grammar.recursive_individual_creation(genotype, grammar.start_rule()[0], 0)
    return {'genotype': genotype, 'fitness': None, 'tree_depth' : tree_depth}


def make_initial_population():
    for i in range(params['POPSIZE']):
        yield generate_random_individual()


def evaluate(ind, eval_func):
    mapping_values = [0 for i in ind['genotype']]
    phen, tree_depth = grammar.mapping(ind['genotype'], mapping_values)
    quality, other_info = eval_func.evaluate(phen)
    ind['phenotype'] = phen
    ind['fitness'] = quality
    ind['other_info'] = other_info
    ind['mapping_values'] = mapping_values
    ind['tree_depth'] = tree_depth


def setup(parameters_file_path = None):
    if parameters_file_path is not None:
        load_parameters(file_name=parameters_file_path)
    set_parameters(sys.argv[1:])
    if params['SEED'] is None:
        params['SEED'] = int(datetime.now().microsecond)
    logger.prepare_dumps()
    random.seed(params['SEED'])
    grammar.set_path(params['GRAMMAR'])
    grammar.read_grammar()
    grammar.set_max_tree_depth(params['MAX_TREE_DEPTH'])
    grammar.set_min_init_tree_depth(params['MIN_TREE_DEPTH'])


def evolutionary_algorithm(evaluation_function=None, parameters_file=None):
    setup(parameters_file_path=parameters_file)
    population = list(make_initial_population())
    best = (sys.maxsize, -1) # Fitness, Generation
    it = 0
    while it <= params['GENERATIONS']:
        for i in tqdm(population):
            if i['fitness'] is None:
                evaluate(i, evaluation_function)
        population.sort(key=lambda x: x['fitness'])

        logger.evolution_progress(it, population, evaluation_function)

        best = min(best, (population[0]['fitness'], it))
        if evaluation_function.stop_criteria(it, best, population):
            break

        new_population = population[:params['ELITISM']]
        while len(new_population) < params['POPSIZE']:
            if random.random() < params['PROB_CROSSOVER']:
                p1 = tournament(population, params['TSIZE'])
                p2 = tournament(population, params['TSIZE'])
                ni = crossover(p1, p2)
            else:
                ni = tournament(population, params['TSIZE'])
            ni = mutate(ni, params['PROB_MUTATION'])
            new_population.append(ni)
        population = new_population
        it += 1

class EngineSGE:
    def __init__(self, parameters_file):
        self.sge_parameters = parameters_file
        load_parameters(parameters_file)
    
    def evaluate(self, individual):
        ''' SGE Fitness Function '''
        return 0, None
    
    def evolution_progress(self, population):
        ''' SGE progress_report.csv additional information '''
        return ""
    
    def stop_criteria(self, it, best, population):
        ''' SGE evolutionary cycle stop criteria '''
        return False

    def evolutionary_algorithm(self):
        ''' SGE's evolutionary algorithm core '''
        evolutionary_algorithm(evaluation_function=self, parameters_file=None)
        return self