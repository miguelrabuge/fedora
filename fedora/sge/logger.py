import numpy as np
from .parameters import params
import json
import os



def evolution_progress(generation, pop, eval_func):
    fitness_samples = [i['fitness'] for i in pop]
    data = '%4d\t%.6e\t%.6e\t%.6e' % (generation, np.min(fitness_samples), np.mean(fitness_samples), np.std(fitness_samples))
    data += '\t' + eval_func.evolution_progress(pop)
    if params['VERBOSE']:
        print(data)
    save_progress_to_file(data)
    if (generation + 1) % params['SAVE_STEP'] == 0:
        save_step(generation, pop)


def save_progress_to_file(data):
    with open('%s/run_%d/progress_report.csv' % (params['EXPERIMENT_NAME'], params['RUN']), 'a') as f:
        f.write(data + '\n')


def save_step(generation, population):
    if params["INCLUDE_GENOTYPE"]:
        c = json.dumps(population)
    else:
        simplified_population = [{k: v for k, v in individual.items() if k != "genotype"} for individual in population]
        c = json.dumps(simplified_population)
        
    open('%s/run_%d/iteration_%d.json' % (params['EXPERIMENT_NAME'], params['RUN'], generation), 'a').write(c)


def save_parameters():
    params_lower = dict((k.lower(), v) for k, v in params.items())
    c = json.dumps(params_lower)
    open('%s/run_%d/parameters.json' % (params['EXPERIMENT_NAME'], params['RUN']), 'a').write(c)


def prepare_dumps():
    try:
        os.makedirs('%s/run_%d' % (params['EXPERIMENT_NAME'], params['RUN']))
    except FileExistsError as e:
        pass
    save_parameters()