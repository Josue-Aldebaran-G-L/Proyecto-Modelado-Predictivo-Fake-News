"""
nas_ga.py - Busqueda de Arquitectura Neuronal (NAS) con Algoritmo Genetico.

Este modulo implementa un algoritmo genetico usando la biblioteca DEAP
para explorar automaticamente el espacio de hiperparametros de TextCNN.
Cada individuo (cromosoma) codifica una configuracion completa de la red.

Autor: Curso de Modelado Predictivo 2026 - ESCOM IPN
"""

import random
from deap import base, creator, tools, algorithms

# ---------------------------------------------------------------------------
# Espacio de busqueda
# ---------------------------------------------------------------------------
EMBED_OPTS = [50, 100]
FILTER_OPTS = [64, 128, 256]
KERNEL_OPTS = [[3, 4, 5], [2, 3, 4], [3, 5, 7]]
DROPOUT_OPTS = [0.3, 0.5, 0.7]
LR_OPTS = [1e-2, 1e-3, 5e-4]


def decode(individual):
    """Decode chromosome (list of 5 ints) to TextCNN config dict."""
    return {
        'embed_dim': EMBED_OPTS[individual[0]],
        'num_filters': FILTER_OPTS[individual[1]],
        'kernel_sizes': KERNEL_OPTS[individual[2]],
        'dropout': DROPOUT_OPTS[individual[3]],
        'lr': LR_OPTS[individual[4]]
    }


def setup_toolbox(evaluate_func):
    """Configure DEAP toolbox with GA operators."""
    # Only create if not already created (avoid DEAP error on re-run)
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_embed", random.randint, 0, 1)
    toolbox.register("attr_filters", random.randint, 0, 2)
    toolbox.register("attr_kernels", random.randint, 0, 2)
    toolbox.register("attr_dropout", random.randint, 0, 2)
    toolbox.register("attr_lr", random.randint, 0, 2)

    toolbox.register("individual", tools.initCycle, creator.Individual,
        (toolbox.attr_embed, toolbox.attr_filters,
         toolbox.attr_kernels, toolbox.attr_dropout, toolbox.attr_lr), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Upper bounds per gene: embed(0-1), filters(0-2), kernels(0-2), dropout(0-2), lr(0-2)
    gene_upper = [len(EMBED_OPTS) - 1, len(FILTER_OPTS) - 1,
                  len(KERNEL_OPTS) - 1, len(DROPOUT_OPTS) - 1, len(LR_OPTS) - 1]

    def bounded_mutate(individual, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.randint(0, gene_upper[i])
        return (individual,)

    toolbox.register("evaluate", evaluate_func)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", bounded_mutate, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_evolution(toolbox, pop_size=10, ngen=10, cxpb=0.7, mutpb=0.3, verbose=True):
    """Run the genetic algorithm and return (final_population, logbook)."""
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", lambda vals: max(v[0] for v in vals))
    stats.register("avg", lambda vals: sum(v[0] for v in vals) / len(vals))

    population = toolbox.population(n=pop_size)
    result, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=cxpb, mutpb=mutpb, ngen=ngen,
        stats=stats, verbose=verbose
    )
    return result, logbook


def plot_evolution(logbook):
    """Plot fitness evolution curve (max and avg per generation)."""
    import matplotlib.pyplot as plt
    gen = logbook.select("gen")
    fit_max = logbook.select("max")
    fit_avg = logbook.select("avg")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, fit_max, 'b-', label='Mejor fitness')
    plt.plot(gen, fit_avg, 'r--', label='Fitness promedio')
    plt.xlabel('Generacion')
    plt.ylabel('F1-Score (proxy)')
    plt.title('Curva de Evolucion del Algoritmo Genetico')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
