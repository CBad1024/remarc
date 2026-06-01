import pytest 
from evodm.envs import define_chen_landscapes
from examples.run import evaluate_best_single_drug

def test_evaluate_best_single_drug():
    best_drug, best_fitness, trajectories = evaluate_best_single_drug(define_chen_landscapes())
    print(f"Best drug: {best_drug}, Best fitness: {best_fitness}")
    assert best_drug is not None