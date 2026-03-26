"""by lyuwenyu
"""

from .solver import BaseSolver
from .det_solver import DetSolver
from .det_solver_adn import DetSolverADN


from typing import Dict 

TASKS :Dict[str, BaseSolver] = {
    'detection': DetSolver,
    'detection-adn': DetSolverADN,
}