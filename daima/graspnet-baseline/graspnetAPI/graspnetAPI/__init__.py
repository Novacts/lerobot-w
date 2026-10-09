__author__ = 'mhgou'
__version__ = '1.2.11'

from .graspnet import GraspNet
from .grasp import Grasp, GraspGroup, RectGrasp, RectGraspGroup

# GraspNetEval has heavy deps (dexnet, IPython) – only load when actually needed
try:
    from .graspnet_eval import GraspNetEval
except ImportError:
    GraspNetEval = None
