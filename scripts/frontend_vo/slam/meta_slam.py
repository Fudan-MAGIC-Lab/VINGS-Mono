#!/usr/bin/env python3

from abc import abstractclassmethod
from icecream import ic
import torch
from torch import nn

from frontend_vo.vo_factor_graph.variables import Variable, Variables
from frontend_vo.vo_factor_graph.factor_graph import FactorGraphManager

from frontend_vo.solvers.nonlinear_solver import iSAM2, LevenbergMarquardt, Solver

# An abstract learned SLAM model
class SLAM(nn.Module):
    def __init__(self, name, device):
        super().__init__()
        self.name = name
        # self.args = args
        self.device = device
        self.factor_graph_manager = FactorGraphManager()
        self.state = None
        self.delta = None
    
    # This is our spin_once
    def forward(self, batch):
        # TODO Parallelize frontend/backend
        assert("data" in batch)

        # Frontend
        output = self._frontend(batch["data"], self.state, self.delta)
        
        if output == False:
            return output
        else:
            x0, factors, viz_out = output
        
        # Backend
        # 确实，这个Backend一直是空的；
        # Dangerous Option.
        '''
        self.factor_graph_manager.add(factors)
        factor_graph = self.factor_graph_manager.get_factor_graph()
        
        self.state, self.delta = self._backend(factor_graph, x0)
        
        # What will happen if I open globalBA.
        # if type(self.backend) is iSAM2:
        #    self.factor_graph_manager.reset_factor_graph()

        self.backend = iSAM2()
        '''
        # TTD 2024/04/24
        torch.cuda.empty_cache()
        
        return [self.state, viz_out]

    # Converts sensory inputs to factors and initial guess (x0)
    @abstractclassmethod # Implemented by derived classes
    def _frontend(self, mini_batch, last_state, last_delta):
        raise
    
    # Solves the factor graph, given an initial estimate
    @abstractclassmethod # Implemented by derived classes
    def _backend(self, factor_graph, x0):
        raise


class MetaSLAM(SLAM):
    def __init__(self, name, device):
        super().__init__(name, device)

    @abstractclassmethod
    def calculate_loss(self, x, mini_batch):
        # mini_batch contains the ground-truth parameters as well
        pass
    