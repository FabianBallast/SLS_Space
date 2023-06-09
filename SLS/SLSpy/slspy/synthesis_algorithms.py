import cvxpy
import cvxpy as cp
import numpy as np

from .core import SystemModel, ControllerModel, SynthesisAlgorithm
from .system_models import *

from .sls import *
from .iop import *

'''
To create a new synthesis algorithm, inherit the following base function and customize the specified methods.

class SynthesisAlgorithm:
    def __init__(self,system_model=None):
    def synthesizeControllerModel(self):
        return controller_model
'''


class SLS(SynthesisAlgorithm):
    '''
    Synthesizing the controller using System Level Synthesis method.
    '''

    def __init__(self,
                 system_model=None,
                 FIR_horizon=1,
                 state_feedback=True,
                 solver=None,
                 noise_free=False
                 ):
        self._FIR_horizon = FIR_horizon
        self._state_feedback = state_feedback
        self._system_model = None

        self.setSystemModel(system_model=system_model)

        self.resetObjAndCons()

        self.setSolver(solver=solver)

        self._sls_constraints = SLS_Cons_SLS()

        self._noise_free = noise_free
        self.initialised = False

    def setSystemModel(self, system_model):
        if isinstance(system_model, SystemModel):
            self._system_model = system_model

        self.initializePhi()

        return self

    def setSolver(self, solver):
        # optimizer is embedded in the solver
        if not isinstance(solver, SLS_Solver):
            solver = None
        if solver is None:
            self._solver = SLS_Sol_CVX()
        else:
            self._solver = solver
        self._solver._sls = self

    def getSolver(self):
        return self._solver

    def initializePhi(self):
        self._Phi_x = self._Phi_xx = []
        self._Phi_u = self._Phi_ux = []
        self._Phi_xy = []
        self._Phi_uy = []

        if self._system_model is None:
            return

        self._use_state_feedback_version = self._state_feedback or self._system_model._state_feedback

        Nx = self._system_model._Nx
        Nu = self._system_model._Nu

        if self._use_state_feedback_version:
            for tau in range(self._FIR_horizon + 1):
                self._Phi_x.append(cp.Variable(shape=(Nx, Nx)))
                self._Phi_u.append(cp.Variable(shape=(Nu, Nx)))
        else:
            Ny = self._system_model._Ny

            for tau in range(self._FIR_horizon + 1):
                self._Phi_xx.append(cp.Variable(shape=(Nx, Nx)))
                self._Phi_ux.append(cp.Variable(shape=(Nu, Nx)))
                self._Phi_xy.append(cp.Variable(shape=(Nx, Ny)))
                self._Phi_uy.append(cp.Variable(shape=(Nu, Ny)))

    # overload plus and less than or equal operators as syntactic sugars
    def __add__(self, obj_or_cons):
        return self.addObjOrCons(obj_or_cons)

    def __lshift__(self, obj_or_cons_or_system):
        if isinstance(obj_or_cons_or_system, SystemModel):
            return self.setSystemModel(system_model=obj_or_cons_or_system)
        else:
            return self.setObjOrCons(obj_or_cons=obj_or_cons_or_system)

    def resetObjAndCons(self):
        self.resetObjectives()
        self.resetConstraints()

    def resetObjectives(self):
        self._objectives = []
        self._optimal_objective_value = float('inf')

    def resetConstraints(self):
        self._constraints = []

    def addObjOrCons(self, obj_or_cons):
        if isinstance(obj_or_cons, SLS_Constraint):
            self._constraints.append(obj_or_cons)
        elif isinstance(obj_or_cons, SLS_Objective):
            self._objectives.append(obj_or_cons)
        return self

    def setObjOrCons(self, obj_or_cons):
        if isinstance(obj_or_cons, SLS_Constraint):
            self._constraints = []
            self._constraints.append(obj_or_cons)
        elif isinstance(obj_or_cons, SLS_Objective):
            self._objectives = []
            self._objectives.append(obj_or_cons)
        return self

    def getOptimalObjectiveValue(self):
        return self._optimal_objective_value

    def get_SLS_Problem(self):
        if isinstance(self._solver, SLS_Sol_CVX):
            return self._solver.get_SLS_Problem()
        else:
            return None

    def sanityCheck(self):
        # we can extend the algorithm to work for non-state-feedback SLS
        # if not self._state_feedback:
        #    return self.errorMessage('Only support state-feedback case for now.')

        if self._system_model is None:
            return self.errorMessage('The system is not yet assigned.')
        if not isinstance(self._system_model, LTI_System) and not isinstance(self._system_model, LTV_System):
            return self.errorMessage('The system must be an LTI_System or LTV_System.')
        if not isinstance(self._FIR_horizon, int):
            return self.errorMessage('FIR horizon must be integer.')
        if self._FIR_horizon < 1:
            return self.errorMessage('FIR horizon must be at least 1.')
        if self._noise_free and self._system_model._x0 is None:
            return self.errorMessage("Provide an initial state when there is no noise.")

        return True

    def create_optimisation_problem(self, reference: np.ndarray = None) -> None:
        """
        Create the optimisation problem using CVXPY. Using x0 as a parameter, this only has to be done once.

        :param reference: Reference to follow.
        """
        self._optimal_objective_value = float('inf')
        if not self.sanityCheck():
            return None
        if not self._system_model.sanityCheck():
            self.errorMessage('System model check fails.')
            return None

        # variables used by both the state-feedback and output-feedback versions
        Nx = self._system_model._Nx
        Nu = self._system_model._Nu

        if self._use_state_feedback_version != (self._state_feedback or self._system_model._state_feedback):
            self.initializePhi()

        if self._use_state_feedback_version:
            self.controller = SLS_StateFeedback_FIR_Controller(
                Nx=Nx, Nu=Nu,
                FIR_horizon=self._FIR_horizon
            )
        else:
            # output-feedback
            Ny = self._system_model._Ny

            self.controller = SLS_OutputFeedback_FIR_Controller(
                Nx=Nx, Nu=Nu, Ny=Ny, D22=self._system_model._D22,
                FIR_horizon=self._FIR_horizon
            )

        # objective
        objective_value = 0
        for obj in self._objectives:
            objective_value = obj.addObjectiveValue(sls=self, objective_value=objective_value, x_ref=reference)

        # add SLS main constraints
        self._sls_constraints._state_feedback = self._use_state_feedback_version
        constraints = self._sls_constraints.addConstraints(sls=self)

        # the constraints might also introduce additional terms at the objective
        for cons in self._constraints:
            objective_value = cons.addObjectiveValue(sls=self, objective_value=objective_value)
            constraints = cons.addConstraints(sls=self, constraints=constraints)

        self._solver.setOptions(solver=cvxpy.ECOS, verbose=False)  # numericFocus=3, warmStart=True)
        self._solver.set_objective_and_constraints(objective_value=objective_value, constraints=constraints)
        self.initialised = True

    def synthesizeControllerModel(self, reference=None):
        if not self.initialised or True:
            self.create_optimisation_problem(reference)

        # for obj in self._objectives:
        #     obj.state_est.value = self._system_model._x0

        problem_value, solver_status = self._solver.solve()
        total = self._FIR_horizon + 1

        if solver_status == 'infeasible':
            self.warningMessage('SLS problem infeasible')
            return None
        elif solver_status == 'unbounded':
            self.warningMessage('SLS problem unbounded')
            return None
        else:
            # save the solved problem for the users to examine if needed
            self._optimal_objective_value = problem_value
            if self._use_state_feedback_version:
                self.controller._Phi_x = [None] * total
                self.controller._Phi_u = [None] * total
                for tau in range(total):
                    self.controller._Phi_x[tau] = self._Phi_x[tau].value
                    self.controller._Phi_u[tau] = self._Phi_u[tau].value
            else:
                self.controller._Phi_xx = [None] * total
                self.controller._Phi_ux = [None] * total
                self.controller._Phi_xy = [None] * total
                self.controller._Phi_uy = [None] * total
                for tau in range(total):
                    self.controller._Phi_xx[tau] = self._Phi_xx[tau].value
                    self.controller._Phi_ux[tau] = self._Phi_ux[tau].value
                    self.controller._Phi_xy[tau] = self._Phi_xy[tau].value
                    self.controller._Phi_uy[tau] = self._Phi_uy[tau].value

            self.controller.initialize()
            return self.controller


class IOP(SynthesisAlgorithm):
    '''
    Synthesizing the controller using Input-Output Parametrization method, c.f.
        Furieri et al., ``An Input-Output Parametrization of Stabilizing Controllers: Amidst Youla and System Level Synthesis,'' 2019.
    This is a finite impulse response implementation of the proposal
    '''

    def __init__(self,
                 system_model=None,
                 FIR_horizon=1
                 ):
        self._FIR_horizon = FIR_horizon
        self._system_model = None
        self.setSystemModel(system_model=system_model)

        self.resetObjAndCons()

        self._iop_problem = None
        self._iop_constraints = IOP_Cons_IOP()

    # overload plus and less than or equal operators as syntactic sugars
    def __add__(self, obj_or_cons):
        return self.addObjOrCons(obj_or_cons)

    def __lshift__(self, obj_or_cons_or_system):
        if isinstance(obj_or_cons_or_system, SystemModel):
            return self.setSystemModel(system_model=obj_or_cons_or_system)
        else:
            return self.setObjOrCons(obj_or_cons=obj_or_cons_or_system)

    def resetObjAndCons(self):
        self.resetObjectives()
        self.resetConstraints()

    def resetObjectives(self):
        self._objectives = []
        self._optimal_objective_value = float('inf')

    def resetConstraints(self):
        self._constraints = []

    def addObjOrCons(self, obj_or_cons):
        if isinstance(obj_or_cons, IOP_Constraint):
            self._constraints.append(obj_or_cons)
        elif isinstance(obj_or_cons, IOP_Objective):
            self._objectives.append(obj_or_cons)
        return self

    def setObjOrCons(self, obj_or_cons):
        if isinstance(obj_or_cons, IOP_Constraint):
            self._constraints = []
            self._constraints.append(obj_or_cons)
        elif isinstance(obj_or_cons, IOP_Objective):
            self._objectives = []
            self._objectives.append(obj_or_cons)
        return self

    def getOptimalObjectiveValue(self):
        return self._optimal_objective_value

    def get_IOP_Problem(self):
        return self._iop_problem

    def sanityCheck(self):
        # we can extend the algorithm to work for non-state-feedback SLS
        # if not self._state_feedback:
        #    return self.errorMessage('Only support state-feedback case for now.')

        if self._system_model is None:
            return self.errorMessage('The system is not yet assigned.')
        if not isinstance(self._system_model, LTI_FIR_System):
            return self.errorMessage('The system must be LTI FIR.')
        if not isinstance(self._FIR_horizon, int):
            return self.errorMessage('FIR horizon must be integer.')
        if self._FIR_horizon < 1:
            return self.errorMessage('FIR horizon must be at least 1.')

        return True

    def synthesizeControllerModel(self):
        self._optimal_objective_value = float('inf')
        if not self.sanityCheck():
            # simple sanity check
            return None

        Ny = self._system_model._Ny
        Nu = self._system_model._Nu

        controller = IOP_FIR_Controller(
            Ny=Ny, Nu=Nu,
            FIR_horizon=self._FIR_horizon
        )

        total = self._FIR_horizon + 1

        # initialize the variables
        self._X = [None] * total
        self._W = [None] * total
        self._Y = [None] * total
        self._Z = [None] * total
        for tau in range(total):
            self._X[tau] = cp.Variable(shape=(Ny, Ny))
            self._W[tau] = cp.Variable(shape=(Ny, Nu))
            self._Y[tau] = cp.Variable(shape=(Nu, Ny))
            self._Z[tau] = cp.Variable(shape=(Nu, Nu))

        # objective
        objective_value = 0
        for obj in self._objectives:
            objective_value = obj.addObjectiveValue(
                iop=self,
                objective_value=objective_value
            )

        # add IOP constraints
        constraints = self._iop_constraints.addConstraints(iop=self)

        # the constraints might also introduce additional terms at the objective
        for cons in self._constraints:
            objective_value = cons.addObjectiveValue(
                iop=self,
                objective_value=objective_value
            )
            constraints = cons.addConstraints(
                iop=self,
                constraints=constraints
            )

        # obtain results and put into controller
        self._iop_problem = cp.Problem(cp.Minimize(objective_value), constraints)
        self._iop_problem.solve()

        if self._iop_problem.status == 'infeasible':
            self.warningMessage('IOP problem infeasible')
            return None
        elif self._iop_problem.status == 'unbounded':
            self.warningMessage('IOP problem unbounded')
            return None
        else:
            # save the solved problem for the users to examine if needed
            self._optimal_objective_value = self._iop_problem.value
            controller._X = [None] * total
            controller._W = [None] * total
            controller._Y = [None] * total
            controller._Z = [None] * total
            for tau in range(total):
                controller._X[tau] = self._X[tau].value
                controller._W[tau] = self._W[tau].value
                controller._Y[tau] = self._Y[tau].value
                controller._Z[tau] = self._Z[tau].value

        controller.initialize()
        return controller
