from .components import SLS_Constraint
import cvxpy as cp
import numpy as np

'''
To create a new SLS constraint, inherit the following base function and customize the specified methods.

class SLS_Constraint:
    def addConstraints(self, sls, constraints):
        return constraints
'''


class SLS_Cons_SLS(SLS_Constraint):
    '''
    The discrete-time SLS constrains
    '''

    def __init__(self, state_feedback=False):
        self._state_feedback = state_feedback

    def addConstraints(self, sls, constraints=None):
        '''
        state-feedback constraints:
        [ zI-A, -B2 ][ Phi_x ] = I
                     [ Phi_u ]

        output-feedback constriants:
        [ zI-A, -B2 ][ Phi_xx Phi_xy ] = [ I 0 ]
                     [ Phi_ux Phi_uy ]
        [ Phi_xx Phi_xy ][ zI-A ] = [ I ]
        [ Phi_ux Phi_uy ][ -C2  ]   [ 0 ]
        '''
        if constraints is None:
            # avoid using empty list as default arguments, which can cause unwanted issue
            constraints = []

        Nx = sls._system_model._Nx
        Nu = sls._system_model._Nu

        # sls constraints
        # the below constraints work for output-feedback case as well because
        # sls._Phi_x = sls._Phi_xx and sls._Phi_u = sls._Phi_ux
        # Phi_x, Phi_u are in z^{-1} RH_{\inf}. Therefore, Phi_x[0] = 0, Phi_u = 0
        constraints += [sls._Phi_x[0] == np.zeros([Nx, Nx])]
        constraints += [sls._Phi_u[0] == np.zeros([Nu, Nx])]
        constraints += [sls._Phi_x[1] == np.eye(Nx)]
        constraints += [sls._Phi_u[sls._FIR_horizon] == np.zeros([Nu, Nx])]
        # constraints += [
        #     (sls._system_model._A @ sls._Phi_x[sls._FIR_horizon] +
        #      sls._system_model._B2 @ sls._Phi_u[sls._FIR_horizon]) == np.zeros([Nx, Nx])
        # ]
        for tau in range(1, sls._FIR_horizon):
            constraints += [
                sls._Phi_x[tau + 1] == (
                        sls._system_model._A[tau - 1] @ sls._Phi_x[tau] +
                        sls._system_model._B2[tau - 1] @ sls._Phi_u[tau]
                )
            ]

        if not self._state_feedback:
            Ny = sls._system_model._Ny

            # Phi_xx, Phi_ux, and Phi_xy are in z^{-1} RH_{\inf}.
            # Phi_uy is in RH_{\inf} instead of z^{-1} RH_{\inf}.
            constraints += [sls._Phi_xy[0] == np.zeros([Nx, Ny])]

            # output-feedback constraints
            constraints += [
                sls._Phi_xy[1] == sls._system_model._B2 @ sls._Phi_uy[0]
            ]
            constraints += [
                (sls._system_model._A @ sls._Phi_xy[sls._FIR_horizon] +
                 sls._system_model._B2 @ sls._Phi_uy[sls._FIR_horizon]) == np.zeros([Nx, Ny])
            ]
            constraints += [
                (sls._Phi_xx[sls._FIR_horizon] @ sls._system_model._A +
                 sls._Phi_xy[sls._FIR_horizon] @ sls._system_model._C2) == np.zeros([Nx, Nx])
            ]
            constraints += [
                sls._Phi_ux[1] == sls._Phi_uy[0] @ sls._system_model._C2
            ]
            constraints += [
                (sls._Phi_ux[sls._FIR_horizon] @ sls._system_model._A +
                 sls._Phi_uy[sls._FIR_horizon] @ sls._system_model._C2) == np.zeros([Nu, Nx])
            ]
            for tau in range(1, sls._FIR_horizon):
                constraints += [
                    sls._Phi_xy[tau + 1] == (
                            sls._system_model._A @ sls._Phi_xy[tau] +
                            sls._system_model._B2 @ sls._Phi_uy[tau]
                    )
                ]

                constraints += [
                    sls._Phi_xx[tau + 1] == (
                            sls._Phi_xx[tau] @ sls._system_model._A +
                            sls._Phi_xy[tau] @ sls._system_model._C2
                    )
                ]

                constraints += [
                    sls._Phi_ux[tau + 1] == (
                            sls._Phi_ux[tau] @ sls._system_model._A +
                            sls._Phi_uy[tau] @ sls._system_model._C2
                    )
                ]
        return constraints


class SLS_Cons_State(SLS_Constraint):
    """
    State constraints
    """

    def __init__(self, state_feedback=False, maximum_state=None):
        super().__init__()
        self._state_feedback = state_feedback
        self.max_state = maximum_state

    def addConstraints(self, sls, constraints=None):
        """
        State constraints: |Phi_x x_0| <= max_state
        """
        if constraints is None:
            # avoid using empty list as default arguments, which can cause unwanted issue
            constraints = []

        Nx = sls._system_model._Nx
        if self.max_state is None or self.max_state.shape[0] != Nx:
            raise Exception("No maximum value or incorrect dimensions provided for state constraint")

        for tau in range(2, sls._FIR_horizon):  # Phi_x[0] = zero, Phi_x[1] = eye. No need to check those.
            constraints += [
                sls._Phi_x[tau] @ sls._system_model._x0 <= self.max_state,
                -sls._Phi_x[tau] @ sls._system_model._x0 <= self.max_state
            ]

        return constraints


class SLS_Cons_Input(SLS_Constraint):
    """
    Input constraints
    """

    def __init__(self, state_feedback=False, maximum_input=None):
        super().__init__()
        self._state_feedback = state_feedback
        self.max_input = maximum_input

    def addConstraints(self, sls, constraints=None):
        """
        Input constraints: |Phi_u x_0| <= max_input
        """
        if constraints is None:
            # avoid using empty list as default arguments, which can cause unwanted issue
            constraints = []

        Nu = sls._system_model._Nu
        if self.max_input is None or self.max_input.shape[0] != Nu:
            raise Exception(f"No maximum value or incorrect dimensions provided for input constraint.")

        for tau in range(1, sls._FIR_horizon-1):  # Phi_u[0] = zero, Phi_u[t_FIR] = zero. No need to check those.
            constraints += [
                sls._Phi_u[tau] @ sls._system_model._x0 <= self.max_input,
                -sls._Phi_u[tau] @ sls._system_model._x0 <= self.max_input
            ]

        return constraints


class SLS_Cons_dLocalized(SLS_Constraint):
    def __init__(self,
                 base=None,
                 act_delay=0, comm_speed=1, d=1
                 ):
        '''
        act_delay: actuation delay
        comm_speed: communication speed
        d: for d-localized
        '''
        if isinstance(base, SLS_Cons_dLocalized):
            self._act_delay = base._act_delay
            self._comm_speed = base._comm_speed
            self._d = base._d
        else:
            self._act_delay = act_delay
            self._comm_speed = comm_speed
            self._d = d

    def addConstraints(self, sls, constraints):
        # localized constraints
        # get localized supports
        Phi_x = sls._Phi_x
        Phi_u = sls._Phi_u

        commsAdj = np.absolute(sls._system_model._A) > 0
        localityR = np.linalg.matrix_power(commsAdj, self._d - 1) > 0

        # performance helpers
        absB2T = np.absolute(sls._system_model._B2).T

        # adjacency matrix for available information 
        info_adj = np.eye(sls._system_model._Nx) > 0
        transmission_time = -self._comm_speed * self._act_delay
        for t in range(1, sls._FIR_horizon + 1):
            transmission_time += self._comm_speed
            while transmission_time >= 1:
                transmission_time -= 1
                info_adj = np.dot(info_adj, commsAdj)

            support_x = np.logical_and(info_adj, localityR)
            support_u = np.dot(absB2T, support_x) > 0

            # shutdown those not in the support
            for ix, iy in np.ndindex(support_x.shape):
                if support_x[ix, iy] == False:
                    constraints += [Phi_x[t][ix, iy] == 0]

            for ix, iy in np.ndindex(support_u.shape):
                if support_u[ix, iy] == False:
                    constraints += [Phi_u[t][ix, iy] == 0]

        return constraints


class SLS_Cons_Robust(SLS_Constraint):
    '''
    Robust SLS (state-feedback) constraints
    '''

    def __init__(self,
                 gamma_coefficient=0
                 ):
        self._gamma_coefficient = gamma_coefficient

        # this matches the index and avoids the confusion
        self._Delta = []
        self._gamma = cp.Variable(1)

    def getStabilityMargin(self):
        return self._gamma.value

    def addObjectiveValue(self, sls, objective_value):
        '''
        introduce one more term: 
            gamma_coefficient * gamma
        to the objective
        '''
        self._objective_expression = self._gamma_coefficient * self._gamma

        return objective_value + self._objective_expression

    def addConstraints(self, sls, constraints):
        '''
        [ zI-A, -B2 ][ Phi_x ] = I + Delta
                     [ Phi_u ]
        || Delta ||_{E_1} <= gamma
        '''
        # reset constraints
        hat_Phi_x = sls._Phi_x
        hat_Phi_u = sls._Phi_u

        Nx = sls._system_model._Nx

        # recycle delta for better performance
        len_delta = len(self._Delta)
        if len_delta > 1:
            if (self._Delta[1].shape[0] != Nx) or (self._Delta[1].shape[1] != Nx):
                self._Delta = []
                len_delta = 0
        for t in range(len_delta, sls._FIR_horizon + 1):
            self._Delta.append(cp.Variable(shape=(Nx, Nx)))

        constraints = [hat_Phi_x[1] - self._Delta[0] == np.eye(Nx)]
        # move Delta to the left hand side to omit creating zero
        constraints += [
            - self._Delta[sls._FIR_horizon] == (
                    sls._system_model._A @ hat_Phi_x[sls._FIR_horizon]
                    + sls._system_model._B2 @ hat_Phi_u[sls._FIR_horizon]
            )
        ]

        for t in range(1, sls._FIR_horizon):
            constraints += [
                hat_Phi_x[t + 1] - self._Delta[t] == (
                        sls._system_model._A @ hat_Phi_x[t]
                        + sls._system_model._B2 @ hat_Phi_u[t]
                )
            ]

        # E_l (elementwise l1 norm) robustness
        constraints += [
            cp.norm(cp.bmat([self._Delta]), 'inf') <= self._gamma
        ]

        return constraints


class SLS_Cons_ApproxdLocalized(SLS_Cons_dLocalized, SLS_Cons_Robust):
    def __init__(self,
                 rob_coeff=0,
                 **kwargs
                 ):
        SLS_Cons_dLocalized.__init__(self, **kwargs)

        base = kwargs.get('base')
        if isinstance(base, SLS_Cons_ApproxdLocalized):
            self._rob_coeff = base._rob_coeff
        else:
            self._rob_coeff = rob_coeff

        SLS_Cons_Robust.__init__(self, gamma_coefficient=rob_coeff)

    def addObjectiveValue(self, sls, objective_value):
        return SLS_Cons_Robust.addObjectiveValue(self, sls, objective_value)

    def addConstraints(self, sls, constraints):
        constraints = SLS_Cons_Robust.addConstraints(self, sls, constraints)
        constraints = SLS_Cons_dLocalized.addConstraints(self, sls, constraints)

        return constraints
