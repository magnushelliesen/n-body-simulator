"""
By: Magnus Kvåle Helliesen
"""

import numpy as np
from math import sqrt
from typing import Tuple

# Let's make a class holding properties of a celestial body
class Body:
    """
    Class representing a celestial body.

    Parameters:
    - position_0 (tuple): Initial position of the body in the form (x, y, z).
    - velocity_0 (tuple): Initial velocity of the body in the form (vx, vy, vz).
    - mass (float): Mass of the body.

    Attributes:
    - position_0 (tuple): Initial position.
    - velocity_0 (tuple): Initial velocity.
    - mass (float): Mass of the body.
    - dim (int): Dimension of the space in which the body is embedded.
    - position_t (tuple): Current position.
    - velocity_t (tuple): Current velocity.
    """

    def __init__(self,
                 position_0: Tuple[float, ...],
                 velocity_0: Tuple[float, ...],
                 mass: float):
        """
        Initialize a celestial body.

        Parameters:
        - position_0 (tuple): Initial position.
        - velocity_0 (tuple): Initial velocity.
        - mass (float): Mass of the body.
        """

        # Cheking for type errors
        if isinstance(position_0, tuple) is False:
            raise TypeError('position_0 must be tuple')
        if isinstance(velocity_0, tuple) is False:
            raise TypeError('velocity_0 must be tuple')
        
        if len(position_0) != len(velocity_0):
            raise IndexError('position_0 and velocity_0 must have same length')
        
        if all(isinstance(x, (float, int)) for x in position_0) is False:
            raise TypeError('all values of position_0 must be float or int')
        if all(isinstance(x, (float, int)) for x in velocity_0) is False:
            raise TypeError('all values of velocity_0 must be int')

        if isinstance(mass, (float, int)) is False:
            raise TypeError('mass must be float or int')

        # Setting attributes that cannot be changed
        self._position_0 = position_0
        self._velocity_0 = velocity_0
        self._mass = mass
        self._dim = len(position_0)

        # Setting attributes that can be changed
        self._position_t = position_0
        self._velocity_t = velocity_0

    @property
    def position_0(self):
        return self._position_0

    @property
    def velocity_0(self):
        return self._velocity_0

    @property
    def mass(self):
        return self._mass

    @property
    def dim(self):
        return self._dim

    @property
    def position_t(self):
        return self._position_t

    @property
    def velocity_t(self):
        return self._velocity_t

    @position_t.setter
    def position_t(self, value):
        if isinstance(value, tuple) is False:
            raise TypeError('position must be tuple')
        if all(isinstance(x, (float, int)) for x in value) is False:
            raise ValueError('all values of position must be float or int')
        self._position_t = value

    @velocity_t.setter
    def velocity_t(self, value):
        if isinstance(value, tuple) is False:
            raise TypeError('velocity must be tuple')
        if all(isinstance(x, (float, int)) for x in value) is False:
            raise ValueError('all values of velocity must be float or int')
        self._velocity_t = value

    def __str__(self):
        return f'Body with mass {self.mass} and initial position {self.position_0} and velocity {self.velocity_0}'

    def __repr__(self):
        return self.__str__()


# Let's make a class that holds a bunch of celestial bodies
class System:
    """
    Class representing a system of celestial bodies.

    Class attributes:
    - G (float): Gravitational constant.
    - dim (int): Dimension of the system.
    - N (int): Number of time steps in simulation.
    - delta_t (float): Time step size.

    Parameters:
    - bodies (Body): A variable number of Body objects representing the celestial bodies in the system.

    Attributes:
    - G (float): Gravitational constant.
    - bodies (dict): Dictionary of Body objects in the system.
    - t (numpy.ndarray): Time array for simulation results.
    - r (dict): Dictionary of position arrays for each body in the system.
    """

    G = None
    dim = None
    N = None
    delta_t = None

    def __init__(self, *bodies: Body):
        """
        Initialize a system of celestial bodies.

        Parameters:
        - bodies (Body): A variable number of Body objects representing the celestial bodies in the system.
        """

        # Cheking for type errors
        if all(isinstance(x, Body) for x in bodies) is False:
            raise TypeError('all bodies must be Body object')

        if all(x.dim==System.dim for x in bodies) is False:
            raise TypeError(f'all bodies must be have dimension {System.dim}')

        # Setting attributes that cannot be changed
        self._bodies = {i: body for i, body in enumerate(bodies)}

        # Setting simulation attributes that may be changed by simulation method
        self._t, self._r = None, None

    @property
    def G(self):
        return self._G

    @property
    def bodies(self):
        return self._bodies

    @property
    def t(self):
        return self._t

    @property
    def r(self):
        return self._r

    def __str__(self):
        return f'System in {System.dim} dimension with {len(self)} bodies'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.bodies)

    # Method that simulates object trajectories
    def simulate(self):
        """
        Simulate the trajectories of celestial bodies in the system.

        This method performs a simulation of the celestial bodies' trajectories over a specified number of time steps and stores the simulation results.

        Raises:
        - TypeError: If G is not a float or int, N is not an int, or delta_t is not a float.

        Results:
        - The simulation results are stored in the `t` (time) and `r` (position) attributes of the System object.

        Returns:
        None
        """

        # Checking for type errors
        if isinstance(System.G, (float, int)) is False:
            raise TypeError('G must be float or int')

        if isinstance(System.N, int) is False:
            raise TypeError('System.N must be int')

        if isinstance(System.delta_t, float) is False:
            raise TypeError('System.delta_t must be float')

        # Setting up for simulation
        t = np.zeros(System.N+1)
        v, r = {}, {}
        for i, body in self.bodies.items():
            v[i] = np.zeros((System.N+1, System.dim))
            r[i] = np.zeros((System.N+1, System.dim))
            v[i][0] = body.velocity_0
            r[i][0] = body.position_0
            body.position_t = body.position_0
            body.velocity_t = body.velocity_0

        # Simulating
        for n in range(System.N):
            t[n+1] = System.delta_t+t[n]
            for i, body in self.bodies.items():
                # Calculate net forces on body from every other body
                F = np.zeros(System.dim)
                for other_body in [val for key, val in self.bodies.items() if key != i]:
                    r_ij = np.array(body.position_t)-np.array(other_body.position_t)
                    r_ij_norm = np.linalg.norm(r_ij)
                    r_ij_hat = r_ij/r_ij_norm
                    F -= (System.G*body.mass*other_body.mass/r_ij_norm**2)*r_ij_hat

                v[i][n+1] = v[i][n]+System.delta_t*F/body.mass
                r[i][n+1] = r[i][n]+System.delta_t*v[i][n]

            # Update position and velocity in body instances
            for i, body in self.bodies.items():
                body.position_t = tuple(r[i][n+1])
                body.velocity_t = tuple(v[i][n+1])

        # Storing simulation results
        self._t, self._r = t, r
