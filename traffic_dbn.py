#for part 4 depends on python version 
#from __future__ import annotation 

#for tewzi3 l data bel functions
# @dataclass shortcut for glodal and coupling paramaters 
from dataclasses import dataclass

# for arrays and matrices 
import numpy as np 

#block_diag place matrices along diagonal
#from scipy.linalg import block_diag 
from scipy.linalg import block_diag

#vehicale model
import ssm1 as ssm

VehicleSSM = ssm.VehicleSSM

N_STATE = len(ssm.STATE_NAMES) #11 vehicle states
N_OBS =len(ssm.OBS_NAMES)      #13 measurements /sensors

# Positions of the states we use inside the 11-number vector
#we will choose 6 only cz no need for others they are handled by ssm 
IDX_PX = ssm.STATE_NAMES.index("px") # =0 east position [m]
IDX_PY = ssm.STATE_NAMES.index("py") # =1 north position [m] 
IDX_PSI = ssm.STATE_NAMES.index("psi") # =2 heading angle [rad]
IDX_VX =ssm.STATE_NAMES.index("vx") # =3 longitudinal or forward velocity [m/s]
IDX_VY =ssm.STATE_NAMES.index("vy") # =4 lateral or sideways velocity [m/s]
IDX_DELTA =ssm.STATE_NAMES.index("delta") # =10 steering angle [rad]


#GLOBAL TRAFFIC STATE G 
#Layer A

GLOBAL_NAMES = ["density_veh_km", "flow_veh_h", "mean_speed_mps", "congestion"] #4 global traffic states

#Number of global states
N_GLOBAL = len(GLOBAL_NAMES) #4 global traffic states

#Position inside G of the 2 values the vehicles react to.
IDX_MEAN_SPEED = GLOBAL_NAMES.index("mean_speed_mps") # =2 mean speed [m/s]
IDX_CONGESTION = GLOBAL_NAMES.index("congestion") # =3 congestion [0-1]



@dataclass
class GlobalParams:
    """Layer A: how the global traffic states G changes over time."""

    #Small tau = G reacts fast to the car. Large tau = G reacts slowly to the car.
    #tau(TheGreekletterTau): time constant for the global traffic states G [s]
    tau: float = 2.0 #[seconds]

    #Speed of completely free traffic. Used to compute congestion.
    v_free: float = 25.0 #[m/s]

    #Distance between vehicles only 1 V2V
    fallback_spacing : float = 500.0 #[m]

    #Random noise added to the global traffic states G.
    noise_std: tuple = (2.0, 50.0, 0.2, 0.02)


@dataclass
class CouplingParams:
    #How the global state and the neighbours push on each car.

    #macro: a car is pulled toward the average speed of thraffic.
    gain_macro: float = 0.15 #[1/s]

    #The weight W_ij: how much does car j influence car i.
    range_m: float = 40.0 #[m]how far ahead a car still matters   
    lane_sigma: float = 2.0 #[m] sideways tolerance for "same line"
    ahead_softness: float = 2.0 #[m] ssmoothness of the "is ahead switch"

    #the reaction phi_ij: a simple car-following role
    gain_speed: float = 0.5 #[1/s] react to the speed difference
    gain_gap: float = 0.05 #[1/s^2] react to the gap error
    standard_gap: float = 5.0 #[m] the gap a driver wants at zero speed
    time_headway: float = 1.2 #[s] extra gap wanted per m/s of speed

    # The total push
    a_max: float = 3.0 #[m/s^2] maximum acceleration


#PART 3: GRAPH STRUCTURE (THE ARROW OF DBN)
#THE NODES NAMES ARE:
#G[t] = global traffic states at time t
#x1[t] = state of vehicle 1 at time t
#y1[t] = observation(sensors) of vehicle 1 at time t
#u1[t] = driver input of vehicle 1 at time t

# an arrow is written as pair: (parent, child) = (from, to)

def two_slice_edges(n_vehicles):
    #Return the list of all arrows betweentime t-1 and time t.
    edges = [("G[t-1]", "G[t]")]  # Global state G[t-1] influences G[t]
    # Global state G[t] influences all vehicle states x_i[t]

    for i in range(1,n_vehicles +1):
        #f"x{i}[t-1]" is an f-string with i = 2 it becomes the text "x2[t-1]"
        edges+=[
            (f"x{i}[t-1]",f"G[t]"),  # car -> global (G is built from the cars)
            ("G[t]", f"x{i}[t]"),  # global -> car (the car reacts to the traffic)
            (f"x{i}[t-1]", f"x{i}[t]"),  # The car remembers its own past
            (f"u{i}[t-1]", f"x{i}[t]"),  # driver inputs move the car
            (f"x{i}[t]", f"y{i}[t]"),  # The sensors measure the car's state
            (f"u{i}[t]", f"y{i}[t]"),  # inputs also influence or reach the sensors (e.g. brake lights)
        ]

        #Arrows from every OTHER car j into car i (the interaction W_ij)
        #builds a list. "if j != i" skips the car itself.
        edges += [
            (f"x{j}[t-1]", f"x{i}[t]") 
            for j in range(1, n_vehicles + 1) 
            if j != i
        ]
    return edges


def is_acyclic(edges):
    #Check if the graph is acyclic (no loops).
    #Keep node have no parents left.
    
    #The set of all nodes in the graph names
    nodes = {name for edge in edges for name in edge}

    # in_degree[node]: how many parents does each node have 
    #dict.fromkeys(nodes,0) makes dictionary: every node starts at 0.
    in_degree = dict.fromkeys(nodes, 0)

    # Calculate in-degrees
    for _, child in edges:  # "_" means "I don't care about the parent name or need the value"
        in_degree[child] += 1 #count one more parent for this child 

    #nodes with no parents left (in_degree = 0) can be removed first
    ready = [name for name in nodes if in_degree[name] == 0]

    removed = 0
    while ready:      # repeat while the list is note empty
        node = ready.pop()  # take one node out of the list
        removed += 1
        # Removing this node frees its childrem from one parent of each
        for parent, child in edges:
            if parent == node:
                in_degree[child] -= 1
                if in_degree[child] == 0: #no parents left:it can go too
                    ready.append(child) 

    #if every node was removed, there is no loop.
    return removed == len(nodes)


#PART 4: THE TRAFFIC DYNAMIC BAYESIAN NETWORK (DBN) CLASS (skeleton)


#The model keep ONE  long vector for the whole system
# ex: z= [ G (4 numbers) | car 1 (11) | car 2 (11) | car 3 (11) ]
#so with 3 cars: 4 + 3*11 = 37 numbers in the vector z

class TrafficDBN:
    #Joint model of the global traffic states G and N VEHICALES 

    def __init__ (self,vehicles,glob=None, coupling=None):
        #vehicles: list of VehicleSSM objects, one for each car
        self.vehicles = list(vehicles) # the list of VehicleSSM objects
        self.N = len(self.vehicles) #how many cars
        self.dt = self.vehicles[0].dt  # time step in seconds (from the SSM)

        #global traffic states G
        self.glob = glob or GlobalParams()

        #how the global state and the neighbours push on each car
        self.cpl = coupling or CouplingParams()

        # size of the joint vector z: G plus 11 numbers per car 
        self.dim = N_GLOBAL + self.N * N_STATE


        #NOTES:
        #noise matrices (used by thr simulation and the filter)
        # Process noise of G: a diagonal matrix with with the variance (std squared)
        #of each G variable, scaled by the time step.
        q_global = np.diag(np.square(self.glob.noise_std)) * self.dt

        # Joint process noise Q: blocks along the daigonal, one for G, and one per car.
        #self.course_rows = ssm.IDX_COURSE + N_OBS * np.arange(self.N)

# i have problem in the line before so i have fix it from ai with the 3 lines bellow 
                # Joint process noise Q: blocks along the diagonal, one for G, and one per car.
        self.Q = block_diag(q_global, *[v.Q * self.dt for v in self.vehicles])

        # Joint measurement noise R: one block per car (G has no sensors).
        self.R = block_diag(*[v.R for v in self.vehicles])

        # Which rows of the measurement vector are ANGLES (the GPS course).
        self.course_rows = ssm.IDX_COURSE + N_OBS * np.arange(self.N)



        #joint measurement noise R: one block per car (G has no sensor)
        self.R = block_diag(*[v.R for v in self.vehicles])



    #Moving between "G and a table of cars" and "one long vector z"

    def pack(self,G,X):
        #note: G(4,) and X (n,11) -> one vector z of length 4 + 11 N.
        #np.reval flattens the table X row by row; np.concatenate joins.
        return np.concatenate([G, np.ravel(X)])   

    def unpack(self, z):
        # 1 vector z -> G(4,) and X(V,11).
        G = z[:N_GLOBAL]  # the first 4 numbers 
        X = z[N_GLOBAL:].reshape(self.N,N_STATE) # the reset, as a table
        return G, X

    def state_slice(self, i):
        # wheere car i (0,1,2,...) sites inside z,as a slice.
        start = N_GLOBAL + i * N_STATE
        return slice(start, start + N_STATE)

    """
    example:
       car 0 → positions 4 to 14
       car 1 → positions 15 to 25
       car 2 → positions 26 to 36
    """

# Part 5: Layerr A (Global Trafic State G)

def macro_summary(self, X):
    #Look at all cars and compute the 4 trafic numbers from them 
    # X:  table of car state,shape(N,11)
    #returns: array [density, flow, mean_speed, congestion]

    g = self.glob

    #mean speed, Ground speed of each car from its 2 speed component
    # X[;, IDX_VX] means "column IDX_V of every row" = all forward speed

    speeds = np.hypot(X[:, IDX_VX], X[:, IDX_VY])
    mean_speed + speeds.mean()  #average over all cars

    #Density: how close together are the cars?
    if self.N > 1:
        #positions of all cars, shape (N, 2): east and north
        pos = X[:, [IDX_PX, IDX_PY]]

#####################################










    def macro_summary(self, X):
        g = self.glob

        # --- mean speed ---------------------------------------------------
        # Ground speed of each car from its two speed components.
        # X[:, IDX_VX] means "column IDX_VX of every row" = all forward speeds.
        speeds = np.hypot(X[:, IDX_VX], X[:, IDX_VY])
        mean_speed = speeds.mean()                      # average over all cars

        # --- density: how close together are the cars? -------------------
        if self.N > 1:
            # positions of all cars, shape (N, 2): east and north
            pos = X[:, [IDX_PX, IDX_PY]]

            # Distance between EVERY pair of cars, as an N x N table.
            # pos[:, None, :] has shape (N, 1, 2) and pos[None, :, :] has shape
            # (1, N, 2). Subtracting them pairs every car with every other car
            # ("broadcasting"). norm(..., axis=2) turns each difference into a
            # distance.
            dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)

            # A car is at distance 0 from itself. Set the diagonal to infinity
            # so a car is never counted as its own nearest neighbour.
            np.fill_diagonal(dist, np.inf)

            # For each car, the distance to its nearest neighbour (min over each
            # row), then the average over all cars.
            spacing = dist.min(axis=1).mean()
        else:
            # with one car there is no neighbour to measure
            spacing = g.fallback_spacing

        # max(spacing, 1.0) avoids dividing by almost zero if two cars overlap.
        density = 1000.0 / max(spacing, 1.0)            # cars per kilometre

        # --- flow: the classic relation flow = density x speed ------------
        # The 3.6 converts m/s into km/h, so the flow comes out in cars per hour.
        flow = density * mean_speed * 3.6

        # --- congestion: 0 = free road, 1 = stopped traffic ---------------
        # np.clip keeps the value between 0 and 1.
        congestion = np.clip(1.0 - mean_speed / g.v_free, 0.0, 1.0)

        return np.array([density, flow, mean_speed, congestion])

    def global_step(self, G, X):
        """One time step of G (without noise).

        G moves a small part of the way toward what the cars show right now.
        The part is dt / tau.
        """
        target = self.macro_summary(X)                  # what the cars say now
        return G + (self.dt / self.glob.tau) * (target - G)
 
