from typing import Tuple
import numpy as np
import cv2

"""
1. Create N particles.
2. Move the particles according to the measurement + noise --> Prediction
3. Compare the prediction with the measurement. Add higher weight to the particle close to the measurement and filter out the lower weight particles. 
4. Generate a new set of N particles (same number as original) by sampling proportionally to particle weights. Particles with higher weight are more likely to survive; low-weight particles are discarded. After resampling, the particle cloud naturally becomes denser around likely positions of the vehicle.
5. Can be extended to multiple sensor by vectorizing
"""

class ParticleFilter:
    def __init__(self, map_array: np.ndarray, num_particles: int = 3000) -> None:
        """
        Particle filter for robot localization on a 2D map.

        Args:
            map_array (np.ndarray): Grayscale map (2D numpy array).
            num_particles (int): Number of particles in the filter.
        """
        self.map: np.ndarray = map_array
        self.HEIGHT, self.WIDTH = self.map.shape

        # Robot initial state
        self.rx: float = self.WIDTH / 4
        self.ry: float = self.HEIGHT / 4
        self.rtheta: float = 0.0

        # Particle cloud
        self.NUM_PARTICLES: int = num_particles
        self.particles: np.ndarray = self.init_particles()

        # Noise parameters
        self.SIGMA_STEP: float = 0.5
        self.SIGMA_TURN: float = np.radians(5)
        self.SIGMA_POS: float = 2
        self.SIGMA_TURN_P: float = np.radians(10)
        self.SIGMA_SENSOR: float = 2

        # Control parameters
        self.STEP: int = 5
        self.TURN: float = np.radians(25)

    # ---------------------------------------------------
    # Initialization
    # ---------------------------------------------------
    def init_particles(self) -> np.ndarray:
        """
        Initialize random particles across the map.

        Returns:
            np.ndarray: Array of shape (N, 3) with [x, y, theta] for each particle.
        """
        particles = np.random.rand(self.NUM_PARTICLES, 3)
        particles *= np.array((self.WIDTH, self.HEIGHT, np.radians(360)))
        return particles

    # ---------------------------------------------------
    # Robot motion
    # ---------------------------------------------------
    def move_robot(self, fwd: float, turn: float) -> Tuple[float, float, float]:
        """
        Move the robot with Gaussian noise.

        Args:
            fwd (float): Forward movement step.
            turn (float): Turn angle (radians).

        Returns:
            Tuple[float, float, float]: Updated (x, y, theta) of the robot.
        """
        fwd_noisy = np.random.normal(fwd, self.SIGMA_STEP, 1)
        self.rx += fwd_noisy * np.cos(self.rtheta)
        self.ry += fwd_noisy * np.sin(self.rtheta)

        turn_noisy = np.random.normal(turn, self.SIGMA_TURN, 1)
        self.rtheta += turn_noisy

        return self.rx, self.ry, self.rtheta

    # ---------------------------------------------------
    # Particle motion (prediction)
    # ---------------------------------------------------
    def move_particles(self, fwd: float, turn: float) -> None:
        """
        Move all particles deterministically (prediction step).

        Args:
            fwd (float): Forward movement.
            turn (float): Turn angle (radians).
        """
        self.particles[:, 0] += fwd * np.cos(self.particles[:, 2])
        self.particles[:, 1] += fwd * np.sin(self.particles[:, 2])
        self.particles[:, 2] += turn

        self.particles[:, 0] = np.clip(self.particles[:, 0], 0.0, self.WIDTH - 1)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0.0, self.HEIGHT - 1)

    # ---------------------------------------------------
    # Sensor model
    # ---------------------------------------------------
    def sense(self, x: float, y: float, noisy: bool = False) -> float:
        """
        Simulate sensor measurement at a location.

        Args:
            x (float): X-coordinate.
            y (float): Y-coordinate.
            noisy (bool): If True, add Gaussian noise.

        Returns:
            float: Sensor reading (map intensity).
        """
        x = int(x)
        y = int(y)
        if noisy:
            return float(np.random.normal(self.map[y, x], self.SIGMA_SENSOR, 1))
        return float(self.map[y, x])

    # ---------------------------------------------------
    # Weight computation (measurement update)
    # ---------------------------------------------------
    def compute_weights(self, robot_sensor: float) -> np.ndarray:
        """
        Compute weights for all particles based on sensor measurement.

        Args:
            robot_sensor (float): Sensor reading from the real robot.

        Returns:
            np.ndarray: Array of weights for each particle.
        """
        errors = np.zeros(self.NUM_PARTICLES)
        for i in range(self.NUM_PARTICLES):
            elevation = self.sense(self.particles[i, 0], self.particles[i, 1], False)
            errors[i] = abs(robot_sensor - elevation)

        weights = np.max(errors) - errors
        # Zero out edge particles
        weights[
            (self.particles[:, 0] == 0) |
            (self.particles[:, 0] == self.WIDTH - 1) |
            (self.particles[:, 1] == 0) |
            (self.particles[:, 1] == self.HEIGHT - 1)
        ] = 0.0

        return weights ** 3

    # ---------------------------------------------------
    # Resampling
    # ---------------------------------------------------
    def resample(self, weights: np.ndarray) -> None:
        """
        Resample particles based on their weights.

        Args:
            weights (np.ndarray): Normalized weights for each particle.
        """
        probs = weights / np.sum(weights)
        new_index = np.random.choice(self.NUM_PARTICLES, size=self.NUM_PARTICLES, p=probs)
        self.particles = self.particles[new_index, :]

    # ---------------------------------------------------
    # Add noise to particles
    # ---------------------------------------------------
    def add_noise(self) -> None:
        """
        Add Gaussian noise to particle positions and angles.
        """
        noise = np.concatenate((
            np.random.normal(0, self.SIGMA_POS, (self.NUM_PARTICLES, 1)),
            np.random.normal(0, self.SIGMA_POS, (self.NUM_PARTICLES, 1)),
            np.random.normal(0, self.SIGMA_TURN_P, (self.NUM_PARTICLES, 1)),
        ), axis=1)
        self.particles += noise

    # ---------------------------------------------------
    # Display
    # ---------------------------------------------------
    def display(self) -> None:
        """
        Display the map, particles, robot position, and best guess.
        """
        lmap = cv2.cvtColor(self.map, cv2.COLOR_GRAY2BGR)

        # Particles
        for i in range(self.NUM_PARTICLES):
            cv2.circle(lmap,
                       (int(self.particles[i, 0]), int(self.particles[i, 1])),
                       1, (255, 0, 0), 1)

        # Robot
        cv2.circle(lmap, (int(self.rx), int(self.ry)), 5, (0, 255, 0), 10)

        # Best guess (mean of particles)
        px = np.mean(self.particles[:, 0])
        py = np.mean(self.particles[:, 1])
        cv2.circle(lmap, (int(px), int(py)), 5, (0, 0, 255), 5)

        cv2.imshow('map', lmap)

    # ---------------------------------------------------
    # Main step
    # ---------------------------------------------------
    def step(self, fwd: float, turn: float) -> None:
        """
        Perform one cycle of motion + measurement update.

        Args:
            fwd (float): Forward motion.
            turn (float): Turn angle (radians).
        """
        self.move_robot(fwd, turn)
        self.move_particles(fwd, turn)

        if fwd != 0:
            robot_sensor = self.sense(self.rx, self.ry, noisy=True)
            weights = self.compute_weights(robot_sensor)
            self.resample(weights)
            self.add_noise()

        self.display()

    # ---------------------------------------------------
    # Estimate current best state
    # ---------------------------------------------------
    def estimate_state(self) -> Tuple[float, float, float]:
        """
        Estimate the current best guess of robot position and orientation.

        Returns:
            Tuple[float, float, float]: (x, y, theta) estimated from particle cloud.
        """
        x = float(np.mean(self.particles[:, 0]))
        y = float(np.mean(self.particles[:, 1]))
        theta = float(np.mean(self.particles[:, 2]))
        return x, y, theta

if __name__ == "__main__":
    # Load map outside the class
    map_img = cv2.imread("map.png", 0)
    if map_img is None:
        raise ValueError("map.png not found")

    pf = ParticleFilter(map_img, num_particles=3000)

    while True:
        pf.display()

        key = cv2.waitKey(0)
        if key == ord('w'):   # forward
            fwd, turn = pf.STEP, 0
        elif key == ord('a'): # turn left
            fwd, turn = 0, -pf.TURN
        elif key == ord('d'): # turn right
            fwd, turn = 0, pf.TURN
        elif key == ord('q'): # quit
            break
        else:
            fwd, turn = 0, 0

        pf.step(fwd, turn)

        # Example: print best estimate
        print("Estimate:", pf.estimate_state())

    cv2.destroyAllWindows()
