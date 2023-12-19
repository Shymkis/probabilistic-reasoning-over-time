# Author    : Robert Geraghty
# Contact   : robertrgeraghty@gmail.com
# Date      : Feb 16, 2020

import random
import time
import numpy as np
import random
import time
import numpy as np

try:
    from CS5313_Localization_Env import maze
except:
    print(
        'Problem finding CS5313_Localization_Env.maze... Trying to "import maze" only...'
    )
    try:
        import maze

        print("Successfully imported maze")
    except Exception as ex:
        print("Could not import maze")
        print("----->LOOK HERE FOR EXCEPTION MESSAGE<-----")
        print(ex)
        print("----->LOOK HERE FOR EXCEPTION MESSAGE<-----")
try:
    from CS5313_Localization_Env import RobotLocalization as viz
except:
    print(
        'Problem finding CS5313_Localization_Env.RobotLocalization... Trying to "import RobotLocalization" only...'
    )
    try:
        import RobotLocalization as viz

        print("Successfully imported RobotLocalization")
    except Exception as ex:
        print("Could not import RobotLocalization")
        print("----->LOOK HERE FOR EXCEPTION MESSAGE<-----")
        print(ex)
        print("----->LOOK HERE FOR EXCEPTION MESSAGE<-----")
try:
    from CS5313_Localization_Env import localization_env as le
except:
    print(
        'Problem finding CS5313_Localization_Env.localization_env... Trying to "import localization_env" only...'
    )
    try:
        import localization_env as le

        print("Successfully imported localization_env")
    except Exception as ex:
        print("Could not import localization_env")
        print("----->LOOK HERE FOR EXCEPTION MESSAGE<-----")
        print(ex)
        print("----->LOOK HERE FOR EXCEPTION MESSAGE<-----")
from enum import Enum


# Change this to true to print out information on the robot location and heading
printouts = True
# Change this to true inorder to print out the map as a dataframe to console every time move() is called, as well as the Transition Tables to csv files named "heading.csv" and "location.csv". Won't do anything if printouts is false expect import pandas
df = False
if df:
    from pandas import DataFrame


class Directions(Enum):
    """An Enum containing the directions S, E, N, W, and St (stationary) and their respective (x, y) movement tuples. Ex. S = (0, 1) meaning down one row, and stationary in the columns."""

    S = (0, 1)
    E = (1, 0)
    N = (0, -1)
    W = (-1, 0)
    St = (0, 0)

    def get_ortho(self, value):
        """ Return the Direction Enums orthogonal to the given direction

        Arguements:\n
        value           -- The given direction for which the orthogonal directions will be based on.\n

        Returns:\n
        A list of directions orthogonal to the given direction.
        """
        if value in [self.N, self.S]:
            return [self.W, self.E]
        return [self.N, self.S]


class Headings(Enum):
    """An enum containing the headings S, E, N, W and their respective (x, y) movement tuples"""

    S = (0, 1)
    E = (1, 0)
    N = (0, -1)
    W = (-1, 0)

    def get_ortho(self, value):
        """ Return the Headings Enums orthogonal to the given heading

        Arguements:\n
        value           -- The given heading for which the orthogonal heading will be based on.\n

        Returns:\n
        A list of headings orthogonal to the given heading.
        """
        if value in [self.N, self.S]:
            return [self.W, self.E]
        return [self.N, self.S]


class Environment:
    """ An environment for testing a randomly moving robot around a maze.

    Important Class Variables\n
    map                     -- The map of the the maze. A 2d list of lists in the form list[x][y] where a value of 1 signifies there is a wall, 0 signifies the cell is traversable, and 'x' denotes the robot location.\n
    location_transitions     -- The table of transition probabilities for each cell. Format is [x][y][heading][direction] which will return the probabilities of moving the direction, given the robot's current x, y, and heading.\n
    heading_transitions     -- The table of transition probabilities for the headings given each cell. Format is [x][y][heading][heading] which will return the probabilities of each heading for the next time step given the robot's current x, y, and heading.\n
    robot_location          -- The current location of the robot, given as a tuple in the for (x, y).
    robot_heading           -- The current heading of the robot, given as a Headings enum.
    """

    def __init__(
        self,
        action_bias,
        observation_noise,
        action_noise,
        dimensions,
        seed=None,
        window_size=[750, 750],
    ):
        """Initializes the environment. The robot starts in a random traversable cell.

        Arguements:\n
        action_bias             -- Provides a bias for the robots actions. Positive values increase the likelihood of South and East movements, and negative favor North and West. (float in range -1-1)\n
        observation_noise       -- The probability that any given observation value will flip values erroneously. (float in range 0-1)\n
        action_noise            -- The probability that an action will move either direction perpendicular to the inteded direction. (float in range 0-1)\n
        dimensions              -- The dimensions of the map, given in the form (x,y). (tuple in range (1+, 1+))\n
        seed (optional)         -- The random seed value. (int) default=10\n
        window_size(optional)   -- The [x, y] size of the display. Default is [750, 750]. Should be the same aspect ratio as the maze to avoid strange looking graphics.

        Return:\n
        No return
        """
        # the pygame state
        self.running = True

        # Step counter
        self.steps = 0

        # save the bias, noise, and map sizze parameters
        self.action_bias = action_bias
        self.observation_noise = observation_noise
        self.action_noise = action_noise
        self.dimensions = dimensions

        # set the random seed and display it
        self.seed = seed if seed != None else random.randint(1, 10000)
        random.seed(self.seed)

        # creat the map and list of free cells
        self.map = maze.make_maze(dimensions[0], dimensions[1], seed)
        self.free_cells = [
            (x, y)
            for x in range(dimensions[0])
            for y in range(dimensions[1])
            if self.map[x][y] == 0
        ]

        # create the transistion table
        self.location_transitions = self.create_locations_table()
        self.headings_transitions = self.create_headings_table()

        if df:
            DataFrame(self.location_transitions).transpose().to_csv("location.csv")
            DataFrame(self.headings_transitions).transpose().to_csv("heading.csv")

        # set the robot location and print
        self.robot_location = self.free_cells[
            random.randint(0, len(self.free_cells) - 1)
        ]

        self.location_priors, self.heading_priors = self.compute_prior_probabilities()
        self.observation_tables = self.create_observation_tables()
        self.map[self.robot_location[0]][self.robot_location[1]] = "x"

        # Set the robot heading
        self.robot_heading = random.choice(
            [
                h
                for h in Headings
                if self.traversable(self.robot_location[0], self.robot_location[1], h)
            ]
        )

        # gen initial headings probs
        probs = {}
        # prob_sum = 0
        for h in le.Headings:
            # num = random.random()
            probs[h] = 1
        #     prob_sum += num
        # for h in le.Headings:
        #     probs[h] /= prob_sum

        # init viz
        self.window_size = window_size
        self.game = viz.Game()
        self.game.init_pygame(self.window_size)
        self.game.update(
            self.map,
            self.robot_location,
            self.robot_heading,
            [[0] * self.dimensions[1]] * self.dimensions[0],
            probs,
        )
        self.game.display()

        if printouts:
            print("Random seed:", self.seed)
            print("Robot starting location:", self.robot_location)
            print("Robot starting heading:", self.robot_heading)

            if df:
                print(DataFrame(self.map).transpose())

    def compute_prior_probabilities(self):
        location_priors = {}
        for cell in self.free_cells:
            location_priors[cell] = 1 / len(self.free_cells)

        heading_priors = {}
        for heading in Headings:
            heading_priors[heading] = 0
            for cell in self.free_cells:
                for heading2 in Headings:
                    heading_priors[heading] += self.headings_transitions[cell[0]][
                        cell[1]
                    ][heading2][heading]
            heading_priors[heading] /= len(self.free_cells) * 4

        return location_priors, heading_priors

    def random_dictionary_sample(self, probs):
        sample = random.random()
        prob_sum = 0
        for key in probs.keys():
            prob_sum += probs[key]
            if prob_sum > sample:
                return key

    def move(self):
        """Updates the robots heading and moves the robot to a new position based off of the transistion table and its current location and new heading.

        Return:\n
        A list of the observations modified by the observation noise, where 1 signifies a wall and 0 signifies an empty cell. The order of the list is [S, E, N, W]
        """

        # get the new location
        self.map[self.robot_location[0]][self.robot_location[1]] = 0
        probs = self.location_transitions[self.robot_location[0]][
            self.robot_location[1]
        ][self.robot_heading]

        direction = self.random_dictionary_sample(probs)

        self.robot_location = (
            self.robot_location[0] + direction.value[0],
            self.robot_location[1] + direction.value[1],
        )

        self.map[self.robot_location[0]][self.robot_location[1]] = "x"

        # Get the new heading
        h_probs = self.headings_transitions[self.robot_location[0]][
            self.robot_location[1]
        ][self.robot_heading]
        self.robot_heading = self.random_dictionary_sample(h_probs)

        # # get the new location
        # self.map[self.robot_location[0]][self.robot_location[1]] = 0
        # probs = self.location_transitions[self.robot_location[0]][
        #     self.robot_location[1]
        # ][self.robot_heading]

        self.steps += 1
        # return the new observation
        if printouts:
            print()
            print(
                "---------------------------Steps: "
                + str(self.steps)
                + " ---------------------------------"
            )
            print(self.robot_location)
            print(self.robot_heading)
            print(direction)
            if df:
                print(DataFrame(self.map).transpose())

        # if self.running:
        #     self.game.update(
        #         self.map,
        #         self.robot_location,
        #         self.robot_heading,
        #         location_probs,
        #         headings_probs,
        #     )
        #     self.running = self.game.display()
        # else:
        #     print("Pygame closed. Quiting...")
        #     self.game.quit()

        return self.observe()

    def update(self, location_probs, headings_probs):
        """Updates the visualizer to represent where your filtering method estimates the robot to be, and where it estimates the robot is heading.

        Arguments:\n
        location_probs: The probability of the robot being in any (x, y) cell in the map. Created from your project code. Format list[x][y] = float\n
        headings_probs: The probability of the robot's current heading being any given heading. Created from your project code. Format dict{<Headings enum> : float, <Headings enum> : float,... }\n
        """
        if self.running:
            self.game.update(
                self.map,
                self.robot_location,
                self.robot_heading,
                location_probs,
                headings_probs,
            )
            self.running = self.game.display()
        else:
            print("Pygame closed. Quiting...")
            self.game.quit()

    def observe(self):
        """Observes the walls at the current robot location

        Return:\n
        A list of the observations modified by the observation noise, where 1 signifies a wall and 0 signifies an empty cell. The order of the list is [S, E, N, W]
        """
        # get the neighboring walls to create the true observation table
        observations = [
            0
            if self.traversable(
                self.robot_location[0], self.robot_location[1], direction
            )
            else 1
            for direction in Directions
            if direction != Directions.St
        ]
        # apply observation noise
        observations = [
            1 - x if random.random() < self.observation_noise else x
            for x in observations
        ]
        return observations

    def create_observation_tables(self):
        observation_table = []
        for x in range(self.dimensions[0]):
            observation_table.append({})
            for y in range(self.dimensions[1]):
                if self.map[x][y] == 1:
                    observation_table[x][y] = -1
                    continue

                observation_table[x][y] = {}

                observations = [
                    0
                    if self.traversable(
                        x, y, direction
                    )
                    else 1
                    for direction in Directions
                    if direction != Directions.St
                ]

                for a in [0, 1]:
                    for b in [0, 1]:
                        for c in [0, 1]:
                            for d in [0, 1]:
                                potential_obs = (a, b, c, d)
                                num_wrong = 0
                                for i in range(len(observations)):
                                    if observations[i] != potential_obs[i]:
                                        num_wrong += 1
                                prob = (1 - self.observation_noise) ** (len(
                                    observations
                                )-num_wrong) * self.observation_noise ** num_wrong
                                observation_table[x][y][potential_obs] = prob
        return observation_table

    def create_locations_table(self):
        temp = []
        # loop through the x dim
        for x in range(self.dimensions[0]):
            temp.append([])
            # loop through the y dim
            for y in range(self.dimensions[1]):
                # If the cell is not traversable than set its value in the transition table to -1
                if self.map[x][y] == 1:
                    temp[x].append(-1)
                    continue
                temp[x].append({})
                for heading in list(Headings):
                    probs = {}

                    # Compute Transistion probabilities ignoring walls
                    for direction in Directions:
                        if direction.name == heading.name:
                            probs[direction] = 1 - self.action_noise
                        elif direction in Directions.get_ortho(
                            Directions, Directions[heading.name]
                        ):
                            probs[direction] = self.action_noise / 2
                        else:
                            probs[direction] = 0
                        # init stationary probability
                        probs[Directions.St] = 0

                        # account for walls. If there is a wall for one of the transition probabilities add the probability to the stationary probability and set the transisition probability to 0
                    for direction in Directions:
                        if not self.traversable(x, y, direction):
                            probs[Directions.St] += probs[direction]
                            probs[direction] = 0

                    # add the new transistion probabilities
                    temp[x][y].update({heading: probs})
        return temp

    def create_headings_table(self):
        temp = []
        # loop through the x dim
        for x in range(self.dimensions[0]):
            temp.append([])
            # loop through the y dim
            for y in range(self.dimensions[1]):
                # If the cell is not traversable than set its value in the transition table to -1
                if self.map[x][y] == 1:
                    temp[x].append(-1)
                    continue
                temp[x].append({})

                for heading in Headings:
                    probs = {}
                    # Handle case when the current heading is traversable
                    if self.traversable(x, y, heading):
                        for new_heading in Headings:
                            if heading == new_heading:
                                probs[new_heading] = 1
                            else:
                                probs[new_heading] = 0
                        temp[x][y].update({heading: probs})
                        continue

                    # If the current heading is not traversable

                    # Find which headings are available
                    headings_traversablity = {}
                    for new_heading in Headings:
                        if self.traversable(x, y, new_heading):
                            headings_traversablity[new_heading] = 1
                        else:
                            headings_traversablity[new_heading] = 0

                    # Sum these values for later arithmetic
                    total_traversable = sum(list(headings_traversablity.values()))
                    se_traversable = (
                        headings_traversablity[Headings.S]
                        + headings_traversablity[Headings.E]
                    )
                    nw_traversable = (
                        headings_traversablity[Headings.N]
                        + headings_traversablity[Headings.W]
                    )

                    # Compute the heading probabilities for traversable headings
                    for new_heading in Headings:
                        if self.traversable(x, y, new_heading):
                            if new_heading in [Headings.S, Headings.E]:
                                probs[new_heading] = (
                                    1 / total_traversable
                                    + self.action_bias / se_traversable
                                )

                            else:
                                probs[new_heading] = (
                                    1 / total_traversable
                                    - self.action_bias / nw_traversable
                                )
                        else:
                            probs[new_heading] = 0

                    # normalize heading probabilities
                    probs_sum = sum([probs[x] for x in Headings])
                    for h in Headings:
                        probs[h] /= probs_sum

                    # add the new transistion probabilities
                    temp[x][y].update({heading: probs})
        return temp

    def traversable(self, x, y, direction):
        """
        Returns true if the cell to the given direction of (x,y) is traversable, otherwise returns false.

        Arguements:\n
        row         -- the x coordinate of the initial cell\n
        col         -- the y coordinate of the initial cell\n
        direction   -- the direction of the cell to check for traversablility. Type: localization_env.Directions enum or localization_env.Headings\n

        Return:\n
        A boolean signifying whether the cell to the given direction is traversable or not
        """
        # see if the cell in the direction is traversable. If statement to handle out of bounds errors
        if (
            x + direction.value[0] >= 0
            and x + direction.value[0] < self.dimensions[0]
            and y + direction.value[0] >= 0
            and y + direction.value[0] < self.dimensions[1]
        ):
            if self.map[x + direction.value[0]][y + direction.value[1]] == 0:
                return True
        return False

    def dummy_location_and_heading_probs(self):
        """
        Returns a dummy location probability table and a dummy heading probability dictionary for testing purposes

        Returns:\n
        location probability table: Format is list[x][y] = float between (0-1)\n
        Headings probability table: Format is dict{<Heading enum> : float between (0-1)}
        """

        loc_probs = list()
        sum_probs = 0
        for x in range(self.dimensions[0]):
            loc_probs.append([])
            for y in range(self.dimensions[1]):
                if self.map[x][y] == 1:
                    loc_probs[x].append(0.0)
                else:
                    num = random.random()
                    loc_probs[x].append(num)
                    sum_probs += num
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                loc_probs[x][y] /= sum_probs

        hed_probs = {}
        sample = np.random.rand(4)
        sample = (sample / np.sum(sample)).tolist()
        i = 0
        for heading in le.Headings:
            hed_probs[heading] = sample[i]
            i += 1

        return loc_probs, hed_probs


if __name__ == "__main__":
    env = Environment(0.1, 0.1, 0.2, (10, 10), window_size=[1000, 1000])
    # print("Starting test. Press <enter> to make move")
    location, heading = env.dummy_location_and_heading_probs()

    done = False
    while env.running:

        observation = env.move()

        if printouts:
            print(observation)
        time.sleep(0.25)

