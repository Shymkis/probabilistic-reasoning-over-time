# CS5313_Localization_Env

Localization environment for the second project of CS5313/7313 - Advanced Artificial Intelligence

## About

This is a testing environment for robot localization. The robot's movements are modelled by a DBN, where the hidden
states are the location and heading of the robot. The robot will try to move in the direction of its heading
every step, but may go orthogonally to the intedended direction due to simulated action noise. It will only change heading
when it runs into a wall, and then it will select from one of the traversable headings at its current location, 
influenced by an action bias. At every step the robot will return an observation, which may be faulty due to observation
noise, as the evidence variable.

## Requirements

This code requires the `pygame` library for the visualizations, and `numpy`. In the code there is a variable named `df`,
which when set to `True` will also print DataFrames. If this is set to true you will need `pandas` as well.

## Installation

Download the repository and add the whole folder "CS5313_Localization_Env" to your project as a subfolder.
In order to make use of the code, import it with

```python
from CS5313_Localization_Env import localization_env as le
```

## Usage

First you must initialize an environment. This is done with:

```python
env = le.Environment(
        action_bias, 
        observation_noise, 
        action_noise, 
        dimensions, 
        seed=seed, 
        window_size=[x,y])
```

where you supply the values for the arguments. This will also initialize the display. (Note: window_size and dimensions
should have the same aspect ratio to avoid stretching of the display)

Then, in order to move your robot within the environment, it is as simple as:

```python
env.update(location_probabilities, heading_probabilities)
observation = env.move()
```

where `update` updates the visualizer, with the arguments as your program's estimation of where
the robot is (location) and where it is going (heading). More can be seen on these arguements in the code comments. 

`move()` moves the robot in the environment and returns the next observation as a tuple. We update before we move, otherwise the robot's location in the visualizer would be one time step ahead of where your program is guessing it is.

Observations can also be retrieved with:

```python
observation = env.observe()
```

which will not update the display or move the robot.

You can access every variable within the environment with the form:

```python
env.<variable_name>
```

## Visualizer

In the visualizer the color of the cells represent the probability you passed to `env.update()` for each cell, where darker
higher probability. The yellow/orange cell represents the highest probability cell. The circle represents the robot's true
location and the 4 lines on the robot represent the probabilities you passed for the robot's heading. The bold white line
is the robot's true heading.

The visualizer can be stopped by closing the window, or programmatically by setting `self.running` to false.

## Directions and Headings Enums

Directions and Headings enums are both used to represent direction. The location transition table uses Directions to represent movements in each of the cardinal directions as well as a stationary move, which happens when action noise causes the robot to run into a wall. This class can be accessed with `le.Directions` for interpreting the location transition table.

Headings are used by both transition tables, as well as the `robot_heading` variable. In the location transition table it is used as a lookup for probabilities, and in the headings table it is also used for the keys of the output probability dictionary.
This class can be accessed with `le.Headings`

More can be learned by reading the comments in `localization_env.py`.

## Variables

The variables in the environment are as follows:

```text
action_bias             The supplied action bias, changing won't do anything
observation_bias        The supplied observation bias, changing wont do anything
action_noise            The supplied action noise, changing won't do anything
dimensions              The map dimensions, changing will BREAK the program
seed                    The random seed, changing won't do anything
map                     The map, changing will BREAK the program
free_cells              A list of traversable cells, changing won't do anything
location_transitions    The transition table for the robots location, changing will BREAK the program
heading_transitions     The transition table for the robot's heading, changing will BREAK the program
location_priors         The prior probabilities for each free cell, given as a dictionary of 
                        {(x1, y1) : prob, ..., (xn, yn) : prob} where each (xi, yi) is a free cell.
                        Does not provide probabilities for non-traversable cells.
heading_priors          The prior probabilities for each heading, given as a diction of 
                        {Heading.S : prob, ..., Heading.W : prob}
observation_tables      The observation probabilities at any given location. Format of 
                        observation_tables[x][y][observation_tuple] = probability of that observation
robot_location          The location of the robot, changing will BREAK the program
robot_heading           The heading of the robot, changing will BREAK the program
steps                   The number of moves the robot has made, with the first move being step 1
running                 True if the visualizer is running, false otherwise. You can set to false to 
                        stop the visualizer.
```

You shouldn't need to change any variables after initializing the enviroment, and doing so will either do nothing or break the program, so don't.

More detailed descriptions of the variables, such as their type and format can be seen in the comments.

## Other Notes

The coordinate system used in this environment is  (x, y).

All the generated mazes are completely enclosed by walls. This is to prevent action noise from causing the robot to go out of bounds. I'm mainly mentioning this so if you edit the the maze generation code you make sure to leave the bounding walls.

Headings and Directions are interchangable in some places in the code. The main difference between the two is the inclusion of the "Stationary" direction in `Directions` and of course their name. If you are wanting to edit this code make sure you are using the right Enum in the right places.

## Support

If you find a bug please email me at rrg053@utulsa.edu, as I don't generally check the class discussion board on harvey.

## Authors
James Hale & Robert Geraghty
