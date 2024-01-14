"""
The animation code is stolen/borrowed from https://matplotlib.org/2.1.2/gallery/animation/simple_3danim.html
and made to work with this particular setup.
The code is not pretty, I admit it, but it works ATM.
I might rewrite it later...
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from physics import Body, System

# We set up how we want the System class to behave
System.dim = 3
System.G = 2
System.N = 30000
System.delta_t = 0.01

# Defining some bodies
body1 = Body((4, 4, 2), (-1, 1, 0), 5)
body2 = Body((-4, -4, -2), (1, -1, 0), 5)
body3 = Body((0, 0, 0), (0, 0, 0), 3)
body4 = Body((-8, 0, -8), (-1, -1, -1), 0.00001)

# And puting them in a system
system = System(body1, body2, body3, body4)

# Simulating
system.simulate()

def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        start, end = max(num-20, 0), num
        line.set_data(data[0:2, start:end])
        line.set_3d_properties(data[2, start:end])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure(figsize=(10, 10))
ax = p3.Axes3D(fig)

data = [val[::50, :].T for _, val in system.r.items()]
mass = [val.mass for key, val in system.bodies.items()]

lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1],
                 f'{col}--', markevery=[-1], marker='o', ms=5*(mas)**(2/3))[0]
         for dat, mas, col in zip(data, mass, ['r', 'b', 'g', 'k', 'm', 'c', 'y'])]

# Setting the axes properties
ax.set_xlim3d([-20, 20])
ax.set_xlabel('x')

ax.set_ylim3d([-20, 20])
ax.set_ylabel('y')

ax.set_zlim3d([-20, 20])
ax.set_zlabel('z')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, int(System.N/50), fargs=(data, lines),
                                   interval=1, blit=False)

plt.show()

# Specify the output filename for the MP4 video
output_file = "n-body-simulator-the-movie.mov"

# Define the writer to save the animation as an MP4
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10, metadata=dict(artist='Magnus Kvåle Helliesen'), bitrate=1800)

# Save the animation as an MP4
line_ani.save(output_file, writer=writer)

# Specify the output filename for the GIF
gif_output_file = "n-body-simulator-the-movie.gif"

# Save the animation as a GIF
line_ani.save(gif_output_file, writer='pillow', fps=10)