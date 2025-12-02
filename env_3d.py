#!/usr/bin/env python3
"""
3D Environment Conversion from 2D MiniGrid.
Extends 2D grid environments to 3D with height dimension.
"""

import gymnasium as gym
import minigrid
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

print("=" * 80)
print(" " * 20 + "3D ENVIRONMENT VIEWER (From 2D MiniGrid)")
print("=" * 80)
print()

# Load a 2D environment
env_name = 'MiniGrid-DoorKey-8x8-v0'
print(f"Loading 2D environment: {env_name}")
env_2d = gym.make(env_name, render_mode='rgb_array')
obs, info = env_2d.reset(seed=42)

print(f"[SUCCESS] 2D Environment loaded!")
print()

# Get 2D environment details
grid_2d = env_2d.unwrapped.grid
width = env_2d.unwrapped.width
height = env_2d.unwrapped.height
agent_pos = env_2d.unwrapped.agent_pos

print("=" * 80)
print("2D ENVIRONMENT DETAILS:")
print("=" * 80)
print(f"  Grid size (2D):  {width} x {height}")
print(f"  Agent position:  {agent_pos}")
print(f"  Mission:         {obs['mission']}")
print()

# Design 3D environment parameters
FLOOR_HEIGHT = 0
CEILING_HEIGHT = 10  # Very tall rooms!
WALL_THICKNESS = 0.3

print("=" * 80)
print("3D ENVIRONMENT DESIGN:")
print("=" * 80)
print(f"  Grid size (3D):  {width} x {height} x {CEILING_HEIGHT}")
print(f"  Floor level:     Z = {FLOOR_HEIGHT}")
print(f"  Ceiling level:   Z = {CEILING_HEIGHT}")
print(f"  Room height:     {CEILING_HEIGHT} units (VERY TALL!)")
print(f"  Wall thickness:  {WALL_THICKNESS} units")
print()

print("=" * 80)
print("CONVERTING 2D TO 3D:")
print("=" * 80)
print()
print("Conversion process:")
print("  1. Each 2D grid cell becomes a vertical column (0 to Z=10)")
print("  2. Walls extend from floor to ceiling")
print("  3. Doors become vertical doorways")
print("  4. Objects placed on the floor (Z=0)")
print("  5. Agent can move in X, Y (height stays at floor level)")
print()

# Parse 2D grid to identify objects
def parse_2d_grid():
    """Extract objects, walls, doors from 2D grid"""
    walls = []
    doors = []
    keys = []
    goals = []

    for x in range(width):
        for y in range(height):
            cell = grid_2d.get(x, y)
            if cell is not None:
                if cell.type == 'wall':
                    walls.append((x, y))
                elif cell.type == 'door':
                    doors.append((x, y, cell.color, cell.is_locked))
                elif cell.type == 'key':
                    keys.append((x, y, cell.color))
                elif cell.type == 'goal':
                    goals.append((x, y))

    return walls, doors, keys, goals

walls, doors, keys, goals = parse_2d_grid()

print(f"Objects found in 2D environment:")
print(f"  Walls:  {len(walls)} cells")
print(f"  Doors:  {len(doors)} doors")
print(f"  Keys:   {len(keys)} keys")
print(f"  Goals:  {len(goals)} goals")
print(f"  Agent:  1 at position {agent_pos}")
print()

print("=" * 80)
print("CREATING 3D VISUALIZATION:")
print("=" * 80)
print()

# Create 3D figure
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Function to create a 3D cube (for walls)
def create_cube(x, y, z_bottom, z_top, color='gray', alpha=0.7):
    """Create vertices for a 3D cube"""
    vertices = [
        [x, y, z_bottom], [x+1, y, z_bottom], [x+1, y+1, z_bottom], [x, y+1, z_bottom],  # Bottom
        [x, y, z_top], [x+1, y, z_top], [x+1, y+1, z_top], [x, y+1, z_top]  # Top
    ]

    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]]   # Top
    ]

    return Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidth=0.5)

# Function to create a 3D box (for objects on floor)
def create_box(x, y, size=0.3, height=0.3, color='red', alpha=0.9):
    """Create a small box for objects"""
    offset = (1 - size) / 2  # Center in cell
    vertices = [
        [x+offset, y+offset, 0], [x+offset+size, y+offset, 0],
        [x+offset+size, y+offset+size, 0], [x+offset, y+offset+size, 0],
        [x+offset, y+offset, height], [x+offset+size, y+offset, height],
        [x+offset+size, y+offset+size, height], [x+offset, y+offset+size, height]
    ]

    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]]
    ]

    return Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='black', linewidth=1)

print("Drawing 3D structures...")

# Draw walls (tall columns from floor to ceiling)
print(f"  Adding {len(walls)} wall columns...")
for wall_x, wall_y in walls:
    cube = create_cube(wall_x, wall_y, FLOOR_HEIGHT, CEILING_HEIGHT, color='gray', alpha=0.6)
    ax.add_collection3d(cube)

# Draw doors (openings in walls with frame)
print(f"  Adding {len(doors)} doors...")
for door_x, door_y, door_color, is_locked in doors:
    # Door frame (thinner than walls)
    color_map = {'red': 'red', 'yellow': 'yellow', 'blue': 'blue', 'green': 'green'}
    door_vis_color = color_map.get(door_color, 'brown')

    # Draw door as a partial wall (representing closed door)
    cube = create_cube(door_x, door_y, FLOOR_HEIGHT, CEILING_HEIGHT * 0.8,
                      color=door_vis_color, alpha=0.5)
    ax.add_collection3d(cube)

    # Add "LOCKED" indicator if locked
    if is_locked:
        ax.text(door_x + 0.5, door_y + 0.5, CEILING_HEIGHT * 0.4, 'LOCKED',
               fontsize=8, color='red', fontweight='bold')

# Draw keys (small objects on floor)
print(f"  Adding {len(keys)} keys...")
for key_x, key_y, key_color in keys:
    color_map = {'red': 'red', 'yellow': 'gold', 'blue': 'blue', 'green': 'green'}
    key_vis_color = color_map.get(key_color, 'yellow')

    box = create_box(key_x, key_y, size=0.4, height=0.2, color=key_vis_color, alpha=1.0)
    ax.add_collection3d(box)

    # Add label
    ax.text(key_x + 0.5, key_y + 0.5, 0.5, 'KEY', fontsize=9, color=key_vis_color,
           fontweight='bold', ha='center')

# Draw goals (glowing boxes)
print(f"  Adding {len(goals)} goals...")
for goal_x, goal_y in goals:
    box = create_box(goal_x, goal_y, size=0.6, height=0.3, color='lime', alpha=0.8)
    ax.add_collection3d(box)

    # Add label
    ax.text(goal_x + 0.5, goal_y + 0.5, 0.6, 'GOAL', fontsize=10, color='green',
           fontweight='bold', ha='center')

# Draw agent (larger box to represent agent)
print(f"  Adding agent at {agent_pos}...")
agent_x, agent_y = agent_pos
box = create_box(agent_x, agent_y, size=0.5, height=0.5, color='orange', alpha=1.0)
ax.add_collection3d(box)

# Add agent label
ax.text(agent_x + 0.5, agent_y + 0.5, 1.0, 'AGENT', fontsize=11, color='darkorange',
       fontweight='bold', ha='center')

# Draw floor grid
print("  Drawing floor grid...")
for x in range(width + 1):
    ax.plot([x, x], [0, height], [FLOOR_HEIGHT, FLOOR_HEIGHT], 'k-', alpha=0.2, linewidth=0.5)
for y in range(height + 1):
    ax.plot([0, width], [y, y], [FLOOR_HEIGHT, FLOOR_HEIGHT], 'k-', alpha=0.2, linewidth=0.5)

# Draw ceiling grid (optional, for reference)
print("  Drawing ceiling...")
for x in range(width + 1):
    ax.plot([x, x], [0, height], [CEILING_HEIGHT, CEILING_HEIGHT], 'b-', alpha=0.1, linewidth=0.3)
for y in range(height + 1):
    ax.plot([0, width], [y, y], [CEILING_HEIGHT, CEILING_HEIGHT], 'b-', alpha=0.1, linewidth=0.3)

print()

# Set labels and limits
ax.set_xlabel('X (East-West)', fontsize=12, fontweight='bold')
ax.set_ylabel('Y (North-South)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (Height)', fontsize=12, fontweight='bold')

ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_zlim(FLOOR_HEIGHT, CEILING_HEIGHT)

# Set view angle
ax.view_init(elev=25, azim=45)

# Title
ax.set_title(f'\nMission: {obs["mission"]}',
            fontsize=14, fontweight='bold', pad=20)

# Add grid
ax.grid(True, alpha=0.3)

# Save figure
output_file = r'c:\Users\20492\dev\toddler-ai\3d_environment.png'
plt.tight_layout()
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"[SUCCESS] 3D visualization saved to: {output_file}")
print()

# Show the plot
plt.show()

print("=" * 80)
print("3D ENVIRONMENT STRUCTURE:")
print("=" * 80)
print()
print("Structure Overview:")
print(f"  - Grid dimensions: {width} x {height} x {CEILING_HEIGHT}")
print(f"  - Total volume: {width * height * CEILING_HEIGHT} cubic units")
print(f"  - Room height: {CEILING_HEIGHT} units (very tall!)")
print()
print("Components:")
print(f"  - Floor: Flat plane at Z={FLOOR_HEIGHT}")
print(f"  - Ceiling: Flat plane at Z={CEILING_HEIGHT}")
print(f"  - Walls: Vertical columns extending from floor to ceiling")
print(f"  - Doors: Vertical doorways (currently shown as partial walls)")
print(f"  - Objects: Placed on floor (keys, goals)")
print(f"  - Agent: At floor level, can move in X-Y plane")
print()
print("3D Navigation:")
print("  - X-axis: East-West movement (left-right)")
print("  - Y-axis: North-South movement (forward-backward)")
print("  - Z-axis: Height (currently fixed at floor level)")
print()
print("Future Extensions:")
print("  - Add flying/jumping actions to move in Z-axis")
print("  - Multi-level floors at different heights")
print("  - Stairs/elevators between levels")
print("  - 3D obstacles (hanging objects, bridges)")
print()

print("=" * 80)
print("SUMMARY:")
print("=" * 80)
print()
print(f"Successfully converted 2D {env_name} to 3D!")
print(f"  Original: {width} x {height} (2D grid)")
print(f"  3D Version: {width} x {height} x {CEILING_HEIGHT} (very tall rooms!)")
print()
print(f"Visualization saved to: {output_file}")
print("Open this file to see the 3D environment structure!")
print()
print("=" * 80)

env_2d.close()
