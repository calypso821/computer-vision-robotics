import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

def dh_joint(parameters):
    # Input: List of DH parameters for a joint, written as [a, alpha, d, theta]
    # Output: 4x4 homogeneous transformation matrix
    
    A = np.zeros((4, 4))

    A[0, 0] = np.cos(parameters[3])
    A[1, 0] = np.sin(parameters[3])
    A[0, 1] = -np.sin(parameters[3]) * np.cos(parameters[1])
    A[1, 1] = np.cos(parameters[3]) * np.cos(parameters[1])
    A[2, 1] = np.sin(parameters[1])
    A[0, 2] = np.sin(parameters[3]) * np.sin(parameters[1])
    A[1, 2] = -np.cos(parameters[3]) * np.sin(parameters[1])
    A[2, 2] = np.cos(parameters[1])
    A[0, 3] = parameters[0] * np.cos(parameters[3])
    A[1, 3] = parameters[0] * np.sin(parameters[3])
    A[2, 3] = parameters[2]
    A[3, 3] = 1
    
    return A


def stanford_manipulator(a1, a2, a3):
    A1 = dh_joint([0, -np.pi/2, 5, a1])
    A2 = dh_joint([0, np.pi/2, 5, a2])
    A3 = dh_joint([0, 0, 2 + a3, 0])
    return [A1, A2, A3]

def antropomorphic_manipulator(a1, a2, a3):
    A1 = dh_joint([0, np.pi/2, 3, a1])
    A2 = dh_joint([3, 0, 0, a2])
    A3 = dh_joint([3, 0, 0, a3])
    return [A1, A2, A3]

def show_system(ax, M):
    
    x_1 = np.array([M[0, 3], M[0, 3] + M[0, 0]])
    y_1 = np.array([M[1, 3], M[1, 3] + M[1, 0]])
    z_1 = np.array([M[2, 3], M[2, 3] + M[2, 0]])
    
    x_2 = np.array([M[0, 3], M[0, 3] + M[0, 1]])
    y_2 = np.array([M[1, 3], M[1, 3] + M[1, 1]])
    z_2 = np.array([M[2, 3], M[2, 3] + M[2, 1]])
    
    x_3 = np.array([M[0, 3], M[0, 3] + M[0, 2]])
    y_3 = np.array([M[1, 3], M[1, 3] + M[1, 2]])
    z_3 = np.array([M[2, 3], M[2, 3] + M[2, 2]])
    
    ax.plot3D(x_1, y_1, z_1, 'red', linewidth=4);
    ax.plot3D(x_2, y_2, z_2, 'green', linewidth=4);
    ax.plot3D(x_3, y_3, z_3, 'blue', linewidth=4);


def create_robot_visualization(robot_function, robot_name, slider_params):
    """
    Create interactive visualization for a robot arm
    
    Parameters:
    robot_function: function that returns transformation matrices (A1, A2, A3)
    robot_name: string for plot title
    slider_params: list of tuples [(name, min, max, init_val), ...] for each slider
    """
    # Create figure and axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.25)
    
    # Create sliders
    sliders = []
    for i, (name, min_val, max_val, init_val) in enumerate(slider_params):
        ax_slider = plt.axes([0.2, 0.1 - (i*0.04), 0.6, 0.03])
        slider = Slider(ax_slider, name, min_val, max_val, valinit=init_val)
        sliders.append(slider)
    
    def update(val):
        ax.cla()  # Clear only the 3D axis
        
        # Get current values from sliders
        a1 = sliders[0].val
        a2 = sliders[1].val
        a3 = sliders[2].val
        
        # Get transformation matrices
        A1, A2, A3 = robot_function(a1, a2, a3)
        
        # Calculate points
        origin = np.array([0, 0, 0, 1])
        s1 = A1 @ origin
        s2 = (A1 @ A2) @ origin
        s3 = (A1 @ A2 @ A3) @ origin
        
        # Create arrays for plotting
        x_points = [0, s1[0], s2[0], s3[0]]
        y_points = [0, s1[1], s2[1], s3[1]]
        z_points = [0, s1[2], s2[2], s3[2]]
        
        # Plot segments as lines
        ax.plot3D(x_points, y_points, z_points, 'k-', linewidth=2)
        
        # Plot coordinate systems
        show_system(ax, np.eye(4))  # Base
        show_system(ax, A1)         # First joint
        show_system(ax, A1 @ A2)    # Second joint
        show_system(ax, A1 @ A2 @ A3)  # Third joint
        
        # Plot joint points
        ax.scatter3D(x_points, y_points, z_points, color='red', s=100)
        
        # Label points
        ax.text(0, 0, 0, 'Origin', fontsize=10)
        ax.text(s1[0], s1[1], s1[2], 'Segment 1', fontsize=10)
        ax.text(s2[0], s2[1], s2[2], 'Segment 2', fontsize=10)
        ax.text(s3[0], s3[1], s3[2], 'Segment 3', fontsize=10)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{robot_name} Robot Arm')
        
        # Set specific axis limits
        ax.set_xlim(-5, 5)
        ax.set_ylim(-3, 10)
        ax.set_zlim(0, 10)
        
        # Add grid
        ax.grid(True)
        
        fig.canvas.draw_idle()
    
    # Connect sliders to update function
    for slider in sliders:
        slider.on_changed(update)
    
    # Initial plot
    update(None)
    
    plt.show()



slider_params_stanford = [
    ('θ1 (rad)', -np.pi, np.pi, 0),
    ('θ2 (rad)', -np.pi, np.pi, 0),
    ('d3', 0, 4, 0)
]
#create_robot_visualization(stanford_manipulator, "Stanford", slider_params_stanford)


slider_params_anthro = [
    ('θ1 (rad)', -np.pi, np.pi, np.pi/2),
    ('θ2 (rad)', -np.pi, np.pi, 3* np.pi/4),
    ('θ3 (rad)', -np.pi, np.pi, -3* np.pi/4)
]
create_robot_visualization(antropomorphic_manipulator, "Anthropomorphic", slider_params_anthro)