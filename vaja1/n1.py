import matplotlib.pyplot as plt
import numpy as np

width = 2.5
a = 0.5
f = 0.1
z = 10

def get_x(z):
    return f * width / z

for i in range(301):
    i /= 10
    x = get_x(z + i * a) * 100
    #print(f"t: {i}s x: {x:.4f} cm")


def plot_time_series(z, a, get_x):
    # Create time points from 0 to 30 seconds with 0.1s intervals
    t = np.linspace(0, 30, 301)
    
    # Calculate x values
    x = [get_x(z + t_i * a) * 100 for t_i in t]
    
    # Create the figure and axis
    plt.figure(figsize=(12, 6))
    
    # Plot the data
    plt.plot(t, x, 'b-', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Position (cm)', fontsize=12)
    plt.title('Position vs Time', fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize ticks
    plt.xticks(np.arange(0, 31, 5))
    
    # Add margins for better visibility
    plt.margins(x=0.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()


plot_time_series(z, a, get_x)