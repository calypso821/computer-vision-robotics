import numpy as np

# 1. Instead of just θ₁, we now have three angles: θ₁, θ₂, θ₃
# 2. J = [∂x/∂θ₁  ∂x/∂θ₂  ∂x/∂θ₃]
#        [∂y/∂θ₁  ∂y/∂θ₂  ∂y/∂θ₃]
# 3. J * [Δθ₁] = -error
#        [Δθ₂]
#        [Δθ₃]

class RobotArm3Segments:
    def __init__(self, l1, l2, l3):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        
    def forward_kinematics(self, thetas):
        """Calculate end effector position given three joint angles"""
        theta1, theta2, theta3 = thetas
        
        # Cumulative angles
        c1 = np.cos(theta1)
        c12 = np.cos(theta1 + theta2)
        c123 = np.cos(theta1 + theta2 + theta3)
        
        s1 = np.sin(theta1)
        s12 = np.sin(theta1 + theta2)
        s123 = np.sin(theta1 + theta2 + theta3)
        
        # End effector position
        x = self.l1 * c1 + self.l2 * c12 + self.l3 * c123
        y = self.l1 * s1 + self.l2 * s12 + self.l3 * s123
        
        return np.array([x, y])
        
    def jacobian(self, thetas):
        """Calculate the Jacobian matrix for all three angles"""
        theta1, theta2, theta3 = thetas
        
        # Cumulative angles
        c1 = np.cos(theta1)
        c12 = np.cos(theta1 + theta2)
        c123 = np.cos(theta1 + theta2 + theta3)
        
        s1 = np.sin(theta1)
        s12 = np.sin(theta1 + theta2)
        s123 = np.sin(theta1 + theta2 + theta3)
        
        # Partial derivatives for x
        dx_dtheta1 = -self.l1 * s1 - self.l2 * s12 - self.l3 * s123
        dx_dtheta2 = -self.l2 * s12 - self.l3 * s123
        dx_dtheta3 = -self.l3 * s123
        
        # Partial derivatives for y
        dy_dtheta1 = self.l1 * c1 + self.l2 * c12 + self.l3 * c123
        dy_dtheta2 = self.l2 * c12 + self.l3 * c123
        dy_dtheta3 = self.l3 * c123
        
        return np.array([
            [dx_dtheta1, dx_dtheta2, dx_dtheta3],
            [dy_dtheta1, dy_dtheta2, dy_dtheta3]
        ])
        
    def inverse_kinematics(self, target_pos, initial_thetas=None, max_iter=100, tolerance=1e-6):
        """
        Calculate required joint angles to reach target position
        Returns: array of three angles [theta1, theta2, theta3]
        """
        if initial_thetas is None:
            initial_thetas = np.array([0.0, 0.0, 0.0])
            
        thetas = initial_thetas.copy()
        
        for i in range(max_iter):
            # Current end effector position
            current_pos = self.forward_kinematics(thetas)
            
            # Error between current and target position
            error = target_pos - current_pos
            
            # Check if we're close enough
            if np.linalg.norm(error) < tolerance:
                return thetas
                
            # Calculate Jacobian
            J = self.jacobian(thetas)
            
            # Solve for angle changes using pseudo-inverse
            # (since we have more variables than constraints)
            delta_thetas = np.linalg.pinv(J) @ error
            
            # Update angles
            thetas += delta_thetas
            
            # Optional: Normalize angles to [-π, π]
            thetas = np.mod(thetas + np.pi, 2 * np.pi) - np.pi
            
        raise Exception("Inverse kinematics did not converge")

# Example usage
def main():
    # Create robot arm with segment lengths
    arm = RobotArm3Segments(l1=2.0, l2=1.5, l3=1.0)
    
    # Target position
    target = np.array([3.0, 2.0])
    
    try:
        # Initial guess for angles
        initial_angles = np.array([0.0, 0.0, 0.0])
        
        # Solve inverse kinematics
        thetas = arm.inverse_kinematics(target, initial_angles)
        print(f"Solution found:")
        print(f"theta1 = {np.degrees(thetas[0]):.2f}°")
        print(f"theta2 = {np.degrees(thetas[1]):.2f}°")
        print(f"theta3 = {np.degrees(thetas[2]):.2f}°")
        
        # Verify solution
        final_pos = arm.forward_kinematics(thetas)
        print(f"\nTarget position: {target}")
        print(f"Achieved position: {final_pos}")
        print(f"Error: {np.linalg.norm(target - final_pos):.6f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()