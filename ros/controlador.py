#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np

class PIDController:
    def __init__(self):
        rospy.init_node('pid_velocity_controller')
        
        # ganancias controlador 
        self.kp_linear = 0.5
        self.ki_linear = 0.01
        self.kd_linear = 0.1
        self.kp_angular = 0.8
        self.ki_angular = 0.01
        self.kd_angular = 0.2
        
        # variables de estado 
        self.last_error_linear = 0
        self.integral_linear = 0
        self.last_error_angular = 0
        self.integral_angular = 0
        self.last_time = rospy.Time.now()
        
        # topicos, suscribirse y publicar. Odometria y velocidades angular y linear 
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # set point inicial 
        self.target_linear = 0.5  # m/s
        self.target_angular = 0.0  # rad/s
        
    def odom_callback(self, msg):
        current_linear = msg.twist.twist.linear.x
        current_angular = msg.twist.twist.angular.z
        
        now = rospy.Time.now()
        dt = (now - self.last_time).to_sec()
        
        # Controlador clasico 
        error_linear = self.target_linear - current_linear
        self.integral_linear += error_linear * dt
        derivative_linear = (error_linear - self.last_error_linear) / dt
        output_linear = (self.kp_linear * error_linear + 
                        self.ki_linear * self.integral_linear + 
                        self.kd_linear * derivative_linear)
        
        
        error_angular = self.target_angular - current_angular
        self.integral_angular += error_angular * dt
        derivative_angular = (error_angular - self.last_error_angular) / dt
        output_angular = (self.kp_angular * error_angular + 
                         self.ki_angular * self.integral_angular + 
                         self.kd_angular * derivative_angular)
        
        # Publicar velocidad
        cmd = Twist()
        cmd.linear.x = output_linear
        cmd.angular.z = output_angular
        self.cmd_vel_pub.publish(cmd)
        
        # actualizacion de varialbes 
        self.last_error_linear = error_linear
        self.last_error_angular = error_angular
        self.last_time = now

if __name__ == '__main__':
    try:
        controller = PIDController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
