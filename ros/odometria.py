#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance, TwistWithCovariance
from sensor_msgs.msg import JointState

class PreciseOdometry:
    def __init__(self):
        rospy.init_node('encoder_odometry')
        
        # parametros fisicos 
        self.wheel_radius = 0.065  
        self.wheel_base = 0.30     
        self.encoder_resolution = 4096  
        
        # condiciones iniciales 
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_left_encoder = 0
        self.last_right_encoder = 0
        self.last_time = rospy.Time.now()
        
        # topicos 
        self.odom_pub = rospy.Publisher('/encoder_odom', Odometry, queue_size=10)
        self.encoder_sub = rospy.Subscriber('/joint_states', JointState, self.encoder_callback)
        
    def encoder_callback(self, msg):
      
        left_encoder = msg.position[0]
        right_encoder = msg.position[1]
        
        now = rospy.Time.now()
        dt = (now - self.last_time).to_sec()
        
        # calculo de desplazamiento de ruedas
        delta_left = (left_encoder - self.last_left_encoder) * self.wheel_radius
        delta_right = (right_encoder - self.last_right_encoder) * self.wheel_radius
        
        # desplazamiento lineal y angular 
        delta_distance = (delta_right + delta_left) / 2.0
        delta_theta = (delta_right - delta_left) / self.wheel_base
        
        # actualizacion de posicion
        self.x += delta_distance * np.cos(self.theta)
        self.y += delta_distance * np.sin(self.theta)
        self.theta += delta_theta
        
        # normalizacion
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        # calculo de velocidades 
        linear_vel = delta_distance / dt
        angular_vel = delta_theta / dt
        
        # publicar valores de odometria 
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = np.sin(self.theta/2)
        odom.pose.pose.orientation.w = np.cos(self.theta/2)
        
        odom.twist.twist.linear.x = linear_vel
        odom.twist.twist.angular.z = angular_vel
        
        self.odom_pub.publish(odom)
        
        # Actualizar valores anteriores
        self.last_left_encoder = left_encoder
        self.last_right_encoder = right_encoder
        self.last_time = now

if __name__ == '__main__':
    try:
        odometry = PreciseOdometry()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
