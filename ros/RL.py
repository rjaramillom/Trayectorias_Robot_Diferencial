#!/usr/bin/env python
import rospy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import os
from sensor_msgs.msg import Image, LaserScan, PointCloud2
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
import math

class DeepQLearningNavigator:
    def __init__(self):
        rospy.init_node('deep_q_learning_navigator')
        
        # Parámetros de la red 
        self.state_size = 10  
        self.action_size = 8  
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.current_pose = None
        self.goal_pose = None
        self.obstacle_distances = [float('inf')]*3  # ubicacon de obstaculos 
        self.velocity = Twist()
        
        # suscripciones 
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        # Solo para probar 
        #self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        
        # publicaciones 
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        
        self.bridge = CvBridge()
        self.last_action = None
        self.last_state = None
        self.last_reward = 0
        self.total_reward = 0
        self.episode = 0
        self.steps = 0
        
        ´´´# se carga modelo 
        self.model_file = os.path.expanduser('~/.ros/dqn_amr.h5')
        if os.path.exists(self.model_file):
            self.model.load_weights(self.model_file)
            self.target_model.load_weights(self.model_file)
            rospy.loginfo("Modelo DQN cargado desde disco")
        
        rospy.Timer(rospy.Duration(0.1), self.control_loop)
        rospy.loginfo("Navegador con Deep Q-Learning iniciado")
        ´´´
      # creacion de modelo 
    def _build_model(self):
        
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
        if self.goal_pose:
            self.publish_goal_marker()

    def goal_callback(self, msg):
        self.goal_pose = msg.pose
        self.episode += 1
        self.total_reward = 0
        rospy.loginfo(f"Nuevo objetivo establecido. Iniciando episodio {self.episode}")
        self.publish_goal_marker()

    # En prueba
    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[ranges == float('inf')] = msg.range_max
        
        # Dividir en 3 sectores (izquierda, centro, derecha)
        num_sectors = 3
        sector_size = len(ranges) // num_sectors
        self.obstacle_distances = [
            np.min(ranges[:sector_size]),          # Izquierda
            np.min(ranges[sector_size:2*sector_size]),  # Centro
            np.min(ranges[2*sector_size:])         # Derecha
        ]
    #Camara rgbd
    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            height, width = depth_image.shape
            sector_width = width // 3
            
            for i in range(3):
                sector = depth_image[:, i*sector_width:(i+1)*sector_width]
                valid_depths = sector[sector > 0]
                if valid_depths.size > 0:
                    self.obstacle_distances[i] = min(self.obstacle_distances[i], np.min(valid_depths))
        except Exception as e:
            rospy.logwarn(f"Error procesando imagen de profundidad: {str(e)}")
    # vector de estados 
    def get_state(self):
        if not self.current_pose or not self.goal_pose:
            return None
            
        # calculo de distancias 
        dx = self.goal_pose.position.x - self.current_pose.position.x
        dy = self.goal_pose.position.y - self.current_pose.position.y
        goal_dist = math.sqrt(dx**2 + dy**2)
        
        # calculo de angulo 
        yaw = self.quaternion_to_yaw(self.current_pose.orientation)
        goal_angle = math.atan2(dy, dx) - yaw
        goal_angle = (goal_angle + math.pi) % (2*math.pi) - math.pi  # Normalizar a [-π, π]
        
        # vel actual 
        vel_x = self.velocity.linear.x
        vel_theta = self.velocity.angular.z
        
        return np.array([
            self.obstacle_distances[1],  
            self.obstacle_distances[0],  
            self.obstacle_distances[2],  
            goal_dist,                  
            goal_angle,                
            vel_x,                      
            vel_theta,                  
            *self.obstacle_distances    
        ])

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
      
    # funciones de recompensa 
    def get_reward(self, state, new_state):
        if not state or not new_state:
            return 0
            
        reward = (state[3] - new_state[3]) * 10  # Escalado por factor 10
        
        min_dist = min(new_state[0], new_state[1], new_state[2])
        if min_dist < 0.5:
            reward -= (0.5 - min_dist) * 20  
        reward -= 0.01
        
        if new_state[3] < 0.3:  
            reward += 100
        return reward

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1))
        return np.argmax(act_values[0])

    def execute_action(self, action):
        cmd = Twist()

        if action == 0:   
            cmd.linear.x = 0.3
        elif action == 1: 
            cmd.angular.z = 0.5
        elif action == 2: 
            cmd.angular.z = -0.5
        elif action == 3: 
            cmd.linear.x = -0.2
        elif action == 4: 
            pass
            
        min_dist = min(self.obstacle_distances)
        if min_dist < 0.3:  
            if cmd.linear.x > 0:  
                cmd.linear.x = 0
                cmd.angular.z = 0.5 if self.obstacle_distances[0] > self.obstacle_distances[2] else -0.5
                
        self.velocity = cmd
        self.cmd_vel_pub.publish(cmd)
        return cmd

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])
        
        targets = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)
        
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(next_q_values[i])
                
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def publish_goal_marker(self):
        if not self.goal_pose:
            return
            
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = self.goal_pose
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime = rospy.Duration()
        self.marker_pub.publish(marker)

    def control_loop(self, event):
        if not self.goal_pose:
            return
            
        state = self.get_state()
        if state is None:
            return
            
        action = self.choose_action(state)
        self.execute_action(action)
        
        
        new_state = self.get_state()
        done = False
        
       
        if new_state[3] < 0.3:  # valor del robot
            rospy.loginfo(f"Objetivo alcanzado! Recompensa total: {self.total_reward}")
            done = True
            self.goal_pose = None
            
        reward = self.get_reward(state, new_state)
        self.total_reward += reward
        
        self.remember(state, action, reward, new_state, done)
        
        self.replay()
        
        self.steps += 1
        if self.steps % 100 == 0:
            self.update_target_model()
            self.model.save_weights(self.model_file)
            rospy.loginfo(f"Modelo guardado. ε={self.epsilon:.3f}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        navigator = DeepQLearningNavigator()
        navigator.run()
    except rospy.ROSInterruptException:
        # Guardar modelo al salir
        navigator.model.save_weights(navigator.model_file)
        rospy.loginfo("Modelo DQN guardado al salir")
