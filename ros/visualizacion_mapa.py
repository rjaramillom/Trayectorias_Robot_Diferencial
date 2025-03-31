#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Twist, PointStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import ColorRGBA
from sensor_msgs import point_cloud2
from dynamic_reconfigure.server import Server
from rgbd_safety.cfg import SafetyConfig
import collections

class AdvancedRGBDSafetyController:
    def __init__(self):
        rospy.init_node('advanced_rgbd_safety_controller')
        
        self.config = None
        self.srv = Server(SafetyConfig, self.dynamic_reconfigure_callback)
        
        # convertir imagenes 
        self.bridge = CvBridge()
        
        # buffer para filtro EMA
        self.distance_buffer = collections.deque(maxlen=5)
        self.sector_distances = collections.deque(maxlen=3)
        # topicos 
        # suscripciones
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.camera_info_callback)
        self.pc_sub = rospy.Subscriber('/camera/depth/points', PointCloud2, self.pointcloud_callback)
        
        # publicacion
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.marker_pub = rospy.Publisher('/safety_markers', Marker, queue_size=10)
        self.debug_image_pub = rospy.Publisher('/safety_debug_image', Image, queue_size=10)
        self.obstacle_pub = rospy.Publisher('/nearest_obstacle', PointStamped, queue_size=10)
        
        # variables
        self.current_velocity = Twist()
        self.image_width = 240
        self.image_height = 240
        self.horizontal_fov = 60.0
        self.camera_info = None
        self.nearest_obstacle = None
        
        rospy.loginfo("Inicializacion de controlador")
    # Llama a configuracion y regresa el valor 
    def dynamic_reconfigure_callback(self, config, level):
        self.config = config
        return config
    # se procesa la informacion de la camara 
    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.image_width = msg.width
        self.image_height = msg.height
        if msg.K[0] != 0:
            self.horizontal_fov = 2 * np.arctan(msg.K[2]/msg.K[0]) * 180/np.pi
    #nube de puntos
    def pointcloud_callback(self, msg):
        if self.config is None or not self.config.use_pointcloud:
            return
            
        # se conviertes las coordenas 3d
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_3d = np.array(list(gen))
        
        if len(points_3d) == 0:
            return
            
        distances = np.linalg.norm(points_3d, axis=1)
        angles = np.arctan2(points_3d[:,0], points_3d[:,2]) * 180/np.pi
        in_cone = np.abs(angles) < self.horizontal_fov/2
        
        if np.any(in_cone):
            nearest_idx = np.argmin(distances[in_cone])
            self.nearest_obstacle = points_3d[in_cone][nearest_idx]
            
            # si hay obstáculo cercano se publica 
            obstacle_msg = PointStamped()
            obstacle_msg.header = msg.header
            obstacle_msg.point.x = self.nearest_obstacle[0]
            obstacle_msg.point.y = self.nearest_obstacle[1]
            obstacle_msg.point.z = self.nearest_obstacle[2]
            self.obstacle_pub.publish(obstacle_msg)

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            sectors = self.analyze_sectors(depth_image)
            
            filtered_distances = self.apply_temporal_filter(sectors)
            
            debug_img = self.create_visualization(depth_image, sectors)
            
            self.apply_safety_measures(filtered_distances)
            
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))
            
        except Exception as e:
            rospy.logerr(f"Error en procesamiento: {str(e)}")
    #cálculo de distancias mínimas 

    def analyze_sectors(self, depth_image):
        height, width = depth_image.shape
        sector_width = width // 3
        
        sectors = {
            'left': depth_image[:, :sector_width],
            'center': depth_image[:, sector_width:2*sector_width],
            'right': depth_image[:, 2*sector_width:]
        }
        
        sector_results = {}
        for name, sector in sectors.items():
            valid_depths = sector[sector > 0]
            if valid_depths.size > 0:
                sector_results[name] = np.min(valid_depths)
            else:
                sector_results[name] = float('inf')
                
        return sector_results
    # Filtro EMA
    def apply_temporal_filter(self, sectors):
        self.sector_distances.append(sectors)
        
        # Promedio de las últimas N mediciones
        filtered = {}
        for sector in ['left', 'center', 'right']:
            values = [d[sector] for d in self.sector_distances if sector in d]
            filtered[sector] = np.mean(values) if values else float('inf')
            
        return filtered
    # para depuracion solamente
    def create_visualization(self, depth_image, sectors):
        # visualizacion de imagen RGBD 
        debug_img = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)
        
        # sectores 
        height, width = debug_img.shape[:2]
        cv2.line(debug_img, (width//3, 0), (width//3, height), (0,255,0), 1)
        cv2.line(debug_img, (2*width//3, 0), (2*width//3, height), (0,255,0), 1)
        
        # distancias 
        for i, (name, dist) in enumerate(sectors.items()):
            text = f"{name}: {dist:.2f}m" if dist != float('inf') else f"{name}: ---"
            cv2.putText(debug_img, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # para mapa ocupacional
        if self.nearest_obstacle is not None and self.config.show_3d_marker:
            self.publish_3d_marker()
            
        return debug_img

    def publish_3d_marker(self):
        marker = Marker()
        marker.header.frame_id = "camera_depth_frame"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "safety"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.nearest_obstacle[0]
        marker.pose.position.y = self.nearest_obstacle[1]
        marker.pose.position.z = self.nearest_obstacle[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
        marker.lifetime = rospy.Duration(0.5)
        self.marker_pub.publish(marker)

    def apply_safety_measures(self, sectors):
        if self.config is None:
            return
            
        min_distance = min(sectors.values())
        
        if min_distance < self.config.min_safe_distance:
            self.current_velocity.linear.x = 0
            self.current_velocity.angular.z = 0
            rospy.logwarn(f"¡Obstáculo detectado a {min_distance:.2f}m! Parada de emergencia.")
        
       
        elif min_distance < self.config.slow_down_distance:
            reduction = (min_distance - self.config.min_safe_distance) / \
                      (self.config.slow_down_distance - self.config.min_safe_distance)
            self.current_velocity.linear.x = self.config.max_speed * reduction
            
            # Evasión de obstáculos basada en sectores
            if sectors['center'] < sectors['left'] and sectors['center'] < sectors['right']:
                # Obstáculo en centro, girar hacia el lado más despejado
                if sectors['left'] > sectors['right']:
                    self.current_velocity.angular.z = self.config.avoidance_turn_speed
                else:
                    self.current_velocity.angular.z = -self.config.avoidance_turn_speed
            rospy.loginfo(f"Obstáculo cercano. Velocidad reducida a {self.current_velocity.linear.x:.2f}m/s")        
       
        else:
            self.current_velocity.linear.x = self.config.max_speed
            self.current_velocity.angular.z = 0
        
        # se publica velocidad 
        self.cmd_vel_pub.publish(self.current_velocity)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = AdvancedRGBDSafetyController()
        controller.run()
    except rospy.ROSInterruptException:
        # Detener el robot al salir
        twist = Twist()
        controller.cmd_vel_pub.publish(twist)
