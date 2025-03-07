import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import math

class ObjectReconstructor:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable both depth and color streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Getting camera intrinsics
        self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        # Create an align object
        self.align = rs.align(rs.stream.color)
        
        # Set point cloud filtering parameters
        self.voxel_size = 0.01  # 1cm voxel grid
        self.distance_threshold = 0.05  # 5cm
        
    def capture_frame(self):
        """Capture a frame from the camera and return aligned color and depth images"""
        for _ in range(5):  # Skip first few frames to allow auto-exposure to stabilize
            self.pipeline.wait_for_frames()
            
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        return color_image, depth_image, depth_frame, color_frame
    
    def create_point_cloud(self, color_image, depth_frame, color_frame):
        """Create a colored point cloud from depth and color images"""
        # Create point cloud from depth image
        pc = rs.pointcloud()
        pc.map_to(color_frame)  # Map to color frame
        points = pc.calculate(depth_frame)
        
        # Convert to Open3D format
        pcd = o3d.geometry.PointCloud()
        v = np.asanyarray(points.get_vertices())
        v = v.view(np.float32).reshape(-1, 3)  # XYZ coordinates in camera space
        
        # Filter out invalid points (zeros or very far points)
        mask = (v[:, 2] > 0) & (v[:, 2] < 3.0)  # Keep points within 3 meters
        v = v[mask]
        
        # Get corresponding colors
        colors = color_image.reshape(-1, 3)[mask] / 255.0
        colors = colors[:, [2, 1, 0]]  # Convert BGR to RGB
        
        # Set points and colors
        pcd.points = o3d.utility.Vector3dVector(v)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def preprocess_point_cloud(self, pcd):
        """Preprocess point cloud: downsampling, removing outliers, etc."""
        # Downsample the point cloud
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Remove statistical outliers
        pcd_filtered, _ = pcd_down.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        
        # Optional: Remove radius outliers
        pcd_filtered, _ = pcd_filtered.remove_radius_outlier(
            nb_points=16, radius=0.05)
        
        return pcd_filtered
    
    def segment_objects(self, pcd):
        """Segment objects from the background and identify different objects"""
        # Convert to numpy array for processing
        points = np.asarray(pcd.points)
        
        # Use DBSCAN for clustering
        clustering = DBSCAN(eps=0.03, min_samples=10).fit(points)
        labels = clustering.labels_
        
        # Filter out noise (-1)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        # Create a list to store each object's point cloud
        objects = []
        
        # Colors for visualization
        max_label = max(labels) if len(labels) > 0 else 0
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        
        # Visualize each object
        for label in unique_labels:
            # Get points for this cluster
            indices = np.where(labels == label)[0]
            
            # Ensure we have enough points
            if len(indices) < 50:  # Minimum points for a valid object
                continue
                
            # Create a new point cloud for this object
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(points[indices])
            obj_pcd.colors = o3d.utility.Vector3dVector(colors[indices, :3])
            
            # Compute object properties
            obj_properties = self.analyze_object(obj_pcd)
            
            objects.append({
                'point_cloud': obj_pcd,
                'properties': obj_properties
            })
        
        return objects
    
    def analyze_object(self, pcd):
        """Analyze object properties such as dimensions, shape, etc."""
        # Get points as numpy array
        points = np.asarray(pcd.points)
        
        # Calculate axis-aligned bounding box
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        dimensions = max_bounds - min_bounds
        
        # Calculate center of mass
        center = np.mean(points, axis=0)
        
        # Calculate volume (approximate using convex hull)
        if len(points) > 4:  # Need at least 4 points for 3D convex hull
            try:
                hull = ConvexHull(points)
                volume = hull.volume
            except:
                volume = np.prod(dimensions)  # Fallback to bounding box volume
        else:
            volume = np.prod(dimensions)
        
        # Calculate principal axes (PCA)
        centered_points = points - center
        covariance_matrix = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Calculate elongation (ratio of largest to smallest dimension)
        if eigenvalues[2] > 0:
            elongation = eigenvalues[0] / eigenvalues[2]
        else:
            elongation = 1000  # Very elongated
        
        # Calculate flatness (ratio of medium to smallest dimension)
        if eigenvalues[2] > 0:
            flatness = eigenvalues[1] / eigenvalues[2]
        else:
            flatness = 1000  # Very flat
            
        # Determine if object is more flat or elongated
        shape_type = "unknown"
        if elongation > 3 and flatness < 2:
            shape_type = "elongated"
        elif flatness > 3:
            shape_type = "flat"
        elif max(elongation, flatness) < 2:
            shape_type = "compact"
        else:
            shape_type = "irregular"
        
        # Calculate dimensions along principal axes
        pca_dimensions = np.sqrt(eigenvalues * 4)  # 2σ in each direction
        
        # Convert dimensions from camera units to meters
        dimensions_meters = dimensions
        pca_dimensions_meters = pca_dimensions
        
        return {
            'dimensions': dimensions_meters.tolist(),  # [width, height, depth] in meters
            'pca_dimensions': pca_dimensions_meters.tolist(),  # Principal dimensions
            'volume': volume,
            'center': center.tolist(),
            'principal_axes': eigenvectors.tolist(),
            'elongation': float(elongation),
            'flatness': float(flatness),
            'shape_type': shape_type
        }
    
    def recommend_grasp(self, obj_properties):
        """Recommend suitable grasp based on object properties"""
        shape_type = obj_properties['shape_type']
        dimensions = obj_properties['dimensions']
        pca_dimensions = obj_properties['pca_dimensions']
        
        # Get object size
        max_dim = max(dimensions)
        min_dim = min(dimensions)
        
        # Basic grasp recommendations based on shape and size
        if shape_type == "elongated":
            if max_dim < 0.15:  # Small elongated object
                return {
                    'grasp_type': 'precision_grasp',
                    'approach_vector': obj_properties['principal_axes'][0],  # Approach along major axis
                    'grasp_width': min_dim,
                    'description': 'Use precision grasp (thumb and fingers) from the side of the object'
                }
            else:  # Large elongated object
                return {
                    'grasp_type': 'power_grasp',
                    'approach_vector': obj_properties['principal_axes'][0],  # Approach along major axis
                    'grasp_width': min_dim,
                    'description': 'Use power grasp (whole hand) from the side of the object'
                }
        
        elif shape_type == "flat":
            if max_dim < 0.10:  # Small flat object
                return {
                    'grasp_type': 'precision_pinch',
                    'approach_vector': obj_properties['principal_axes'][2],  # Approach perpendicular to flat surface
                    'grasp_width': pca_dimensions[2],
                    'description': 'Use precision pinch (thumb and index) from above the object'
                }
            else:  # Large flat object
                return {
                    'grasp_type': 'adducted_thumb',
                    'approach_vector': obj_properties['principal_axes'][2],  # Approach perpendicular to flat surface
                    'grasp_width': pca_dimensions[2],
                    'description': 'Use adducted thumb grasp (thumb against side of index) from the edge'
                }
        
        elif shape_type == "compact":
            if max_dim < 0.05:  # Very small object
                return {
                    'grasp_type': 'precision_tripod',
                    'approach_vector': obj_properties['principal_axes'][2],
                    'grasp_width': min_dim,
                    'description': 'Use precision tripod grasp (thumb, index, middle finger)'
                }
            elif max_dim < 0.10:  # Small compact object
                return {
                    'grasp_type': 'precision_disk',
                    'approach_vector': obj_properties['principal_axes'][2],
                    'grasp_width': min_dim,
                    'description': 'Use precision disk grasp (thumb and fingers in circular arrangement)'
                }
            else:  # Larger compact object
                return {
                    'grasp_type': 'spherical_grasp',
                    'approach_vector': obj_properties['principal_axes'][2],
                    'grasp_width': max_dim,
                    'description': 'Use spherical grasp (fingers wrapped around object)'
                }
        
        else:  # Irregular object
            if max_dim < 0.10:  # Small irregular object
                return {
                    'grasp_type': 'precision_grasp',
                    'approach_vector': obj_properties['principal_axes'][2],
                    'grasp_width': min_dim,
                    'description': 'Use precision grasp adapted to object shape'
                }
            else:  # Larger irregular object
                return {
                    'grasp_type': 'power_grasp',
                    'approach_vector': obj_properties['principal_axes'][2],
                    'grasp_width': min_dim,
                    'description': 'Use power grasp adapted to object shape'
                }
    
    def visualize_objects_with_grasps(self, objects):
        """Visualize objects with recommended grasp approaches"""
        # Create visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        for obj in objects:
            pcd = obj['point_cloud']
            properties = obj['properties']
            grasp = obj['grasp']
            
            # Add point cloud
            vis.add_geometry(pcd)
            
            # Add bounding box
            min_bounds = np.array(properties['center']) - np.array(properties['dimensions']) / 2
            max_bounds = np.array(properties['center']) + np.array(properties['dimensions']) / 2
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bounds, max_bounds)
            bbox.color = (1, 0, 0)  # Red bounding box
            vis.add_geometry(bbox)
            
            # Add coordinate frame at object center showing principal axes
            coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.1, origin=properties['center'])
            
            # Rotation matrix from principal axes
            R = np.array(properties['principal_axes'])
            coord.rotate(R, center=properties['center'])
            vis.add_geometry(coord)
            
            # Add grasp approach vector
            approach_point = np.array(properties['center'])
            approach_vector = np.array(grasp['approach_vector']) * 0.2  # Scale for visualization
            
            # Create a cylinder to represent grasp approach
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=0.005, height=np.linalg.norm(approach_vector))
            
            # Position and orient cylinder
            cylinder.paint_uniform_color([0, 1, 0])  # Green for approach
            
            # Rotation to align cylinder with approach vector
            z_axis = np.array([0, 0, 1])
            approach_normalized = approach_vector / np.linalg.norm(approach_vector)
            
            # Rotation axis and angle
            rotation_axis = np.cross(z_axis, approach_normalized)
            rotation_axis_norm = np.linalg.norm(rotation_axis)
            
            if rotation_axis_norm > 1e-6:
                rotation_axis = rotation_axis / rotation_axis_norm
                dot_product = np.dot(z_axis, approach_normalized)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                
                # Create rotation matrix
                rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
                cylinder.rotate(rotation, center=[0, 0, 0])
            
            # Position cylinder at approach start
            cylinder.translate(approach_point - approach_normalized * np.linalg.norm(approach_vector) / 2)
            
            vis.add_geometry(cylinder)
            
        # Set view point
        vis.get_view_control().set_front([0, 0, -1])
        vis.get_view_control().set_lookat([0, 0, 0])
        vis.get_view_control().set_up([0, -1, 0])
        
        # Run visualization
        vis.run()
        vis.destroy_window()
    
    def process_scene(self):
        """Main function to process the scene"""
        # Capture frame
        color_image, depth_image, depth_frame, color_frame = self.capture_frame()
        
        if color_image is None or depth_image is None:
            print("Failed to capture frame")
            return None
        
        # Create point cloud
        pcd = self.create_point_cloud(color_image, depth_frame, color_frame)
        
        # Preprocess point cloud
        pcd_filtered = self.preprocess_point_cloud(pcd)
        
        # Segment objects
        objects = self.segment_objects(pcd_filtered)
        
        # Recommend grasp for each object
        for obj in objects:
            grasp = self.recommend_grasp(obj['properties'])
            obj['grasp'] = grasp
            
            # Print object properties and grasp recommendation
            print("\nObject Analysis:")
            print(f"  Shape type: {obj['properties']['shape_type']}")
            print(f"  Dimensions (m): {[round(d, 3) for d in obj['properties']['dimensions']]}")
            print(f"  Volume (m³): {round(obj['properties']['volume'], 6)}")
            print("\nGrasp Recommendation:")
            print(f"  Grasp type: {grasp['grasp_type']}")
            print(f"  Description: {grasp['description']}")
            print(f"  Grasp width: {round(grasp['grasp_width'], 3)} meters")
        
        # Visualize the results
        self.visualize_objects_with_grasps(objects)
        
        return objects
    
    def cleanup(self):
        """Clean up resources"""
        self.pipeline.stop()


def main():
    # Create object reconstructor
    reconstructor = ObjectReconstructor()
    
    try:
        # Process the scene
        reconstructor.process_scene()
    finally:
        # Clean up
        reconstructor.cleanup()


if __name__ == "__main__":
    main()
