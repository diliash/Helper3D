import copy
import sys

import numpy as np
import trimesh

from .src import SceneGraph, URDFParser, URDFTree


# Sample Points from trimesh.Scene, different point number for different geometry based on the area ratio. The number of points is not strict, just oversample
def SampleSurfaceFromTrimeshScene(trimesh_scene, num_points):
    geo_trans_mapping = {}
    # Get the mapping between the geometry key and its corresponding transformation
    for key in trimesh_scene.graph.nodes_geometry:
        geo_trans_mapping[trimesh_scene.graph[key][1]] = trimesh_scene.graph[key][0]

    points = []
    colors = []
    normals = []
    scene_face_indexes = np.array([], dtype=int)
    barycentric_coordinates = np.empty((1, 3))
    geometry_map = np.array([])
    
    all_triangles = trimesh_scene.triangles
    triangle_to_key_map = trimesh_scene.triangles_node
    
    #print(trimesh_scene.graph.geometry_nodes)

    # Get the points from the geometries in the trimesh.Scene

    for key, geometry in trimesh_scene.geometry.items():
        # Need to deepcopy, or it will influence the original mesh
        geometry = copy.deepcopy(geometry)
        # Take the scene transformation into account
        geometry.apply_transform(np.dot(trimesh_scene.graph["world"][0], geo_trans_mapping[key]))
        # Check the number of points based on the area ratio of the whole trimesh scene, make sure there are some samples for each geometry
        num_geo_points = max(int(geometry.area / trimesh_scene.area * num_points), int(num_points / len(trimesh_scene.geometry)))
        # Some geometry may not have texture uv

        #print(geometry.visual.uv, geometry.visual.material, isinstance(geometry.visual.material, trimesh.visual.material.PBRMaterial))
        if not isinstance(geometry.visual, trimesh.visual.color.ColorVisuals):
            if isinstance(geometry.visual.material, trimesh.visual.material.PBRMaterial):
                geometry.visual.material = geometry.visual.material.to_simple()
        #print(geometry.visual.uv, geometry.visual.material, isinstance(geometry.visual.material, trimesh.visual.material.PBRMaterial))
        if isinstance(geometry.visual, trimesh.visual.color.ColorVisuals):
            result = trimesh.sample.sample_surface(geometry, num_geo_points, sample_color=False)
            colors.append(np.array([0.5, 0.5, 0.5] * num_geo_points))
        else:
            if geometry.visual.uv is None or geometry.visual.material.image is None:
                result = trimesh.sample.sample_surface(geometry, num_geo_points, sample_color=False)
                colors.append(np.array([geometry.visual.material.main_color[:3] / 255] * num_geo_points))
            else:
                result = trimesh.sample.sample_surface(geometry, num_geo_points, sample_color=True)
                colors.append(np.array(result[2])[:, :3] / 255)

        points.append(np.array(result[0]))
        normals.append(geometry.face_normals[result[1]])
        face_indexes = np.array(result[1], dtype=int)
            
        triangles = geometry.vertices[geometry.faces[face_indexes]]
        scene_face_indexes = np.append(scene_face_indexes, face_indexes)
        barycentric_coordinates = np.append(barycentric_coordinates, trimesh.triangles.points_to_barycentric(triangles, result[0]), axis=0)
        geometry_map = np.append(geometry_map, [key] * len(result[0]))

    # Concatenate the array
    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)
    normals = np.concatenate(normals, axis=0)
    
    return points, colors, normals, scene_face_indexes, np.delete(barycentric_coordinates, 0, 0), geometry_map

# Load the URDF into trimesh, return the urdf and controller
# JointInfo: whether to print all the joint information
def getURDF(path, JointInfo=False):
    # Initialize the URDF
    # Parse the URDF file
    parser = URDFParser(path)
    parser.parse()
    # Construct the URDF tree
    links = parser.links
    joints = parser.joints
    tree = URDFTree(links, joints)
    # Construct the scene graph
    urdf = SceneGraph(tree.root)

    # Get the controller
    # Get all the nodes
    controller = urdf.getNodes()
    if JointInfo:
        urdf.printInfo()

    return urdf, controller
