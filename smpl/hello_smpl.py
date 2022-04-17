
from serialization import load_model
import numpy as np
import os
os.chdir('smpl')
import open3d as o3d

import matplotlib.pyplot as plt
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Pointclouds
from pytorch3d.ops.points_alignment import iterative_closest_point
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

if torch.has_cuda:
  device = "cuda"
else:
  device = "cpu"
## Load SMPL model (here we load the female model)
## Make sure path is correct
m = load_model( 'models/basicModel_f_lbs_10_207_0_v1.1.0.pkl' )

# load target obj 
# Set paths
DATA_DIR = "data"
obj_filename = os.path.join(DATA_DIR, "sport.obj")

# Load obj file
target_mesh = load_objs_as_meshes([obj_filename], device=device)

# We scale normalize and center the target mesh to fit in a sphere of radius 1 
# centered at (0,0,0). (scale, center) will be used to bring the predicted mesh 
# to its original center and scale.  Note that normalizing the target mesh, 
# speeds up the optimization but is not necessary!
target_verts = target_mesh.verts_packed()
N = target_verts.shape[0]

center = target_verts.mean(0)
scale = max((target_verts - center).abs().max(0)[0])
target_mesh.offset_verts_(-center)
target_mesh.scale_verts_((1.0 / float(scale)))

target_pcd = Pointclouds(points=[target_mesh.verts_packed().float()])



# Generate mesh from pose ********

## Assign random pose and shape parameters
m.pose[:] = np.random.rand(m.pose.size) * .2 # shape (6890,3)
m.betas[:] = np.random.rand(m.betas.size) * .03 # shape (300,)


# pose mesh
verts = torch.from_numpy(m.r).to(device)
colors = torch.zeros_like(verts)
colors[:,0] = 1 # red
rgb = colors.to(device)
point_cloud = Pointclouds(points=[verts.float()]) #, features=[rgb]
# *******************
# Align point clouds

# Returns:
#   A named tuple `ICPSolution` with the following fields:
#   **converged**: A boolean flag denoting whether the algorithm converged
#       successfully (=`True`) or not (=`False`).
#   **rmse**: Attained root mean squared error after termination of ICP.
#   **Xt**: The point cloud `X` transformed with the final transformation
#       (`R`, `T`, `s`). If `X` is a `Pointclouds` object, returns an
#       instance of `Pointclouds`, otherwise returns `torch.Tensor`.
#   **RTs**: A named tuple `SimilarityTransform` containing
#   a batch of similarity transforms with fields:
#       **R**: Batch of orthonormal matrices of shape `(minibatch, d, d)`.
#       **T**: Batch of translations of shape `(minibatch, d)`.
#       **s**: batch of scaling factors of shape `(minibatch, )`.
#   **t_history**: A list of named tuples `SimilarityTransform`
#       the transformation parameters after each ICP iteration.


icp_solution = iterative_closest_point(target_pcd, point_cloud)

converged = icp_solution['converged']
rmse = icp_solution['rmse']
transformed_pcd = icp_solution['Xt']
RTs = icp_solution['SimilarityTransform']

# m=o3d.geometry.TriangleMesh(o3d.open3d_pybind.utility.Vector3dVector(m.r),
#                             o3d.open3d_pybind.utility.Vector3iVector(m.f))

# m.compute_vertex_normals()
# o3d.visualization.draw_geometries([m])




# ******** pytorch render


# Initialize a camera.
R, T = look_at_view_transform(20, 10, 0)
cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters. 
raster_settings = PointsRasterizationSettings(
    image_size=512, 
    radius = 0.003,
    points_per_pixel = 10
)


# Create a points renderer by compositing points using an alpha compositor (nearer points
# are weighted more heavily). See [1] for an explanation.
rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
renderer = PointsRenderer(
    rasterizer=rasterizer,
    compositor=AlphaCompositor()
)
images = renderer(transformed_pcd)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off");

print("Done")
# ******** pytorch render





# ## Write to an .obj file
# outmesh_path = 'hello_smpl.obj'
# with open( outmesh_path, 'w') as fp:
#     for v in m.r:
#         fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

#     for f in m.f+1: # Faces are 1-based, not 0-based in obj files
#         fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

# ## Print message
# print ('..Output mesh saved to: ', outmesh_path )
