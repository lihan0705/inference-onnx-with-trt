import numpy as np
import torch
import cv2
def map_pc_to_img(pc, image_path):
    """根据指定的图片，点云，计算出来的内外参, 赋予图像深度信息.

    Args:
        pc (List[List]): 指定的点云. 前三列为 x,y,z. 形如 [[x1,y1,z1],...]
        image_path (str): 对应的图片路径
        intrinsic (List[List]): 相机内参
        extrinsic (List[List]): 相机外参
        distortion (List): 相机畸变参数
        output_path (str): 输出图像路径
        point_size (int): 投影后点云的大小
        color_col_indx (int): 颜色赋值使用的列索引
    Returns:
        colors: List[List] 对应点云的颜色信息. 形如 [[r1,g1,b1], ...]
    """
    pc = pc[0,0,0,:,0:2,:]
    pc = pc.squeeze(-1).numpy().tolist()
    # print(pc)
    # paint the point according the axis value

    # scale the image
    img = cv2.imread(image_path)
    # projection_points[:, :2] /= scale_factor
    for idx, point in enumerate(pc):
        img = cv2.circle(img, (int(point[0]), int(point[1])),1,(256,0,0),1)
    # img = cv2.undistort(img, intrinsic_matrix, distortion_matrix)
    plt.imshow(img)
    plt.show()
    pass

def map_pc_to_img_np(pc, image_path):
    """根据指定的图片，点云，计算出来的内外参, 赋予图像深度信息.

    Args:
        pc (List[List]): 指定的点云. 前三列为 x,y,z. 形如 [[x1,y1,z1],...]
        image_path (str): 对应的图片路径
        intrinsic (List[List]): 相机内参
        extrinsic (List[List]): 相机外参
        distortion (List): 相机畸变参数
        output_path (str): 输出图像路径
        point_size (int): 投影后点云的大小
        color_col_indx (int): 颜色赋值使用的列索引
    Returns:
        colors: List[List] 对应点云的颜色信息. 形如 [[r1,g1,b1], ...]
    """
    pc = pc.tolist()
    # print(pc)
    # paint the point according the axis value

    # scale the image
    img = cv2.imread(image_path)
    # projection_points[:, :2] /= scale_factor
    for idx, point in enumerate(pc):
        img = cv2.circle(img, (int(point[0]), int(point[1])),1,(256,0,0),1)
    # img = cv2.undistort(img, intrinsic_matrix, distortion_matrix)
    plt.imshow(img)
    plt.show()

def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, dtype=torch.float):
    """Get the reference points used in SCA and TSA.
    Args:
        H, W: spatial shape of bev.
        Z: hight of pillar.
        D: sample D points uniformly from each pillar.
        device (obj:`device`): The device where
            reference_points should be.
    Returns:
        Tensor: reference points used in decoder, has \
            shape (bs, num_keys, num_levels, 2).
    """

    # reference points in 3D space, used in spatial cross-attention (SCA)
    if dim == '3d':
        zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
        return ref_3d


def project_3d_to_2d(reference_points_np, lidar2img_extric, lidar2img_intric):
    reference_points_cam_np = lidar2img_extric @ reference_points_np.T
    reference_points_cam_np[0:3, :] = lidar2img_intric @ reference_points_cam_np[0:3, :]
    print("cdcd:",reference_points_cam_np)
    reference_points_cam_np[0:2,:] /= reference_points_cam_np[2,:]
    reference_points_cam_np = reference_points_cam_np.T
    return reference_points_cam_np



intri_matrix_fake = [[1.027012053436998713e+03,0.000000000000000000e+00,9.484430095985443359e+02],
                        [0.000000000000000000e+00,1.036028616237759024e+03,2.700725862399331163e+02],
                        [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]]

extri_matrix_fake =  [[-1.552086702081234559e-02,-9.998769963783105119e-01,2.257166456141435074e-03,-1.090122859406091743e-01],
                        [1.282441400766836659e-01,-4.229488323220065293e-03,-9.917336093752777693e-01,7.639675022667467008e-01],
                        [9.916211692087431029e-01,-1.510309710000634631e-02,1.282940109088248071e-01,-2.200157738465011281e+00],
                        [0, 0, 0, 1]]

reference_points = get_reference_points(128, 256)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lidar2img_intric = []
lidar2img_extric = []
lidar2img_intric.append(intri_matrix_fake)
lidar2img_extric.append(extri_matrix_fake)

lidar2img_intric = reference_points.new_tensor(lidar2img_intric)  # (B, N, 4, 4)
lidar2img_intric = lidar2img_intric.repeat(reference_points.shape[0], 1, 1, 1)
lidar2img_intric = lidar2img_intric.to(reference_points.device)

lidar2img_extric = reference_points.new_tensor(lidar2img_extric)  # (B, N, 4, 4)
lidar2img_extric = lidar2img_extric.repeat(reference_points.shape[0], 1, 1, 1)
lidar2img_extric = lidar2img_extric.to(reference_points.device)


reference_points = reference_points.clone()
print("reference_points",reference_points.shape)
pc_range = [0, -25.6, -1, 102.4, 25.6, 8]
reference_points[..., 0:1] = reference_points[..., 0:1] * \
    (pc_range[3] - pc_range[0]) + pc_range[0]
reference_points[..., 1:2] = reference_points[..., 1:2] * \
    (pc_range[4] - pc_range[1]) + pc_range[1]
reference_points[..., 2:3] = reference_points[..., 2:3] * \
    (pc_range[5] - pc_range[2]) + pc_range[2]
print("..",reference_points.shape)
reference_points = torch.cat(
    (reference_points, torch.ones_like(reference_points[..., :1])), -1)
reference_points = reference_points.permute(1, 0, 2, 3)


D, B, num_query = reference_points.size()[:3]
reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, 1, 1, 1).unsqueeze(-1)


# visulization points
reference_points_np = reference_points.numpy().squeeze(-1)
reference_points_np = reference_points_np.squeeze(1).squeeze(1)
reference_points_np = reference_points_np.reshape(-1,4)
#reference_points_np0 = reference_points_np[0,:,:]
#reference_points_np1 = reference_points_np[1,:,:]
#reference_points_np2 = reference_points_np[2,:,:]
#reference_points_np3 = reference_points_np[3,:,:]
#print(reference_points_np3.shape)
"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(reference_points_np[:,0],reference_points_np[:,1],reference_points_np[:,2], c='r')
#ax.scatter(reference_points_np0[:,0],reference_points_np0[:,1],reference_points_np0[:,2], c='r')
#ax.scatter(reference_points_np1[:,0],reference_points_np1[:,1],reference_points_np1[:,2], c='r')
#ax.scatter(reference_points_np2[:,0],reference_points_np2[:,1],reference_points_np2[:,2], c='r')
#ax.scatter(reference_points_np3[:,0],reference_points_np3[:,1],reference_points_np3[:,2], c='r')

ax.set_xlabel('X label')
ax.set_ylabel('Y label')
ax.set_zlabel('Z label')
plt.show()
"""
lidar2img_extric_ = np.array(extri_matrix_fake)
lidar2img_intric_ = np.array(intri_matrix_fake)

reference_points_cam_np = project_3d_to_2d(reference_points_np, lidar2img_extric_, lidar2img_intric_)
map_pc_to_img_np(reference_points_cam_np, "/home/liangdao_hanli/Desktop/task/transformer/data_img.png")
print(reference_points_cam_np[:10,:])

lidar2img_extric = lidar2img_extric.view(
    1, B, 1, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
lidar2img_intric = lidar2img_intric.view(
    1, B, 1, 1, 3, 3).repeat(D, 1, 1, num_query, 1, 1)

print("////////////////////")
print(lidar2img_extric.shape)
print(lidar2img_intric.shape)
print(reference_points.shape)
print("////////////////////")

reference_points_cam = torch.matmul(lidar2img_extric.to(torch.float32),reference_points.to(torch.float32))
reference_points_cam = reference_points_cam.squeeze(-1)

reference_points_cam = reference_points_cam.reshape(-1,4)
print("cbcb",reference_points_cam.shape)
print("cbcb+intric",lidar2img_intric[0,0,0,0,:,:].shape)
print("reference_points_cam[:,0:3].T",reference_points_cam.shape)
lidar2img_intric = lidar2img_intric[0,0,0,0,:,:]
reference_points_cam = reference_points_cam.T
reference_points_cam[0:3,:] = torch.matmul(lidar2img_intric.to(torch.float32),reference_points_cam[0:3,:].to(torch.float32))
print("cbcb",reference_points_cam)

reference_points_cam = reference_points_cam
reference_points_cam[0:2,:] /= reference_points_cam[2,:]
reference_points_cam_torch = reference_points_cam.numpy()
print("bb:", reference_points_cam_torch.shape)
reference_points_cam_torch = reference_points_cam_torch.T
#reference_points_cam_torch = reference_points_cam_torch.squeeze(1).squeeze(1)
#reference_points_cam_torch = reference_points_cam_torch.reshape(-1,4)
print(reference_points_cam_torch[:10,:])
# ref_point = ref_point.numpy()
print(reference_points_cam_torch.shape)
map_pc_to_img_np(reference_points_cam_torch, "/home/liangdao_hanli/Desktop/task/transformer/data_img.png")
reference_points_cam_torch = reference_points_cam_torch.reshape(4,1,1,-1,4,1).squeeze(-1)

eps = 1e-5

bev_mask = (reference_points_cam_torch[..., 2:3] > eps)
bev_mask = (bev_mask & (reference_points_cam_torch[..., 1:2] > 0.0)
            & (reference_points_cam_torch[..., 1:2] < 1.0)
            & (reference_points_cam_torch[..., 0:1] < 1.0)
            & (reference_points_cam_torch[..., 0:1] > 0.0))
print(reference_points_cam_torch.shape)
reference_points_cam_torch = np.transpose(reference_points_cam_torch,(2, 1, 3, 0, 4))
bev_mask = np.transpose(bev_mask,(2, 1, 3, 0, 4)).squeeze(-1)

#print(reference_points_cam_torch[0,0,0,3,:,:])
#print(reference_points_cam_torch.shape)

