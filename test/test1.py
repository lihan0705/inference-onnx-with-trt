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

def create_frustum():
    dbound = [1, 60, 0.5]
    iH, iW = [720, 1920]
    fH, fW = [90, 240]

    ds = (
        torch.arange(*dbound, dtype=torch.float)
        .view(-1, 1, 1)
        .expand(-1, fH, fW)
    )
    D, _, _ = ds.shape

    xs = (
        torch.linspace(0, iW - 1, fW, dtype=torch.float)
        .view(1, 1, fW)
        .expand(D, fH, fW)
    )
    ys = (
        torch.linspace(0, iH - 1, fH, dtype=torch.float)
        .view(1, fH, 1)
        .expand(D, fH, fW)
    )


    frustum = torch.stack((xs, ys, ds), -1)
    return frustum

intri_matrix =  torch.Tensor([[1.027012053436998713e+03,0.000000000000000000e+00,9.484430095985443359e+02],
                        [0.000000000000000000e+00,1.036028616237759024e+03,2.700725862399331163e+02],
                        [0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]])

extri_matrix = torch.Tensor([[-1.552086702081234559e-02,-9.998769963783105119e-01,2.257166456141435074e-03,-1.090122859406091743e-01],
                        [1.282441400766836659e-01,-4.229488323220065293e-03,-9.917336093752777693e-01,7.639675022667467008e-01],
                        [9.916211692087431029e-01,-1.510309710000634631e-02,1.282940109088248071e-01,-2.200157738465011281e+00],
                        [0, 0, 0, 1]])
def get_geometry(
    frustum,
    B=1,N=1
):
    D = frustum.shape[0]
    # undo post-transformation
    # B x N x D x H x W x 3

    points = frustum.repeat(B, N, 1, 1, 1, 1)

    # points = post_rots.matmul(points.unsqueeze(-1))
    # cam_to_lidar

    points = torch.cat(
        (
            points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
            points[:, :, :, :, :, 2:3],
        ),
        5
    )

    print(extri_matrix[:3,3])
    # points += extri_matrix[:3,3].repeat(B, N, 1, 1, 1, 1)
    extrin_matrix = torch.inverse(extri_matrix[:3,:3]).repeat(B, N, D, 1, 1, 1, 1).to(frustum.device)
    points = torch.inverse(intri_matrix).repeat(B, N, D, 1, 1, 1, 1).matmul(points.unsqueeze(-1))
    print(points.shape) 
    points = points.squeeze(-1)
    points -= extri_matrix[:3,3].repeat(B, N, D, 1, 1, 1)
    points = extrin_matrix.matmul(points.unsqueeze(-1))
    points = points.squeeze(-1)
    # points = points - torch.tensor([-51.2000, -51.2000,  -1.0000])
    print(points[0,0,0,0,:])

    # combine = torch.inverse(extri_matrix[:3,:3]).repeat(B, N, D, 1, 1, 1, 1).to(frustum.device)
    # combine = camera2lidar_rots.matmul(torch.inverse(intrins))
    #points = combine.matmul(points.unsqueeze(-1))
    #points = points.squeeze(-1)
    # points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)
    """
    if "extra_rots" in kwargs:
        extra_rots = kwargs["extra_rots"]
        points = (
            extra_rots.view(B, 1, 1, 1, 1, 3, 3)
            .repeat(1, N, 1, 1, 1, 1, 1)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )
    if "extra_trans" in kwargs:
        extra_trans = kwargs["extra_trans"]
        points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)
    """
    return points

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

frustum = create_frustum()
points = get_geometry(frustum)
points = points.reshape(-1, 3)
points = points.numpy()
frustum = frustum.reshape(-1, 3)
# frustum = frustum[:50000, :]
print(frustum.shape)
print(points.shape)
points = points[:200000,:]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2], c='r')
#ax.scatter(reference_points_np0[:,0],reference_points_np0[:,1],reference_points_np0[:,2], c='r')
#ax.scatter(reference_points_np1[:,0],reference_points_np1[:,1],reference_points_np1[:,2], c='r')
#ax.scatter(reference_points_np2[:,0],reference_points_np2[:,1],reference_points_np2[:,2], c='r')
#ax.scatter(reference_points_np3[:,0],reference_points_np3[:,1],reference_points_np3[:,2], c='r')

ax.set_xlabel('X label')
ax.set_ylabel('Y label')
ax.set_zlabel('Z label')
plt.show()