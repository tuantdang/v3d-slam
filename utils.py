import numpy as np
import cv2
import torch
import open3d as o3d

def load_filenames(path):
    file = open(path)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    temp = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list_rgb_depth = [(l[1], l[3]) for l in temp] #
    return list_rgb_depth

def extract_3D_points(seg, depth, cx, cy, fx, fy):
    indices = (torch.from_numpy(seg) > 0).nonzero()
    if indices.shape[0] == 0:
        return None
    indices = indices.cpu().numpy()
    points = np.zeros((indices.shape[0], 3))
    idx = 0
    for v, u in indices:
        d = depth[int(v), int(u)]
        x = ((u - cx) / fx) * d
        y = ((v - cy) / fy )* d
        # print([x, y, d])
        points[idx,:] = [x, y, d]
        idx +=1
    return points

def gaussian_kernel(size=3, sigma=1.0):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    # print(kernel)
    return kernel / np.sum(kernel)
    

def np_to_pc(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)    
    return pcd


def overlay(image, mask, color = (255,255,255), alpha=1.0, resize=None):
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

def blur_with_mask(frame, mask):
    k=4
    out = frame.copy()
    blur =  cv2.blur(frame,(k,k),0)
    out = frame.copy()
    out[mask>0] = blur[mask>0]
    return out

import open3d as o3d
def np_to_pc(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)    
    return pcd

def get_xyz_from_pts(u, v, depth, cx, cy, fx, fy, kernel_size=5):
    # print(f'get_xyz_from_pts = {pts_row}, depth = {depth.shape}')
    d = depth[int(v), int(u)] # height, width in depth image
    h, w = depth.shape
    r = kernel_size
    row1 = v - r if v - r >= 0 else 0
    col1 = u - r if u - r >= 0 else 0 
    row2 = v + r if v + r <= h else 0
    col2 = u + r if u + r <= w else 0 

    kernel = gaussian_kernel(kernel_size)
    new_depth = depth[row1:row2,col1:col2]
    est_depth = np.sum(kernel*new_depth)
    
   
    m, std = np.mean(new_depth), np.sqrt(np.var(new_depth))
    # print(f'\tnew_depth = {new_depth.shape}, m={m:02f}, std={std:02f}')
    if m ==0:
        ratio = 1.0
    else:
        ratio = std/m
    # print(f'Get xyz:  {row1:03d}, {row2:03d}, {col1:03d}, {col2:03d}. Mean : {m:05f}. Std depth {std:05f}. Ratio {ratio:02f}')

    x = ((u - cx) / fx) * est_depth
    y = ((v - cy) / fy )* est_depth
    return np.array([x, y, est_depth]).transpose(), ratio
    # return np.array([0, 0, d]).transpose()



def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    
import torch
from chamfer_distance import ChamferDistance as chamfer_dist
def chamfer(ref, target):
    sr, st = ref.shape, target.shape
    p1 = torch.from_numpy(ref).reshape(1, sr[0], sr[1]).float().to('cuda')
    p2 = torch.from_numpy(target).reshape(1, st[0], st[1]).float().to('cuda')
    chd = chamfer_dist().to('cuda')
    dist1, dist2, idx1, idx2 = chd(p1,p2)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    return loss.cpu().item()

def get_center_seg(seg):
    indices = (torch.from_numpy(seg) > 0).nonzero()
    if indices.shape[0] > 0:
        uy, ux = torch.mean(indices.float(), dim=0)
        uy, ux = int(uy), int(ux)
    return [ux, uy]

def filter_outliers(segs, depth, cfg):
    refined_pc_segs = {}
    pc_segs = {}
    for k in segs:
        seg = segs[k]
        points = extract_3D_points(seg, depth, cfg.cx, cfg.cy, cfg.fx, cfg.fy)
        pcd = np_to_pc(points)
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        np_points = np.asarray(pcd.points)[:, 0:3]
        labels = np.array(pcd.cluster_dbscan(eps=cfg.eps, min_points=cfg.min_points, print_progress=False))
        max_label = labels.max()
        # pcd_list = []
        search_pc, max_size = None, 0
        for l in range(0, max_label+1):
            pc = np_points[labels == l]
            if pc.shape[0] > max_size and pc.shape[0] > cfg.seg_threshold:
                search_pc = pc
                max_size = pc.shape[0]
        if type(search_pc) == np.ndarray:
            refined_pc_segs[k] = search_pc
            # print(f' id = {k}, search_pc = {search_pc.shape}')
    # o3d.visualization.draw_geometries([np_to_pc(refined_segs[k]).transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) for k in refined_segs.keys()],
                                    #   width=640, height=480, left=50, top=50)
        pc_segs[k] = np_points
    return refined_pc_segs, pc_segs

def cal_dist(segs, depth, cfg):
    # Fill out segment
    refined_pc_segs, pc_segs = filter_outliers(segs, depth, cfg)
    keys = refined_pc_segs.keys()
    dist = {}
    for pk1 in keys:
        for pk2 in keys:
            if pk1 != pk2:
                pc1, pc2 = refined_pc_segs[pk1], refined_pc_segs[pk2] 
                c1, c2 = np.mean(pc1, axis=0), np.mean(pc2, axis=0)
                dist3d = np.linalg.norm(c1 - c2)
                dist[f'{pk1}-{pk2}'] = dist3d

    return dist, refined_pc_segs, pc_segs

import seaborn as sns
def paint_pc(pc_segs, colors= np.asarray(sns.color_palette("Set2", 10))):
    pcd = o3d.geometry.PointCloud()
    for i, k in enumerate(pc_segs.keys()):
        pc = np_to_pc(pc_segs[k]).transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pc.paint_uniform_color(colors[i])
        pcd += pc
    return pcd



if __name__ == '__main__':
    # import seaborn as sns
    # colors = np.asarray(sns.color_palette())
    # print(colors)
    s = gaussian_kernel()
    print(type(s))
    print(np.sum(s))
    pass