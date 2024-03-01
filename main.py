import random
import cv2
import numpy as np
from ultralytics import YOLO
import os
import copy
import torch
import matplotlib.pyplot as plt

import utils
from config import ConfigParser
from os import path
import open3d as o3d
import seaborn as sns

cfg = ConfigParser().parse_args()
cfg_dict = vars(cfg) # Convert Namespace into dictionary
for k in cfg_dict:
    print(f'{k} : {cfg_dict[k]}')
print('================================================')
        
list_rgb_depth = utils.load_filenames(path.join(cfg.associations, f'{cfg.dataset_name}.txt'))
print('Number of files: ', len(list_rgb_depth))          

base_path = path.join(cfg.dataset_path, cfg.dataset_name)                            
print(f'base_path = {base_path}')

yolo_seg = cfg.yolo_model
model = YOLO(yolo_seg)
names = model.model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

os.system(f'rm -f {os.path.join(base_path, "seg/*.png") }') # Delete previous results
plt.figure(figsize=(5*2, 5*2))
history = {}
history['pc'] = []
history['2d'] = []

# N = len(list_rgb_depth)
start = 1
count = 1
step = 1
N = 2
out = cv2.VideoWriter(os.path.join(base_path, 'video.mp4'), cv2.VideoWriter_fourcc('M','J','P','G'), 15.0 , (640, 480))

rgb, depth = None, None
# view_pcds = []
while count <= N: #for each rgb
    
    #RGB
    index = start + (count-1)*step
    img_path = os.path.join(base_path, list_rgb_depth[index][0])
    rgb = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) #read rgb
    seg_img = copy.deepcopy(rgb)
    
    print(f'Count = {count} = {img_path}')

    # Depth
    depth_path = os.path.join(base_path, list_rgb_depth[index][1])
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)/cfg.depth_scale
    
    
    results = model.track(img_path, verbose=False, conf=cfg.conf_threshold)
    h, w, _ = rgb.shape
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

    if masks is not None:
        masks = masks.data.cpu()
     
    static_masks = np.zeros((h,w))
    dynamic_masks = np.zeros((h,w))
    all_masks = np.zeros((h,w))
    segs = {}
 
    if masks != None:
        for seg, box in zip(masks.numpy(), boxes):
            cls = int(box.cls.item())
            id = int(box.id.item())
            seg = cv2.resize(seg, (w, h))
            all_masks = np.logical_or(all_masks > 0, seg > 0)
            center = utils.get_center_seg(seg)
            ux, uy = center
            cv2.circle(rgb, (ux, uy), 7, colors[cls], -1)
            cv2.putText(rgb, f'{cls}:{id}', (ux-10, uy), 
                             cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 1, cv2.LINE_AA)
            key = f'{cls}:{id}'
            segs[key] =  seg
    
    cv2.putText(rgb, f'Frame {count:02d}', (50, 30), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    
    curr_dist, refined_pc_segs,  pc_segs = utils.cal_dist(segs, depth, cfg)
    if count > 1:
        prev_segs = history['segs']
        prev_depth = history['depth']
        prev_dist, prev_refined_pc_segs,  prev_pc_segs = utils.cal_dist(prev_segs, prev_depth, cfg)
        
        # Voting for dynamic one
        diff_dict = {}
        dynamic_ids = []
        for pk in prev_dist:
            for ck in curr_dist:
                if ck == pk:
                    diff = np.abs(prev_dist[pk] - curr_dist[ck])
                    cls_id = ck.split('-')[0]
                    if cls_id in diff_dict.keys():
                        if diff > cfg.vote_threshold:
                            diff_dict[cls_id] += 1
                    else:
                        diff_dict[cls_id] = 0
        
        for k in diff_dict:
            if diff_dict[k] > 1:
                dynamic_masks = np.logical_or(dynamic_masks > 0, segs[k])
                dynamic_ids.append(k)
    
        # Measure the deformed objects if their centers don't move
        for pk in prev_refined_pc_segs:
            for ck in refined_pc_segs:
                if ck == pk and ck not in dynamic_ids:
                    pc1, pc2 = prev_refined_pc_segs[ck], refined_pc_segs[ck]
                    cd = utils.chamfer(pc1, pc2)
                    cls = int(ck.split(':')[0])
                    if cls == 0 and cd > cfg.deform_threshold:
                        dynamic_masks = np.logical_or(dynamic_masks > 0, segs[ck])
                
    rgb = utils.overlay(rgb, dynamic_masks)
        
    
    if cfg.use_visualization:
        # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        if count > 1:
            plt.subplot(2,2, 1); plt.imshow(history['rgb'] ); plt.axis('off')
            plt.subplot(2,2, 2); plt.imshow(history['depth']); plt.axis('off')
        plt.subplot(2,2, 3); plt.imshow(rgb); plt.axis('off')
        plt.subplot(2,2, 4); plt.imshow(depth) ; plt.axis('off')
        # plt.show()
        vs = 0.001
        if count == 1:
            before_pc, after_pc = utils.paint_pc(pc_segs).voxel_down_sample(voxel_size=vs), utils.paint_pc(refined_pc_segs).voxel_down_sample(voxel_size=vs)
            # after_pc = after_pc.translate([-1.5, 0, 0])
        else:
            # colors = np.flip(np.array(sns.color_palette("tab10", 10)))
            colors = np.array(sns.color_palette("tab10", 10))
            before_pc, after_pc = utils.paint_pc(pc_segs, colors).voxel_down_sample(voxel_size=vs), utils.paint_pc(refined_pc_segs, colors).voxel_down_sample(voxel_size=vs)
            # after_pc = after_pc.translate([1.5, 0, 0])
            # o3d.visualization.draw_geometries([before_pc],width=1080, height=800, left=50, top=50)
            # o3d.visualization.draw_geometries([after_pc],width=1080, height=800, left=50, top=50)
        # o3d.visualization.draw_geometries([before_pc],  width=800, height=600, left=50, top=50, window_name='Before Refining')
        # o3d.visualization.draw_geometries([after_pc], width=800, height=600, left=50, top=50, window_name='After Refining')
        
        offset = 1.5
        # o3d.visualization.draw_geometries([before_pc.translate([-offset, 0, 0]), after_pc.translate([offset, 0, 0])],width=1080, height=800, left=50, top=50)
        
        # view_pcds.append(after_pc)
        
        o3d.io.write_point_cloud(f'pc/pc_before_{count}.pcd', before_pc, write_ascii=True)
        o3d.io.write_point_cloud(f'pc/pc_after_{count}.pcd', after_pc, write_ascii=True)
        pass
    
    # Misdetection: use information from previous frame
    
    history['segs'] = segs
    history['depth'] = depth
    history['rgb'] = rgb
    img_out_path = os.path.join(base_path, 'seg',  img_path.split('/')[-1])
    # print(f' Write Frame {count:05d} at {path}')
    cv2.imwrite(img_out_path, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
    out.write(cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))
    count += 1  #Next frame
    # break
# End while
out.release()

# o3d.visualization.draw_geometries(view_pcds, width=1080, height=800, left=50, top=50)
# pc1, pc2 = view_pcds
# o3d.visualization.draw_geometries([pc1.translate([-1, 0, 0]), pc2.translate([1, 0, 0])])

exit()
base_path = f'/mnt/3d/map3d/data/tum/extract/{cfg.dataset_name}' 
src = os.path.join(base_path, 'seg/*.png')
dst = f'tuandang@nuc:/home/tuandang/slam/data/extract/{cfg.dataset_name}/rgb'
os.system(f'scp {src} {dst}')