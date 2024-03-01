import os

def read_file_list(filename):
    """
    Read list files of  RGB or Depth 
    
    Parameters:
    ----------
    filename: path the relative list file
    Example: grb.txt or depth.txt
    Content: 
    For RGB:
        1341845948.747856 rgb/1341845948.747856.png
        ...
    Fro Depth:
        1341845948.747899 depth/1341845948.747899.png
        ...
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def read_tum_data_set(base_path, dataset_name, offset=0.0,max_difference=0.05):
    """
    Read full path of RGB and DEPTH files
    
    Parameters
    ----------
    base_path: base path, example: /mnt/3d/map3d/data/tum/extract
    dataset_name: example: rgbd_dataset_freiburg3_walking_halfsphere
    """
    # base_path = '/mnt/3d/map3d/data/tum/extract'
    dataset_path = os.path.join(base_path, dataset_name)
    # print(f'dataset_path = {dataset_path}')
    first_list = read_file_list(os.path.join(dataset_path, 'depth.txt'))
    second_list = read_file_list(os.path.join(dataset_path, 'rgb.txt'))
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    color_files = []
    depth_files = []
    for a, b in matches:
        # files.append((first_list[a][0], second_list[b][0]))
        color_files.append(os.path.join(dataset_path, first_list[a][0]))
        depth_files.append(os.path.join(dataset_path, second_list[b][0]))
    print('Number of files = ', len(depth_files))
    return depth_files, color_files, matches


def make_tum_association(base_path, dataset_name,offset=0.0,max_difference=0.2):
    """
    Create association between RGB and DEPTH with time aligments
    """
    dataset_path = os.path.join(base_path, dataset_name)
    first_list = read_file_list(os.path.join(dataset_path, 'depth.txt'))
    second_list = read_file_list(os.path.join(dataset_path, 'rgb.txt'))
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    print("Number of points = ", len(matches))
    # color_files = []
    # depth_files = []
    file = open('%s/%s.txt' %(base_path, dataset_name), 'w')
    for a, b in matches:
        text = '%s %s %s %s\n' % (b, second_list[b][0], a, first_list[a][0])
        # print(text)
        file.write(text)
    file.close()

import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)

if __name__ == '__main__':
    #rgbd_dataset_freiburg3_sitting_static
    conf = parser.parse_args()
    print('Dataset name = ', conf.name)
    make_tum_association('/mnt/3d/map3d/data/tum/extract', conf.name)
    