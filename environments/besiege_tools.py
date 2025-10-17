import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations
from scipy.spatial.transform import Rotation as R
import uuid
from scipy.spatial import ConvexHull
import re
from AgenticCodes.prompts import *
from AgenticCodes.config import BLOCKPROPERTYPATH,FORBIDEN_BLOCKS,WHEEL_AUTO_ON,LINEAR_BLOCKS
from collections import deque, defaultdict
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations
from scipy.spatial.transform import Rotation as R
import uuid
from scipy.spatial import ConvexHull
from copy import deepcopy

def reorder_parents(lst):
    nodes = [dict(d) for d in lst if 'parent' in d]
    lines = [dict(d) for d in lst if 'parent_a' in d]

    oid2idx = {d['order_id']: i for i, d in enumerate(nodes)}
    root = next((i for i, d in enumerate(nodes) if d['parent'] == -1 and d['bp_id'] == -1), None)

    g = defaultdict(list)
    for i, d in enumerate(nodes):
        p = d['parent']
        if p != -1 and p in oid2idx:
            g[oid2idx[p]].append(i)

    keep = set()
    if root is not None:
        q = deque([root])
        while q:
            u = q.popleft()
            if u in keep:      
                continue
            keep.add(u)
            q.extend(g[u])

    new_nodes = [nodes[i] for i in keep]
    new_oid2idx = {d['order_id']: new_id for new_id, d in enumerate(new_nodes)}

    new_lines = [
        d for d in lines
        if d['parent_a'] in new_oid2idx and d['parent_b'] in new_oid2idx
    ]

    for new_id, d in enumerate(new_nodes):
        d['order_id'] = new_id
        p = d['parent']
        d['parent'] = -1 if p == -1 else new_oid2idx.get(p, -1)

    for d in new_lines:
        d['order_id'] = len(new_nodes) + new_lines.index(d)
        d['parent_a'] = new_oid2idx[d['parent_a']]
        d['parent_b'] = new_oid2idx[d['parent_b']]

    return new_nodes + new_lines

def generate_guid():
    return str(uuid.uuid4())

def facing(q_in):
    q_z_pos = np.array([0, 0, 0, 1])
    q_z_neg = np.array([0, 1, 0, 0])
    q_x_neg = np.array([0, -0.7071068, 0, 0.7071068])
    q_x_pos = np.array([0, 0.7071068, 0, 0.7071068])
    q_y_pos = np.array([-0.7071068,0, 0,0.7071068])
    q_y_neg = np.array([0.7071068,0, 0,0.7071068])

    angle_threshold = 1e-3
    rots = [q_z_pos,q_z_neg,q_x_neg,q_x_pos,q_y_pos,q_y_neg]
    facing = ["z+","z-","x-","x+","y+","y-"]
    r1 = R.from_quat(q_in)
    for q2 in range(len(rots)):
        r2 = R.from_quat(rots[q2])
    
        relative_rotation = r1.inv() * r2
        angle = relative_rotation.magnitude()
    
        if(angle < angle_threshold):
            return facing[q2]
    
    return "Error!Can not found correct direction"


def get_block_info(xml_file):
    """
    Get Block info from XML
    """
    try:
        tree = ET.parse(xml_file)
    except:
        tree = ET.ElementTree(ET.fromstring(xml_file))
    root = tree.getroot()
    
    block_info = {}
    
    global_position = root.find('Global').find('Position')
    x, y, z = map(float, (global_position.get('x'), global_position.get('y'), global_position.get('z')))
    global_rotation = root.find('Global').find('Rotation')
    qx, qy, qz, qw = map(float, (global_rotation.get('x'), global_rotation.get('y'), global_rotation.get('z'), global_rotation.get('w')))
    rotation_matrix = quaternion_to_rotation_matrix((qx, qy, qz, qw))
    block_info["global"]={
        'position':np.array([x, y, z]),
        'rotation':rotation_matrix,
    }
    
    for order_id,block in enumerate(root.findall('Blocks/Block')):
        block_id = block.get('id')
        guid = block.get('guid')
        
        transform = block.find('Transform')
        position = transform.find('Position')
        rotation = transform.find('Rotation')
        scale = transform.find('Scale')
        
        x, y, z = map(float, (position.get('x'), position.get('y'), position.get('z')))
        qx, qy, qz, qw = map(float, (rotation.get('x'), rotation.get('y'), rotation.get('z'), rotation.get('w')))
        sx, sy, sz = map(float, (scale.get('x'), scale.get('y'), scale.get('z')))
        
        rotation_matrix = quaternion_to_rotation_matrix((qx, qy, qz, qw))
        
        block_info[order_id] = {
            'id': block_id,
            'position': np.array([x, y, z]),
            'rotation_matrix': rotation_matrix,
            'scale': np.array([sx, sy, sz])
        }
        
        if block_id in LINEAR_BLOCKS:
            end_position=block.find('Data').find('./Vector3[@key="end-position"]')
            x, y, z = map(float, (end_position.find('X').text, end_position.find('Y').text, end_position.find('Z').text))
            block_info[order_id].update({'end_position':np.array([x, y, z])})
            
            end_rotation=block.find('Data').find('./Vector3[@key="end-rotation"]')
            x, y, z = map(float, (end_rotation.find('X').text, end_rotation.find('Y').text, end_rotation.find('Z').text))
            block_info[order_id].update({'end_rotation':np.array([x, y, z])})
        
        if block_id =='18':
            pre_extended =block.find('Data').find('./Boolean[@key="preextended"]').text
            block_info[order_id].update({'pre_extended':pre_extended})
            if pre_extended=='False':
                block_info[order_id].update({'id':'18_1'})
            

        
    
    return block_info



def plot_block(ax, corners, color='b', alpha=0.35):
    faces = [
        [corners[j] for j in [0, 1, 3, 2]],  
        [corners[j] for j in [4, 5, 7, 6]],  
        [corners[j] for j in [0, 1, 5, 4]],  
        [corners[j] for j in [2, 3, 7, 6]],  
        [corners[j] for j in [0, 2, 6, 4]],  
        [corners[j] for j in [1, 3, 7, 5]]   
    ]
    collection = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors=color, alpha=alpha)
    ax.add_collection3d(collection)
def plot_overlap(ax, corners1, corners2, color='r', alpha=0.5):
    plot_block(ax, corners1, color=color, alpha=alpha)
    plot_block(ax, corners2, color=color, alpha=alpha)
def plot_connections(ax, corners1, corners2, color='g', alpha=0.5):
    plot_block(ax, corners1, color=color, alpha=alpha)
    plot_block(ax, corners2, color=color, alpha=alpha)

def visualize_blocks(block_details, overlaps,connections):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("3D Blocks with Overlaps Highlighted")

    for block_info in block_details:
        if "GlobalPosition" in block_info: continue
        if "corners" in block_info:
            corners, _ = (block_info["corners"],block_info["id"])
        else:
            corners = block_info
        
        plot_block(ax, corners)

    for id1, id2, corners1, corners2 in overlaps:
        plot_overlap(ax, corners1, corners2)

    for id1, id2, corners1, corners2 in connections:
        plot_connections(ax, corners1, corners2)

    min_val = -10  
    max_val = 10   
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)
    ax.set_box_aspect([1, 1, 1])  

    ticks = np.arange(min_val, max_val + 0.5, 0.5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    plt.show()

def llm2xml(block_details,block_sizes):
    with open(block_sizes, 'r', encoding='utf-8') as file:
        block_sizes = json.load(file) 
    xml_block_details = []
    
    global_rt = block_details.pop(0)
    gp,gr = global_rt["GlobalPosition"],global_rt["GlobalRotation"]
    xml_block_details.append({"GlobalPosition":gp,"GlobalRotation":gr})
    gr = R.from_quat(gr).as_matrix()


    for block in block_details:
        corners_gp = block["corners"] #global
        id = block["id"]
        
        manu_gp = block["building_center"] #global
        scale = block["scale"]

        if id not in block_sizes:
            print(f"Warning: size for block ID {id} not defined in block_sizes, skipping this block.")
            continue

        xml_block_infos={}

        block_ori_infos = block_sizes[id] 

        manu_lp = np.linalg.inv(gr)@(manu_gp-gp)

        # print(manu_lp)

        corners_lp = []
        for corner in corners_gp:
            corner_lp = np.linalg.inv(gr)@(corner-gp)
            corners_lp.append(corner_lp)
        corners_lp = np.array(corners_lp)
        
        geo_gp = np.mean(corners_gp, axis=0)
        geo_lp = np.mean(corners_lp, axis=0)
        bc_gc_generated = geo_lp-manu_lp
        bc_gc = block_ori_infos['bc_gc']

        length_generated = np.linalg.norm(bc_gc_generated/scale)
        length_bc_gc = np.linalg.norm(bc_gc)
        if abs(length_generated-length_bc_gc)>1e-4:
            print(block)
            print(length_generated)
            print(length_bc_gc)
            return "Block generation error"
        
        

        bbox_size = block_ori_infos['bbox_size']
        half_bbox_size = np.array(bbox_size) / 2.0
        bbox_lp = []
        for z in [-1, 1]:
            for x in [-1, 1]:
                for y in [-1, 1]:
                    point = (manu_lp+bc_gc) + (x * half_bbox_size[0], y * half_bbox_size[1], z * half_bbox_size[2])
                    bc_point = point-manu_lp
                    point_lp =  manu_lp+bc_point*scale 
                    bbox_lp.append(tuple(point_lp))
        bbox_lp = np.array(bbox_lp)

        if id=="0":
            manu_lr =calculate_startingblock_quaternion(bbox_lp,corners_lp,manu_lp)
        else:
            coplanar_points_generated = find_bottom_face_points(corners_lp,manu_lp)
            coplanar_points = bbox_lp[0:4]
            initial_center = (manu_lp+bc_gc)
            manu_lr = find_rotation_quaternion(coplanar_points, coplanar_points_generated, initial_center, geo_lp,manu_lp)

        xml_block_infos["id"] = id
        xml_block_infos["Transform"] = {"Position":manu_lp,"Rotation":manu_lr,"Scale":scale}
        # xml_block_infos["Data"] = {}
        xml_block_infos["guid"] = generate_guid()
        xml_block_details.append(xml_block_infos)
    return xml_block_details

def xml2json(xml_file, block_sizes):
    block_info = get_block_info(xml_file)
    # print(block_info)

    global_info = block_info.pop('global')
    # gp =  np.around(global_info['position'], decimals=3)
    # gr =  np.around(global_info['rotation'], decimals=3)
    # print(global_info)
    gp =  global_info['position']
    gr =  global_info['rotation']

    with open(block_sizes, 'r', encoding='utf-8') as file:
        block_sizes = json.load(file)  

    block_details = []
    # block_details.append({"GlobalPosition":gp,"GlobalRotation":R.from_matrix(gr).as_quat()})
    # index=0

    for guid, info in block_info.items():
        # index+=1
        # if index>2:
        #     continue
        # if index==1: continue

        block_id = info['id']
        if int(block_id) in FORBIDEN_BLOCKS:
            return False
        
        if block_id not in block_sizes:
            continue

        if str(block_id) in LINEAR_BLOCKS:
            new_detail_infos = {}
            new_detail_infos["id"]  = block_id
            manu_lp_a = info['position']
            manu_gp_a = get_relative_pos(manu_lp_a,gp,gr)
            
            global_rot = R.from_matrix(info["rotation_matrix"])
            # end_rot = R.from_euler('xyz', info["end_rotation"], degrees=True)
            # manu_lp_b = global_rot.apply(end_rot.apply(info["end_position"]))+manu_lp_a
            manu_lp_b = global_rot.apply(info["end_position"])+manu_lp_a
            manu_gp_b = get_relative_pos(manu_lp_b,gp,gr)
            new_detail_infos["building_center"]=manu_gp_a
            new_detail_infos["building_center_b"]=manu_gp_b
            block_details.append(new_detail_infos)
            continue

        block_ori_infos = block_sizes[block_id]  

        # print(block_id)
        # print(info)

        bbox_gp,manu_gp,bp_gp,bbox_lp,bp_lp,my_building_points,my_building_points_buildrotation = get_3d_from_xml(block_ori_infos,info,gp,gr,log=False)
        

        new_detail_infos = {}
        # new_detail_infos["corners"]  =  np.around(bbox_gp, decimals=3)
        new_detail_infos["id"]  = block_id
        new_detail_infos["building_center"]  =  manu_gp
        new_detail_infos["bp_gp"]  =  bp_gp
        # new_detail_infos["scale"]  =  np.around(info['scale'], decimals=3)
        block_details.append(new_detail_infos)
    
    def row_id_or_false(bp_gp: np.ndarray, building_center: np.ndarray, tol: float = 0.1):
        dists = np.round(np.linalg.norm(bp_gp - building_center, axis=1),decimals=2) 
        idx   = np.where(dists <= tol)[0]
        # print(dists)
        return (idx[0],bp_gp,dists) if idx.size else ('False',bp_gp,dists)
    
    def get_parent(building_center,order_id,block_details):
        parent = None
        bp_id = None
        best_candidate = None
        best_candidate_dists = np.array([100])
        best_candidate_orderid = None
        # for prev_order,prev_block in enumerate(block_details[:order_id]):
        for prev_order,prev_block in enumerate(block_details):
            
            if prev_block['id'] in LINEAR_BLOCKS: continue
            building_points = prev_block['bp_gp']
            if building_points.shape==(0,):continue
            # print(building_points)
            # print(building_center)
            # print(f"prev_order:{prev_order},order_id:{block_details[prev_order]}")
            check_results = row_id_or_false(building_points,building_center)
            # return
            if check_results[0]!='False':
                if parent !=None:
                    if np.min(check_results[2])<np.min(best_candidate_dists):
                        best_candidate = check_results[1]
                        best_candidate_dists = check_results[2]
                        best_candidate_orderid = prev_order
                        parent = prev_order
                        bp_id = check_results[0]
                else:  
                    parent=prev_order
                    bp_id = check_results[0]
                    candidate = check_results[1]
                    candidate_dists = check_results[2]
                    best_candidate = candidate
                    best_candidate_dists = candidate_dists
                    best_candidate_orderid = prev_order
            else:
                fail,candidate,candidate_dists = check_results
                if np.min(candidate_dists)<np.min(best_candidate_dists):
                    best_candidate = candidate
                    best_candidate_dists = candidate_dists
                    best_candidate_orderid = prev_order
        if parent ==None:
            
            if min(best_candidate_dists)>=0.5: 
                return int(-1),int(-1)
            
            
            # print(building_center)
            # print(best_candidate_orderid)
            # print(best_candidate)
            # print(best_candidate_dists) 
            raise ValueError('Some Blocks Can not found parents, this may happen in human crafted machine')
        
        return int(parent),int(bp_id)
    
    # print(block_details)
    
    machine_json=[]
    for order_id,block_detail in enumerate(block_details):
        type_id = str(block_detail['id'])
        if order_id==0:
            parent=-1
            bp_id=-1
            machine_json.append(
                {"id":type_id,"order_id":order_id,"parent":parent,"bp_id":bp_id}
            )
        elif type_id not in LINEAR_BLOCKS and order_id!=0:

            building_center = block_detail['building_center']
            parent,bp_id = get_parent(building_center,order_id,block_details)
            machine_json.append(
                {"id":type_id,"order_id":order_id,"parent":parent,"bp_id":bp_id}
            )
        else:
            
            building_center_a = block_detail['building_center']
            parent_a,bp_id_a = get_parent(building_center_a,order_id,block_details)
            
            building_center_b = block_detail['building_center_b']
            parent_b,bp_id_b = get_parent(building_center_b,order_id,block_details)

            machine_json.append(
                {"id":type_id,"order_id":order_id,"parent_a":parent_a,"bp_id_a":bp_id_a,"parent_b":parent_b,"bp_id_b":bp_id_b}
            )
    
    return reorder_parents(machine_json)

def delete_blocks(data, to_delete):
    delete_indices = set()
    order_id_map = {}  
    current_order = 0
    
    for i, item in enumerate(data):
        if "order_id" in item:
            if item["order_id"] in to_delete and item.get("parent", 0) != -1:
                delete_indices.add(i)
            else:
                order_id_map[item["order_id"]] = current_order
                current_order += 1
    
    new_data = []
    for i, item in enumerate(data):
        if i in delete_indices:
            continue
        
        if "order_id" in item:
            new_item = item.copy()
            new_item["order_id"] = order_id_map[item["order_id"]]
            
            if "parent" in new_item and new_item["parent"] != -1:
                pass
            new_data.append(new_item)
        else:
            new_data.append(item)
    
    for item in new_data:
        if "parent" in item and item["parent"] != -1:
            original_parent_order = None
            for orig_item in data:
                if "order_id" in orig_item and orig_item["order_id"] == item["parent"]:
                    original_parent_order = orig_item["order_id"]
                    break
            
            if original_parent_order in order_id_map:
                item["parent"] = order_id_map[original_parent_order]
            else:
                pass
    
    return new_data


def llm2xml_filetree(block_details, block_sizes_path, selected_menu=None):
    with open(block_sizes_path, 'r', encoding='utf-8') as file:
        block_sizes = json.load(file)
    
    global_rt = block_details.pop(0)
    gp, gr_quat = global_rt["GlobalPosition"], global_rt["GlobalRotation"]
    gr_matrix = R.from_quat(gr_quat).as_matrix()

    blocks_to_delete = set()
    blocks_to_delete_feedback = []

    linear = {"id","order_id","parent_a", "bp_id_a", "parent_b", "bp_id_b"}
    non_linear = {"id","order_id", "parent", "bp_id"}
    for i, block in enumerate(block_details):
        if not (set(block.keys()) == linear or set(block.keys()) == non_linear):
            blocks_to_delete.add(i)
            blocks_to_delete_feedback.append(
                f"Warning: Block(orderID {i})structure illegal"
            )
            
    order_id_map = {int(b["order_id"]): b for b in block_details} 

    for i,block in enumerate(block_details):
        is_linear = False
        parent_order_a=-1
        parent_order_b=-1

        format_error = False
        for k,v in block.items():
            if k =="id": 
               if not isinstance(v,str):
                   if isinstance(v,int):
                        v = str(v)
                   else:
                        format_error = True
                   
            if k in["order_id","parent_a", "bp_id_a", "parent_b", "bp_id_b", "parent", "bp_id"]: 
               if not isinstance(v,int):
                   if isinstance(v,str):
                       try:
                           v = int(v)
                       except:
                            format_error = True
        
        if format_error:
            
            blocks_to_delete.add(i)
            blocks_to_delete_feedback.append(f"Warning:order{i}json format illegal")
            continue

        if i==0:
            block_type = str(block["id"])
            order_id = int(block["order_id"])
            parent_order = int(block.get("parent", -2))
            bp_id = int(block.get("bp_id", -2))
            if any([block_type!="0",order_id!=0]):
                blocks_to_delete.add(i)
                blocks_to_delete_feedback.append(f"Warning: startingBlock illegal")
                continue
            if any([parent_order!=-1,bp_id!=-1]):
                block["parent"]=-1
                block["bp_id"]=-1


        order_id = int(block["order_id"])
        parent_order = int(block.get("parent", -1))
        if parent_order==-1 and order_id!=0:
            is_linear = True
            parent_order_a = int(block.get("parent_a", -1))
            parent_order_b = int(block.get("parent_b", -1))
            parents = [parent_order_a,parent_order_b]
        else:
            parents = [parent_order]

        if any(order in blocks_to_delete for order in parents):
            blocks_to_delete.add(order_id)
            blocks_to_delete_feedback.append(f"Warning: Block(orderID {order_id})parent block(orderID {parent_order})illegal, so it is illegal too.")
            continue 

        for i_th_parent,parent_order in enumerate(parents):
            parent_block = order_id_map.get(parent_order)
            if parent_block:
                parent_block_id = str(parent_block["id"])
                if i_th_parent==0:
                    bp_id = int(block.get("bp_id",block.get("bp_id_a",-1)))
                elif i_th_parent==1:
                    bp_id = int(block.get("bp_id_b",-1))
                else:
                    bp_id=-1
                if parent_block_id in block_sizes and bp_id >= len(block_sizes[parent_block_id]["bc_bp"]):
                    blocks_to_delete.add(order_id)
                    blocks_to_delete_feedback.append(f"Warning: Block(orderID {order_id})parent block(ID {parent_block_id})does not exist Constructible point{bp_id}。")
                    continue
        
        if (not is_linear) and str(block.get("id")) in LINEAR_BLOCKS:
            blocks_to_delete.add(order_id)
            blocks_to_delete_feedback.append(f"Warning: Block(orderID {order_id})is linear block but not exist two parents")
            continue
        elif is_linear and (str(block.get("id")) not in LINEAR_BLOCKS):
            blocks_to_delete.add(order_id)
            blocks_to_delete_feedback.append(f"Warning: Block(orderID {order_id})exist two parents but not a linear block")
            continue
    
    # print(blocks_to_delete_feedback)
    
    if blocks_to_delete:
        block_details = [b for b in block_details if int(b["order_id"]) not in blocks_to_delete]

    processed_details = get_3d_from_llm(block_sizes, block_details, gp, gr_matrix, log=False)
    # print(block_details)
    xml_block_details = [{"GlobalPosition": gp, "GlobalRotation": gr_quat}]
    for block in processed_details:
        xml_info = {
            "id": block["id"],
            "order_id": block["order_id"],
            "guid": generate_guid()
        }
        
        if str(block["id"]) in LINEAR_BLOCKS:
            xml_info["Transform"] = {"Position": block["manu_lp_a"], "Rotation": np.array([0,0,0,1]), "Scale": block["scale"]}
            xml_info["end-position"] = block["manu_lp_b"] - block["manu_lp_a"]
        else: 
            manu_lr = R.from_matrix(block["manu_lr"]).as_quat() if block["manu_lr"].shape == (3, 3) else block["manu_lr"]
            xml_info["Transform"] = {"Position": block["manu_lp"], "Rotation": manu_lr, "Scale": block["scale"]}
            if "flip" in block: 
                if WHEEL_AUTO_ON:
                    wheel_auto=True
                else:
                    wheel_auto=False
                xml_info.update({"flip": block["flip"], "auto": wheel_auto, "autobrake": False})
                if selected_menu and "special_props" in selected_menu:
                    xml_info["WheelDoubleSpeed"] = "WheelDoubleSpeed" in selected_menu["special_props"]

        xml_block_details.append(xml_info)
    
    # print("\n".join(blocks_to_delete_feedback))
    
    return xml_block_details, processed_details, "\n".join(blocks_to_delete_feedback)

def check_overlap(block_details,vis=True,corners_parent_llm_parent=None):

    def overlap_log(id1,id2):
        head1 = "Block order_id"
        head2 = "and Block order_id"
        overlap_head = "overlap"
        return f"{head1} {id1} {head2} {id2} {overlap_head}\n"

    
    overlaps = []
    connections = []

    overlaps_info=""
    # print(len(block_details))
    # print(len(corners_parent_llm_parent))
    for i in range(len(block_details)):
        # print(block_details[i])
        for j in range(i + 1, len(block_details)):
            if "GlobalPosition" in block_details[i] or "GlobalPosition" in block_details[j]:continue
            # if np.all(block_details[i] == 0) or np.all(block_details[j] == 0): continue
            if "corners" in block_details[i] and "corners" in block_details[j]:
                corners1, id1 = (block_details[i]["corners"],i)
                corners2, id2 = (block_details[j]["corners"],j)
            else:
                corners1 = block_details[i]
                id1 = i
                corners2 = block_details[j]
                id2 = j
            
            
            results = check_overlap_or_connection(corners1, corners2)
            if results=="connected":
                
                connections.append((id1, id2, corners1, corners2))
            elif results:
                if corners_parent_llm_parent !=None:
                    id1_type = str(corners_parent_llm_parent[id1][0])
                    id1_order = str(corners_parent_llm_parent[id1][1])
                    id1_parent_order = str(corners_parent_llm_parent[id1][2])
                    id2_type = str(corners_parent_llm_parent[id2][0])
                    id2_order = str(corners_parent_llm_parent[id2][1])
                    id2_parent_order = str(corners_parent_llm_parent[id2][2])
                    if id1_order==id2_parent_order:
                        if str(id1_type)=="30":
                            pass
                        else:
                            
                            overlaps_info+=overlap_log(id1,id2)
                            overlaps.append((id1, id2, corners1, corners2))
                    elif id2_order==id1_parent_order:
                        if str(id2_type)=="30":
                            pass
                        else:
                            overlaps_info+=overlap_log(id1,id2)
                            overlaps.append((id1, id2, corners1, corners2))
                    else:
                        overlaps_info+=overlap_log(id1,id2)
                        overlaps.append((id1, id2, corners1, corners2))
                else:
                    overlaps_info+=overlap_log(id1,id2)
                    overlaps.append((id1, id2, corners1, corners2))

    if overlaps:
        found_head = "totally"
        overlaps_head="overlaps"
        overlaps_info+=f"{found_head} {len(overlaps)} {overlaps_head}\n"
    else:
        overlaps_info+="no error"

    if vis:
        visualize_blocks(block_details, overlaps,connections)
    
    #print(overlaps_info)
    return overlaps_info


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() 
        return super().default(obj)

def convert_to_numpy(data):
    no_globalrt = True
    for info in data:
        if "GlobalPosition" in info:
            info["GlobalPosition"] = np.array(info["GlobalPosition"])
            info["GlobalRotation"] = np.array(info["GlobalRotation"])
            no_globalrt = False
        else:

            keys_to_convert = ["corners", "building_center", "scale","manu_lr","manu_lp","bp_lr"]
            for key in keys_to_convert:
                if key in info:
                    info[key] = np.array(info[key])

    new_data = [{"GlobalPosition":np.array([0,5.05,0]),"GlobalRotation":np.array([0,0,0,1])}]
    if no_globalrt:
        new_data.extend(data)
        return new_data

    return data 

def create_xml(data):

    machine = ET.Element("Machine", version="1", bsgVersion="1.3", name="gpt")

    global_elem = ET.SubElement(machine, "Global")
    global_infos = data.pop(0)
    gp = global_infos["GlobalPosition"]
    gr = global_infos["GlobalRotation"]
    position = ET.SubElement(global_elem, "Position", x=str(gp[0]), y=str(gp[1]), z=str(gp[2]))
    rotation = ET.SubElement(global_elem, "Rotation", x=str(gr[0]), y=str(gr[1]), z=str(gr[2]), w=str(gr[3]))

    data_elem = ET.SubElement(machine, "Data")
    string_array = ET.SubElement(data_elem, "StringArray", key="requiredMods")

    blocks_elem = ET.SubElement(machine, "Blocks")

    for info in data:
        
        block_id = info['id']
        
        if info['id']=='18_1':
            block_id ='18'
        
        block = ET.SubElement(blocks_elem, "Block", id=str(block_id), guid=info['guid'])
        transform = ET.SubElement(block, "Transform")
        info_p = info['Transform']['Position']
        position = ET.SubElement(transform, "Position", x=str(info_p[0]), y=str(info_p[1]), z=str(info_p[2]))
        info_r = info['Transform']['Rotation']
        rotation = ET.SubElement(transform, "Rotation", x=str(info_r[0]), y=str(info_r[1]), z=str(info_r[2]), w=str(info_r[3]))
        info_s = info['Transform']['Scale']
        scale = ET.SubElement(transform, "Scale", x=str(info_s[0]), y=str(info_s[1]), z=str(info_s[2]))
        block_data = ET.SubElement(block, "Data")
        if str(info['id'])=="0":
            bmt = ET.SubElement(block_data, "Integer", key="bmt-version")
            bmt.text = "1"
        
        if str(info['id'])=="9":
            bmt = ET.SubElement(block_data,"Single",key = "bmt-slider")
            bmt.text = "10"
            bmt = ET.SubElement(block_data,"StringArray",key = "bmt-contract")
            bmt.text = "L"
            bmt = ET.SubElement(block_data,"Boolean",key = "bmt-toggle")
            bmt.text = "False"

        if str(info['id'])in LINEAR_BLOCKS:
            start_position = ET.SubElement(block_data,"Vector3",key = "start-position")
            ET.SubElement(start_position, "X").text = str(0)
            ET.SubElement(start_position, "Y").text = str(0)
            ET.SubElement(start_position, "Z").text = str(0)
            end_position = ET.SubElement(block_data,"Vector3",key = "end-position")
            ET.SubElement(end_position, "X").text = str(info['end-position'][0])
            ET.SubElement(end_position, "Y").text = str(info['end-position'][1])
            ET.SubElement(end_position, "Z").text = str(info['end-position'][2])

        
        
        if str(info['id'])=="22":
            bmt = ET.SubElement(block_data, "Integer", key="bmt-version")
            bmt.text = "1"
            bmt = ET.SubElement(block_data,"Single",key = "bmt-speed")
            bmt.text = "1"
            bmt = ET.SubElement(block_data,"Single",key = "bmt-acceleration")
            bmt.text = "Infinity"
            bmt = ET.SubElement(block_data, "Boolean", key="bmt-auto-brake")
            bmt.text = "True"
            bmt = ET.SubElement(block_data, "Boolean", key="flipped")
            bmt.text = "False"
        
        if str(info['id'])=="35":
            bmt = ET.SubElement(block_data,"Single",key = "bmt-mass")
            bmt.text = "3"


        if "auto" in info:
            bmt = ET.SubElement(block_data, "Boolean", key="bmt-automatic")
            bmt.text = "True"
            bmt = ET.SubElement(block_data, "Boolean", key="bmt-auto-brake")
            bmt.text = "False"
        if "flip" in info and info["flip"]:
            bmt = ET.SubElement(block_data, "Boolean", key="flipped")
            bmt.text = "True"
        if "WheelDoubleSpeed" in info and info["WheelDoubleSpeed"]:
            bmt = ET.SubElement(block_data, "Single", key="bmt-speed")
            bmt.text = "2"

    tree = ET.ElementTree(machine)
    ET.indent(tree, space="\t", level=0)
    xml_str = ET.tostring(machine, encoding="utf-8", method="xml", xml_declaration=True).decode("utf-8")

    return xml_str

def extract_json_from_string(input_string):

    pattern = r"```json(.*?)```"
    match = re.search(pattern, input_string, re.DOTALL)

    if match:
        json_content = match.group(1).strip()
        try:
            json_dict = json.loads(json_content)
            return json_dict
        except json.JSONDecodeError as e:
            print(f"Decode json Error:{e}")
            return None
    else:
        print("json format not found")
        return None


def llm_feedback_3d(block_sizes, xml_block_details, block_details, autofit_gt=True, overlap_feedback=True):
    with open(block_sizes, 'r', encoding='utf-8') as file:
        block_sizes_content = json.load(file)
        
    gp, gr = xml_block_details[0]["GlobalPosition"], xml_block_details[0]["GlobalRotation"]
    corners_feedback_forquizzer = "block 3D info:\n"
    corners_feedback_forbuilder = "block orientation info:\n"
    corners_parent_llm, corners_parent_llm_parent = [], []
    
    for i, xml_block in enumerate(xml_block_details):
        if "GlobalPosition" in xml_block: continue
        
        block_id, order_id = xml_block["id"], xml_block["order_id"]
        if str(block_id) in LINEAR_BLOCKS:
            corners_parent_llm_parent.append([block_id, order_id, -1])
            corners_parent_llm.append(np.zeros((8,3)))
            continue

        x_transform = xml_block["Transform"]
        pos, rot, scale = x_transform["Position"], x_transform["Rotation"], x_transform["Scale"]
        # print(pos,rot,scale)
        block_info = block_sizes_content[str(block_id)]
        bbox_lp, bbox_gp = get_bbox(pos, rot, scale, block_info['bc_gc'], block_info['bbox_size'], gp, gr)
        
        corners_parent_llm.append(bbox_gp)
        corners_parent_llm_parent.append([block_id, order_id, block_details[i-1]["parent"]])
        
        facing_dir = facing(rot)
        corners_feedback_forquizzer += f"order_id:{order_id}\norientation :{facing_dir}\napproximate rectangular-parallelepiped vertex positions of the block:{bbox_gp.tolist()}\n"
        corners_feedback_forbuilder += f"order_id:{order_id}\norientation :{facing_dir}"

    # Calculate machine dimensions
    corners_arr = np.vstack([c for c in corners_parent_llm if c.size > 0])
    min_vals, max_vals = corners_arr.min(axis=0), corners_arr.max(axis=0)
    lowest_y, highest_y = min_vals[1], max_vals[1]
    left_x, right_x = min_vals[0], max_vals[0]
    back_z, forward_z = min_vals[2], max_vals[2]
    
    geo_center = np.array([(right_x + left_x)/2, (highest_y + lowest_y)/2, (forward_z + back_z)/2])
    
    if autofit_gt:
        xml_block_details[0]["GlobalPosition"][1] -= (lowest_y - 0.5)
        xml_block_details[0]["GlobalPosition"][0] -= geo_center[0]
        xml_block_details[0]["GlobalPosition"][2] -= geo_center[2]

    env_fail = (highest_y - lowest_y > 9.5) or (right_x - left_x > 17) or (forward_z - back_z > 17)
    height, wide, long = round(highest_y - lowest_y, 2), round(right_x - left_x, 2), round(forward_z - back_z, 2)

    # Validate machine structure
    machine_structure_error = ""
    if "corners" in block_details[1]:
        for i, block in enumerate(block_details):
            if "GlobalPosition" in block or str(block.get("id")) in LINEAR_BLOCKS: continue
            if not np.allclose(block["corners"], corners_parent_llm[i], atol=1e-2):
                machine_structure_error += (f"order_id {i}block vertex infor not consistent!\n"
                                         f"vertex info:{block['corners']}\n"
                                         f"Vertex information back calculated from the relative construction point info:{corners_parent_llm[i]}\n")

    overlap_infos = check_overlap(corners_parent_llm, vis=False, corners_parent_llm_parent=corners_parent_llm_parent) if overlap_feedback else "overlap check masked"

    return (corners_feedback_forquizzer, corners_feedback_forbuilder, env_fail, 
            long, wide, height, machine_structure_error, overlap_infos)

FLIP_SENSITIVE_BLOCKS = ["2","46"]
def are_quaternions_similar(q1, angle_threshold=1e-3):
    
    q2 = np.array([0, -0.7071068, 0, 0.7071068])

    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    
    relative_rotation = r1.inv() * r2
    angle = relative_rotation.magnitude()
    
    return angle < angle_threshold

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],  
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],  
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]   
    ])

def check_overlap_or_connection(cube1, cube2):
    def get_bounds(vertices):
        x_min = min(v[0] for v in vertices)
        x_max = max(v[0] for v in vertices)
        y_min = min(v[1] for v in vertices)
        y_max = max(v[1] for v in vertices)
        z_min = min(v[2] for v in vertices)
        z_max = max(v[2] for v in vertices)
        return x_min, x_max, y_min, y_max, z_min, z_max

    x1_min, x1_max, y1_min, y1_max, z1_min, z1_max = get_bounds(cube1)
    x2_min, x2_max, y2_min, y2_max, z2_min, z2_max = get_bounds(cube2)

    if x1_max <= x2_min or x2_max <= x1_min:
        return False
    if y1_max <= y2_min or y2_max <= y1_min:
        return False
    if z1_max <= z2_min or z2_max <= z1_min:
        return False

    x_overlap = x1_min < x2_max and x2_min < x1_max
    y_overlap = y1_min < y2_max and y2_min < y1_max
    z_overlap = z1_min < z2_max and z2_min < z1_max

    # print(f"x_overlap: {x_overlap}, y_overlap: {y_overlap}, z_overlap: {z_overlap}")

    return x_overlap and y_overlap and z_overlap

def find_bottom_face_points(vertices, center_point):

    min_dis = float("inf")
    target_comb = None
    for combination in combinations(vertices, 4):
        avg_point = np.mean(combination, axis=0)
        # print(avg_point)

        distance = np.linalg.norm(avg_point - center_point)
        if distance<min_dis:
            min_dis = distance
            target_comb = combination
        # if np.allclose(avg_point, center_point, atol=0.2):
            
            
        #     return np.array(combination)

    
    
    return np.array(target_comb)  

def match_vertices(initial_vertices, final_vertices, center_point):
    matched_vertices = []
    final_vertices = final_vertices.copy()  

    for initial_vertex in initial_vertices:
        
        initial_vector = initial_vertex - center_point
        best_match = None
        min_angle = float('inf')  

        for final_vertex in final_vertices:
            
            final_vector = final_vertex - center_point

            initial_vector = initial_vector.flatten()
            final_vector = final_vector.flatten()

            dot_product = np.dot(initial_vector, final_vector)
            norm_product = np.linalg.norm(initial_vector) * np.linalg.norm(final_vector)

            
            cosine_theta = np.clip(dot_product / norm_product, -1.0, 1.0)
            angle = np.arccos(cosine_theta)

            if angle < min_angle:
                min_angle = angle
                best_match = final_vertex

        matched_vertices.append(best_match)
        final_vertices = np.array([v for v in final_vertices if not np.allclose(v, best_match)])

    return np.array(matched_vertices)


def find_rotation_quaternion(initial_vertices, final_vertices, initial_center, final_center,manu_lp):


    initial_vectors = initial_vertices - initial_center
    final_vectors = final_vertices - final_center

    H = initial_vectors.T @ final_vectors
    U, _, Vt = np.linalg.svd(H)
    rotation_matrix = Vt.T @ U.T

    if np.linalg.det(rotation_matrix) < 0:
        Vt[2, :] *= -1
        rotation_matrix = Vt.T @ U.T

    rotation_quaternion = R.from_matrix(rotation_matrix).as_quat()

    return rotation_quaternion


def calculate_startingblock_quaternion(initial_vertices, rotated_vertices, center):

    
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    def quaternion_from_axis_angle(axis, angle):
        axis = normalize(axis)
        w = np.cos(angle / 2)
        x = axis[0] * np.sin(angle / 2)
        y = axis[1] * np.sin(angle / 2)
        z = axis[2] * np.sin(angle / 2)
        return np.array([w, x, y, z])

    initial_vector = initial_vertices[0] - center
    rotated_vector = rotated_vertices[0] - center

    cos_theta = np.dot(normalize(initial_vector), normalize(rotated_vector))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  

    rotation_axis = np.cross(initial_vector, rotated_vector)
    rotation_axis = normalize(rotation_axis)

    q = quaternion_from_axis_angle(rotation_axis, angle)

    return q

def add_rotations(q1, q2):
    r1 = R.from_quat(q1) 
    r2 = R.from_quat(q2)  
    r_combined = r1 * r2  
    return r_combined.as_quat()     

def compute_normal_vector(vertices, bp):
    bp = np.round(bp, decimals=3)
    
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    normal = np.zeros(3) 
    epsilon = 0.005  

    if abs(bp[0] - min_coords[0]) < epsilon:
        normal = np.array([-1, 0, 0])  
    elif abs(bp[0] - max_coords[0]) < epsilon:
        normal = np.array([1, 0, 0])   

    elif abs(bp[1] - min_coords[1]) < epsilon:
        normal = np.array([0, -1, 0])  
    elif abs(bp[1] - max_coords[1]) < epsilon:
        normal = np.array([0, 1, 0])  

    elif abs(bp[2] - min_coords[2]) < epsilon:
        normal = np.array([0, 0, -1]) 
    elif abs(bp[2] - max_coords[2]) < epsilon:
        normal = np.array([0, 0, 1])  
    else:
        raise ValueError("Point not on any faces of cuboid")
    
    return normal


def rotation_decomposition(rotation_matrix, reference_vector):

    rot = R.from_matrix(rotation_matrix)
    quat = rot.as_quat()  

    reference_vector = reference_vector / np.linalg.norm(reference_vector)
    rotated_reference_vector = rot.apply(reference_vector)
    rotated_reference_vector = rotated_reference_vector / np.linalg.norm(rotated_reference_vector)

    if np.allclose(reference_vector, rotated_reference_vector, atol=1e-6) or \
       np.allclose(reference_vector, -rotated_reference_vector, atol=1e-6):
        return np.array([0, 0, 0, 1])

    axis = reference_vector
    angle = 2 * np.arccos(np.dot(reference_vector, rotated_reference_vector))

    sin_half_angle = np.sin(angle / 2)
    quaternion_around_reference = np.array([
        axis[0] * sin_half_angle,
        axis[1] * sin_half_angle,
        axis[2] * sin_half_angle,
        np.cos(angle / 2)
    ])

    return quaternion_around_reference
        
def rotation_quaternion(v_from, v_to):
    v_from = v_from / np.linalg.norm(v_from)  
    v_to = v_to / np.linalg.norm(v_to)        

    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)
    
    if np.allclose(cross, 0) and np.allclose(dot, 1):
        return np.array([0, 0, 0, 1])  
    elif np.allclose(cross, 0) and np.allclose(dot, -1):

        if np.isclose(v_from[0], 0) and np.isclose(v_from[1], 0):
            axis = np.array([0, 1, 0])  
        else:
            axis = np.cross(v_from, np.array([0, 1, 0]))  
        axis = axis / np.linalg.norm(axis)
        angle = np.pi
    else:
        angle = np.arccos(dot)
        axis = cross / np.linalg.norm(cross)

    q = R.from_rotvec(axis * angle).as_quat()  
    return q

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w])

def quaternion_conjugate(q):
    """Compute the conjugate of a quaternion."""
    x, y, z, w = q
    return np.array([-x, -y, -z, w])

def compute_q2(q_in, q1):
    """Compute the second quaternion q2 given q_in and q1."""
    q1_conjugate = quaternion_conjugate(q1)
    q2 = quaternion_multiply(q1_conjugate, q_in)
    return q2


def get_relative_pos_list(bp_oldpos,ref_p,ref_r,scale=1,decimals=None):
    bp_newpos = []

    if ref_r.shape[0] != 3 or ref_r.shape[1] != 3:
        ref_r = R.from_quat(ref_r).as_matrix()  

    for point in bp_oldpos:
        point_lp =  ref_p+np.dot(ref_r, point*scale)
        bp_newpos.append(tuple(point_lp))
    bp_newpos = np.array(bp_newpos)

    if decimals!=None:
        bp_newpos = np.round(bp_newpos,decimals=decimals)

    return bp_newpos


def get_relative_pos(bp_oldpos,ref_p,ref_r,scale=1,decimals=None):
    bp_newpos =  ref_p+np.dot(ref_r, bp_oldpos*scale)
    if decimals!=None:
        bp_newpos = np.round(bp_newpos,decimals=decimals)
    return bp_newpos

def get_bbox(manu_lp,manu_lr,scale,bc_gc,bbox_size,gp,gr):

    if manu_lr.shape[0] != 3 or manu_lr.shape[1] != 3:
        manu_lr = R.from_quat(manu_lr).as_matrix()
    if gr.shape[0] != 3 or gr.shape[1] != 3:
        gr = R.from_quat(gr).as_matrix()

    half_bbox_size = np.array(bbox_size) / 2.0
    bbox_lp = []
    for z in [-1, 1]:
        for x in [-1, 1]:
            for y in [-1, 1]:
                point = (manu_lp+bc_gc) + (x * half_bbox_size[0], y * half_bbox_size[1], z * half_bbox_size[2])
                bc_point = point-manu_lp
                point_lp =  manu_lp+np.dot(manu_lr, bc_point*scale)
                bbox_lp.append(tuple(point_lp))
    bbox_lp = np.array(bbox_lp)

    bbox_gp = get_relative_pos_list(bbox_lp,gp,gr,decimals=2)
    return bbox_lp,bbox_gp


def get_mybuildingpoints(bc_bp,manu_lp,manu_lr,gp,gr,bc_gc,bbox_size,scale=1):

    bp_ori = np.array(bc_bp)
    bp_lp = get_relative_pos_list(bp_ori,manu_lp,manu_lr,scale=scale)
    bp_gp = get_relative_pos_list(bp_lp,gp,gr,decimals=2)
    bbox_lp,bbox_gp = get_bbox(manu_lp,manu_lr,scale,bc_gc,bbox_size,gp,gr)

    my_building_points = bp_gp.copy()
    my_building_points_buildrotation=[]
    # print(f"bp_gp:{bp_gp}")

    for i in range(len(my_building_points)):
        # print(f"bp_lp:{bp_lp[i]}")
        # print(f"bbox_lp:{bbox_lp}")
        normal_vector_l = compute_normal_vector(bbox_lp,bp_lp[i])
        rotated_initvec = np.array([0,0,1])
        building_points_rot_quat = rotation_quaternion(rotated_initvec,normal_vector_l)
        my_building_points_buildrotation.append(building_points_rot_quat) 
    my_building_points_buildrotation = np.array(my_building_points_buildrotation)

    return my_building_points,my_building_points_buildrotation

def get_3d_from_xml(block_ori_infos,info,gp,gr,log=False):
    scale = info['scale']
    

    bc_gc = block_ori_infos['bc_gc'] 
    
    bbox_size = block_ori_infos['bbox_size']  

    manu_lp = info['position']
    manu_lr = info['rotation_matrix']
    

    manu_gp = get_relative_pos(manu_lp,gp,gr)
    geo_lp = get_relative_pos(bc_gc,manu_lp,manu_lr,scale=scale)
    geo_gp = get_relative_pos(geo_lp,gp,gr)

    bbox_lp,bbox_gp = get_bbox(manu_lp,manu_lr,scale,bc_gc,bbox_size,gp,gr)

    bp_ori = np.array(block_ori_infos['bc_bp'])
    bp_lp = get_relative_pos_list(bp_ori,manu_lp,manu_lr,scale=scale)
    bp_gp = get_relative_pos_list(bp_lp,gp,gr,decimals=2)

    if str(info['id']) =="30":
        bc_gc = [0,0,0.5] 
        bbox_size = [1,1,1] 


    my_building_points,my_building_points_buildrotation = get_mybuildingpoints(block_ori_infos['bc_bp'],manu_lp,manu_lr,gp,gr,bc_gc,bbox_size,scale=scale)



    if log:
        print(f"scale:{scale}")
        print(f"bc_gc:{bc_gc}")
        print(f"bbox_size:{bbox_size}")
        print(f"manu_lp:{manu_lp}")
        print(f"manu_lr:{manu_lr}")
        print(f"manu_gp:{manu_gp}")
        print(f"geo_lp:{geo_lp}")
        print(f"bbox_lp:{bbox_lp}")
        print(f"geo_gp:{geo_gp}")
        print(f"bbox_gp:{bbox_gp}")
        print(f"bp_lp:{bp_lp}")
        print(f"bp_gp:{bp_gp}")
        print(f"my_building_points:{my_building_points}")
        print(f"my_building_points_buildrotation:{my_building_points_buildrotation}")

    return bbox_gp,manu_gp,bp_gp,bbox_lp,bp_lp,my_building_points,my_building_points_buildrotation

def get_3d_from_llm(block_sizes, input_info, gp, gr, log=False):
    info = deepcopy(input_info)
    for block in info:
        order_id = int(block["order_id"])
        block_id = str(block["id"])
        
        # Handle scale
        if "scale" not in block:
            block["scale"] = np.array([1,1,1])
        else:
            print(f"warning, block {order_id} changed scale, roll back to deafult scale")
            block["scale"] = np.array([1,1,1])

        # Handle rotations
        if "bp_lr" not in block:
            if "manu_lr" not in block:
                block["bp_lr"] = np.array([0,0,0,1])
            elif "manu_lr" in block and str(block.get("parent", "")) not in ("-1", ""):
                print(f"warning, {order_id}has costume rotation but not root block, clear it")
                block["bp_lr"] = np.array([0,0,0,1])
                block.pop("manu_lr", None)

        block_info = block_sizes[block_id]
        parent = int(block.get("parent", -1))

        # Handle parent cases
        if parent == -1:
            if block_id not in LINEAR_BLOCKS and  block_id!="0":
                print("warning, found block not root, and no parents")
            
            if block_id in LINEAR_BLOCKS:
                parent_a, parent_b = int(block["parent_a"]), int(block["parent_b"])
                bp_id_a, bp_id_b = int(block["bp_id_a"]), int(block["bp_id_b"])
                block["bp_lr"] = np.array([0,0,0,1])
                block["manu_lr"] = add_rotations(
                    info[parent_a]["my_building_points_buildrotation"][bp_id_a],
                    block["bp_lr"]
                )
                block["manu_lp_a"] = info[parent_a]["my_building_points"][bp_id_a] - gp
                block["manu_lp_b"] = info[parent_b]["my_building_points"][bp_id_b] - gp
            else:
                if "manu_lr" not in block:
                    block["manu_lr"] = np.array([0,0,0,1])
                    block["manu_lp"] = np.array([0,0,0])
                else:
                    print("warning, LLM generated block's position and rotation")
                    if block["manu_lr"].shape != (3, 3):
                        block["manu_lr"] = R.from_matrix(block["manu_lr"]).as_quat()
        else:
            try:
                bp_id = int(block["bp_id"])
                parent_rot = info[parent]["my_building_points_buildrotation"][bp_id]
                block["manu_lr"] = add_rotations(parent_rot, block["bp_lr"])
                block["manu_lp"] = info[parent]["my_building_points"][bp_id] - gp
            except Exception:
                print(f"warning, parent:{parent},order_id{order_id} my_building_points or my_building_points_buildrotation not exists")
                # print(info[parent])
                pass

        if block_id not in LINEAR_BLOCKS:
            if block_id in FLIP_SENSITIVE_BLOCKS:
                block["flip"] = are_quaternions_similar(block["manu_lr"])

            bc_bp = block_info['bc_bp']
            bc_gc = block_info['bc_gc']
            bbox_size = block_info['bbox_size']
            
            if block_id == "30":
                bc_gc = [0,0,0.5]
                bbox_size = [1,1,1]

            building_points, build_rotation = get_mybuildingpoints(
                bc_bp, block["manu_lp"], block["manu_lr"], gp, gr, 
                bc_gc, bbox_size, scale=block["scale"]
            )
            block["my_building_points"] = building_points
            block["my_building_points_buildrotation"] = build_rotation

            if log:
                print(f"block_id:{block_id}\nscale:{block['scale']}\nbc_gc:{bc_gc}\n"
                      f"bbox_size:{bbox_size}\nmanu_lp:{block['manu_lp']}\n"
                      f"manu_lr:{block['manu_lr']}\nmy_building_points:{building_points}\n"
                      f"my_building_points_buildrotation:{build_rotation}")
    
    return info