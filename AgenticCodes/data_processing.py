import json
import copy
from copy import deepcopy
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import re
import traceback
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
from collections import Counter, defaultdict
from itertools import groupby
import math
import os

from AgenticCodes.config import BLOCKPROPERTY, BLOCKINTRO, FACINGMAP, LIFECYCLE,BLOCKPROPERTYPATH,LINEAR_BLOCKS
from AgenticCodes.utils import extract_json_from_string, write_file, easy_get_bppos,match_case
from environments.besiege_tools import (
    convert_to_numpy, check_overlap, llm2xml_filetree, 
    create_xml, get_bbox, facing,llm_feedback_3d,xml2json,generate_guid,get_3d_from_llm
)

def range_clip(a, bc):
    b, c = bc
    return [b, a] if b <= a <= c else [a, a] if a < b else [b, c]

def modify_json(data, operations):
    
    def add_block(block_type_id, order_id, building_point, order_id_b=None, building_point_b=None):
        tmp_data = deepcopy(data)
        if str(block_type_id) in LINEAR_BLOCKS:
            new_block = {"id":block_type_id,"order_id":len(tmp_data),
                         "parent_a":order_id,"bp_id_a":building_point,
                         "parent_b":order_id_b,"bp_id_b":building_point_b}
        else:
            new_block = {"id":block_type_id,"order_id":len(tmp_data),
                         "parent":order_id,"bp_id":building_point}
            
        tmp_data.append(new_block)
        return tmp_data
    
    def remove_block(order_id):
        tmp_data = deepcopy(data)
        block_type_id = str(tmp_data[order_id]["id"])
        if order_id==0:
            print("can not delete startingBlock")
            pass
        elif str(block_type_id) in LINEAR_BLOCKS:
            tmp_data[order_id]["parent_a"]=-100
            tmp_data[order_id]["parent_b"]=-100
        else:
            tmp_data[order_id]["parent"]=-100
        return tmp_data
    
    def move_block(order_id, new_parent_order_id, new_building_point,
                   new_parent_order_id_b=None, new_building_point_b=None):
        tmp_data = deepcopy(data)
        if new_parent_order_id>=order_id:
            print("Error! Can not move blocks that order_id larger than self")
            return tmp_data

        type_id = tmp_data[order_id]["id"]
        if str(type_id)in LINEAR_BLOCKS:
            tmp_data[order_id]["parent_a"] = new_parent_order_id
            tmp_data[order_id]["bp_id_a"] = new_building_point
            if new_parent_order_id_b==None:
                pass
            else:
                tmp_data[order_id]["parent_b"] = new_parent_order_id_b
                tmp_data[order_id]["bp_id_b"] = new_building_point_b

        else:
            tmp_data[order_id]["parent"] = new_parent_order_id
            tmp_data[order_id]["bp_id"] = new_building_point
            
        return tmp_data

    def log_error(operation,reason):
        error_dict = {
            "operation":"",
            "reason":"",
            
        }
        error_dict["operation"] = operation
        error_dict["reason"]=reason
        return error_dict

    def process_valid_check(data,error_reason,old_parent_to_new_parent,processing_block_id=None):
        
        if "size error" in error_reason:
            threeD_infos = get_machine_3D_infos(data)
            old_size = threeD_infos["size"]
            error_reason_list = error_reason.split(";")
            error_reason_size=None
            for er in error_reason_list:
                if "size error" in er:
                    error_reason_size = er
                    break
            size_list = error_reason_size.replace("size error:","").split(",")
            valid_size = [17,17,9.5]
            valid_size_index = ["long","width","height"]
            machine_size_head="machine size("
            oversize_head = ")oversize, before modification:"
            after_modify_head = ",after modification:"
            for i,size in enumerate(size_list):
                if float(size)>= valid_size[i]:
                    error_log = f"{machine_size_head}{valid_size_index[i]}{oversize_head}{old_size[i]}{after_modify_head}{size}"
                    return error_log

        elif "overlap error" in error_reason:
            from collections import defaultdict
            new_parent_to_old_parent={}
            for k,v in old_parent_to_new_parent.items():
                new_parent_to_old_parent[v]=k

            overlaps_list = []
            # print(error_reason)
            error_reason_list = error_reason.split(";")
            error_reason_overlap=None
            for er in error_reason_list:
                if "overlap error" in er:
                    error_reason_overlap = er
                    break
            overlap_infos = error_reason_overlap.replace("overlap error:","").split('\n')
            overlap_infos = [item for item in overlap_infos if item]
            overlap_infos.pop(-1)
            # print(overlap_infos)
            for line in overlap_infos:
                if "Block order_id" in line and "overlap" in line:
                    parts = line.split()
                    # print(parts)
                    # "Block order_id id1 and Block order_id id2 overlap\n"
                    id1 = int(parts[2])  
                    id2 = int(parts[6])   
                    overlaps_list.append([id1, id2])
            unique_tuples = {tuple(sorted(pair)) for pair in overlaps_list}
            overlaps_list = [list(pair) for pair in unique_tuples]
            id_counts = defaultdict(int)
            # print(error_reason)
            for overlap in overlaps_list:
                a = overlap[0]
                b = overlap[1]
                if a in new_parent_to_old_parent:
                    old_a = new_parent_to_old_parent[a]
                else:
                    old_a = a
                if b in new_parent_to_old_parent:
                    old_b = new_parent_to_old_parent[b]
                else:
                    old_b = b
                overlap[0] = old_a
                overlap[1] = old_b
                id_counts[old_a] += 1
                id_counts[old_b] += 1
            
            error_log=""
            new_add_block_head = "order_id of new added block:"
            overlap_after_op_head ="overlaps after operation:"
            if processing_block_id!=None:
                error_log+=f"{new_add_block_head}{processing_block_id}"
            error_log += f"{overlap_after_op_head}{overlaps_list}"
            return error_log
            


    order_ids = len(data)
    used_bps = {}
    for block_info in data:
        if "parent" in block_info and block_info["parent"]!=-1:
            parent = block_info["parent"]
            used_bp = block_info["bp_id"]
            if parent not in used_bps:
                used_bps[parent] = [used_bp]
            else:
                used_bps[parent].append(used_bp)
        else:
            pass
    #print(used_bps)
    
    removed_blocks = []
    operation_index = -1
    for operation in operations:
        # print(operation)
        operation_index+=1
        if "[" in operation:
            op_list = match_case(operation,"[","]")
        else:
            op_list = re.findall(r'\d+', operation)
        # print(operation)
        if "Move" in operation:
            if len(op_list)<3:
                a_head = "wrong moving format"
                return log_error(operation,f"{a_head}")
            if len(op_list)==3:
                try:
                    moving_block = int(op_list[0])
                    new_parent = int(op_list[1]) 
                    new_buildingpoint = int(op_list[2])
                except:
                    return log_error(operation,f"wrong operation type")

                if new_parent<order_ids:
                    new_parent_type = data[new_parent]["id"]
                    new_parent_bps = len(BLOCKPROPERTY[str(new_parent_type)]["bc_bp"])
                    if new_parent in  used_bps:
                        used_bp_list = used_bps[new_parent]
                        avaliable_new_parent_bps = list(range(new_parent_bps))
                        avaliable_new_parent_bps = list(filter(lambda x: x not in used_bp_list, avaliable_new_parent_bps))
                    else:
                        avaliable_new_parent_bps = list(range(new_parent_bps))
                if moving_block>=order_ids:
                    a_head = "can not move orderid"
                    b_head = "block, this block does not exist"
                    return log_error(operation,f"{a_head} {moving_block} {b_head}")

                

                if new_parent<order_ids:
                    if "parent" in data[moving_block]:
                        old_parent = data[moving_block]["parent"]
                        old_bp = data[moving_block]["bp_id"]
                    else:
                        old_parent = data[moving_block]["parent_a"]
                        old_bp = data[moving_block]["bp_id_a"]

                if moving_block in removed_blocks:
                    a_head = "can not Move removed block"
                    return log_error(operation,f"{a_head} {moving_block}")
                elif new_parent in removed_blocks:
                    a_head = "can not move on order_id"
                    b_head = "block, this block has been removed"
                    return log_error(operation,f"{a_head} {new_parent} {b_head}")
                elif new_parent>=order_ids:
                    a_head = "can not move on order_id"
                    b_head = "block, this block not exist"
                    return log_error(operation,f"{a_head} {new_parent} {b_head}")
                elif new_buildingpoint>=new_parent_bps:
                    a_head = "new_parent, order_id"
                    b_head = "block, not exist Constructible Point"
                    return log_error(operation,f"{a_head} {new_parent} {b_head} {new_buildingpoint}")
                elif new_parent in used_bps and new_buildingpoint in used_bps[new_parent]:
                    a_head = "new_parent, order_id"
                    b_head = "block, Constructible Points"
                    c_head = "is occupied"
                    return log_error(operation,f"{a_head} {new_parent} {b_head} {new_buildingpoint} {c_head}")
                elif new_parent>=moving_block:
                    a_head="can not move to blocks whose order_id larger than target block("
                    return log_error(operation,f"{a_head} {new_parent}>={moving_block})")
                elif old_parent==new_parent and old_bp==new_buildingpoint:
                    a_head = "not effective, block still in parent, order_id"
                    b_head = "Constructible point"
                    c_head = "the avaliable Constructible points on parent:"
                    return log_error(operation,f"{a_head} {old_parent} {b_head} {old_bp} {c_head} {avaliable_new_parent_bps}")
                elif moving_block==0:
                    a_head="can not move starting block"
                    return log_error(operation,f"{a_head} {moving_block}")

                tmp_data = move_block(moving_block,new_parent,new_buildingpoint)
                
                tmp_data_forcheck,old_parent_to_new_parent = complete_modify(tmp_data)
                new_threeDinfo_list = get_3Dinfos_from_json(tmp_data_forcheck)

                
                tmp_old_data,old_mapping = complete_modify(data)
                old_threeDinfo_list = get_3Dinfos_from_json(tmp_old_data)
                
                if "parent" in data[moving_block]:
                    old_facing = old_threeDinfo_list[old_mapping[moving_block]]["abs"]["orient"]
                    new_facing = new_threeDinfo_list[old_parent_to_new_parent[moving_block]]["abs"]["orient"]
                    valid_results,error_reason = valid_check(tmp_data_forcheck,return_error_reason=True)
                else:
                    valid_results=True

                if not valid_results:
                    a_head = "parent before moving:"
                    b_head = ",and facing:"
                    c_head = "parent after moving:"
                    log_error_reason = f"{a_head} {old_parent} {b_head} {old_facing}\n"
                    log_error_reason += f"{c_head} {new_parent} {b_head} {new_facing}\n"
                    try:
                        log_error_reason += process_valid_check(tmp_old_data,error_reason,old_parent_to_new_parent)
                    except:
                        pass
                    return log_error(operation,log_error_reason)


                
                if "parent" in data[moving_block]:
                    used_bp = data[moving_block]["bp_id"]
                    used_bps[old_parent].remove(used_bp)
                    if new_parent in used_bps:
                        used_bps[new_parent].append(new_buildingpoint)
                    else:
                        used_bps[new_parent]=[new_buildingpoint]
                
                data = tmp_data
            elif len(op_list)==5:
                moving_block = int(op_list[0])
                new_parent_a = int(op_list[1]) 
                new_buildingpoint_a = int(op_list[2])
                new_parent_b = int(op_list[3]) 
                new_buildingpoint_b = int(op_list[4])

                if new_parent_a<order_ids:
                    new_parent_type_a = data[new_parent_a]["id"]
                    new_parent_bps_a = len(BLOCKPROPERTY[str(new_parent_type_a)]["bc_bp"])
                    if new_parent_a in  used_bps:
                        used_bp_list = used_bps[new_parent_a]
                        avaliable_new_parent_a_bps = list(range(new_parent_bps_a))
                        avaliable_new_parent_a_bps = list(filter(lambda x: x not in used_bp_list, avaliable_new_parent_a_bps))
                    else:
                        avaliable_new_parent_a_bps = list(range(new_parent_bps_a))
                if new_parent_b<order_ids:
                    new_parent_type_b = data[new_parent_b]["id"]
                    new_parent_bps_b = len(BLOCKPROPERTY[str(new_parent_type_b)]["bc_bp"])
                    if new_parent_b in  used_bps:
                        used_bp_list = used_bps[new_parent_b]
                        avaliable_new_parent_b_bps = list(range(new_parent_bps_b))
                        avaliable_new_parent_b_bps = list(filter(lambda x: x not in used_bp_list, avaliable_new_parent_b_bps))
                    else:
                        avaliable_new_parent_b_bps = list(range(new_parent_bps_b))

                if moving_block>=order_ids:
                    a_head = "can not move orderid"
                    b_head = "block, this block does not exist"
                    return log_error(operation,f"{a_head} {moving_block} {b_head}")

                if new_parent_a<order_ids and new_parent_b<order_ids:
                    if "parent" in data[moving_block]:
                        a_head = "wrong moving format"
                        return log_error(operation,f"{a_head}")
                    else:
                        old_parent_a = data[moving_block]["parent_a"]
                        old_bp_a = data[moving_block]["bp_id_a"]
                        old_parent_b = data[moving_block]["parent_b"]
                        old_bp_b = data[moving_block]["bp_id_b"]

                if moving_block in removed_blocks:
                    a_head = "can not Move removed block"
                    return log_error(operation,f"{a_head} {moving_block}")
                elif new_parent_a in removed_blocks:
                    a_head = "can not move on order_id"
                    b_head = "block, this block has been removed"
                    return log_error(operation,f"{a_head} {new_parent_a} {b_head}")
                elif new_parent_b in removed_blocks:
                    a_head = "can not move on order_id"
                    b_head = "block, this block has been removed"
                    return log_error(operation,f"{a_head} {new_parent_b} {b_head}")
                elif new_parent_a>=order_ids:
                    a_head = "can not move on order_id"
                    b_head = "block, this block not exist"
                    return log_error(operation,f"{a_head} {new_parent_a} {b_head}")
                elif new_parent_b>=order_ids:
                    a_head = "can not move on order_id"
                    b_head = "block, this block not exist"
                    return log_error(operation,f"{a_head} {new_parent_b} {b_head}")
                elif new_buildingpoint_a>=new_parent_bps_a:
                    a_head = "new_parent, order_id"
                    b_head = "block, not exist Constructible Point"
                    return log_error(operation,f"{a_head} {new_parent_a} {b_head} {new_buildingpoint_a}")
                elif new_buildingpoint_b>=new_parent_bps_b:
                    a_head = "new_parent, order_id"
                    b_head = "block, not exist Constructible Point"
                    return log_error(operation,f"{a_head} {new_parent_b} {b_head} {new_buildingpoint_b}")
                elif new_parent_a>=moving_block:
                    a_head="can not move to blocks whose order_id larger than target block("
                    return log_error(operation,f"{a_head} {new_parent_a}>={moving_block})")
                elif new_parent_b>=moving_block:
                    a_head="can not move to blocks whose order_id larger than target block("
                    return log_error(operation,f"{a_head} {new_parent_b}>={moving_block})")
                elif (old_parent_a==new_parent_a and old_bp_a==new_buildingpoint_a) and \
                    (old_parent_b==new_parent_b and old_bp_b==new_buildingpoint_b):
                    a_head = "not effective, block still in parent, order_id"
                    b_head = "Constructible point"
                    c_head = "the avaliable Constructible points on parent:"
                    return log_error(operation,f"{a_head} {old_parent_a} {b_head} {old_bp_a} {c_head} {avaliable_new_parent_a_bps}")
                elif moving_block==0:
                    a_head="can not move starting block"
                    return log_error(operation,f"{a_head} {moving_block}")

                tmp_data = move_block(moving_block,new_parent_a,new_buildingpoint_a,
                                      new_parent_b,new_buildingpoint_b)
                
                tmp_data_forcheck,old_parent_to_new_parent = complete_modify(tmp_data)
                new_threeDinfo_list = get_3Dinfos_from_json(tmp_data_forcheck)

                
                tmp_old_data,old_mapping = complete_modify(data)
                old_threeDinfo_list = get_3Dinfos_from_json(tmp_old_data)
                
                data = tmp_data

        elif "Remove" in operation:
            if len(op_list)!=1:
                print(f"wrong remove format{operation}")
                a_head = "wrong remove format"
                return log_error(operation,f"{a_head}")
            removing_block = int(op_list[0])
            if removing_block in removed_blocks:
                a_head = "can not Remove Blocks that have already been removed"
                return log_error(operation,f"order_id {a_head} {removing_block}")
            elif removing_block>=order_ids:
                a_head = "block does not exists"
                return log_error(operation,f"order_id {removing_block} {a_head}")
            elif removing_block==0:
                a_head = "can not delete starting block"
                return log_error(operation,f"{a_head} (order_id 0)")
            elif removing_block in used_bps and used_bps[removing_block]!=[]:
                a_head = "block has child blocks"
                return log_error(operation,f"order_id {removing_block} {a_head}")
            
            prev_ops = operations[:operation_index]
            
            is_linear_parent = False
            for prev_op in prev_ops:
                if "[" in prev_op:
                    prev_op_list = match_case(prev_op,"[","]")
                else:
                    prev_op_list = re.findall(r'\d+', prev_op)
                if ("Add" in prev_op or "Move" in prev_op):
                    if len(prev_op_list) == 3:
                        if int(prev_op_list[1]) == removing_block:
                            is_linear_parent = True
                    elif len(prev_op_list) == 5:
                        if int(prev_op_list[1]) == removing_block or int(prev_op_list[3]) == removing_block:
                            is_linear_parent = True
            for block in data:
                if "parent_a" in block or "parent_b" in block:
                    # print(block)
                    if int(block["parent_a"])== removing_block or int(block["parent_b"])== removing_block:
                        is_linear_parent = True
            if is_linear_parent:
                a_head = "block has child blocks"
                return log_error(operation,f"order_id {removing_block} {a_head}")

            if "parent" in data[removing_block]:
                parent = data[removing_block]["parent"]
                used_bp = data[removing_block]["bp_id"]
                used_bps[parent].remove(used_bp)

            data = remove_block(removing_block)

            removed_blocks.append(removing_block)


        elif "Add" in operation:
            
            if len(op_list)==3:
                new_block_type = str(op_list[0])
                new_parent = int(op_list[1])
                new_buildingpoint = int(op_list[2])

                if new_parent<order_ids:
                    new_parent_type = data[new_parent]["id"]
                    new_parent_bps = len(BLOCKPROPERTY[str(new_parent_type)]["bc_bp"])
                    if new_parent in  used_bps:
                        used_bp_list = used_bps[new_parent]
                        avaliable_new_parent_bps = list(range(new_parent_bps))
                        avaliable_new_parent_bps = list(filter(lambda x: x not in used_bp_list, avaliable_new_parent_bps))
                    else:
                        avaliable_new_parent_bps = list(range(new_parent_bps))
                        
                if new_parent in removed_blocks:
                    a_head = "can not build on order_id"
                    b_head = "block, this block has been removed"
                    return log_error(operation,f"{a_head} {new_parent} {b_head}")
                elif new_block_type not in BLOCKPROPERTY.keys():
                    a_head = "non-existent block type"
                    return log_error(operation,f"{a_head} {new_block_type}")
                elif new_parent>=order_ids:
                    a_head = "can not build on"
                    b_head = "block, this block not exist"
                    return log_error(operation,f"{a_head} order_id {new_parent} {b_head}")
                elif new_buildingpoint>=new_parent_bps:
                    a_head = "does not exist Constructible point"
                    return log_error(operation,f"new_parent {new_parent} {a_head} {new_buildingpoint}")
                elif new_parent in used_bps and new_buildingpoint in used_bps[new_parent]:
                    a_head="cpt"
                    b_head="is occupied"
                    return log_error(operation,f"new_parent {new_parent} {a_head} {new_buildingpoint} {b_head}")
                elif new_block_type=="0":
                    a_head = "can not add starting block"
                    return log_error(operation,f"{a_head}")
                elif str(new_block_type) in LINEAR_BLOCKS:
                    a_head = "Add linear bolck format wrong"
                    return log_error(operation,f"{a_head}")

                tmp_data = add_block(new_block_type,new_parent,new_buildingpoint)

                new_block_id = len(data)

                
                tmp_data_forcheck,old_parent_to_new_parent = complete_modify(tmp_data)
                new_threeDinfo_list = get_3Dinfos_from_json(tmp_data_forcheck)
                
                tmp_old_data,old_mapping = complete_modify(data)
                old_threeDinfo_list = get_3Dinfos_from_json(tmp_old_data)
                try:
                    new_facing = new_threeDinfo_list[-1]["abs"]["orient"]
                except:
                    print("new_facing error")
                    print(new_threeDinfo_list[-1]["abs"])
                    new_facing= "unknown"
                valid_results,error_reason = valid_check(tmp_data_forcheck,return_error_reason=True)
                if not valid_results:
                    a_head="new added block"
                    b_head = ",and facing:"
                    log_error_reason = f"{a_head} parent:{new_parent} {b_head} {new_facing} \n"
                    log_error_reason += str(process_valid_check(tmp_old_data,error_reason,old_parent_to_new_parent,processing_block_id=new_block_id))
                    return log_error(operation,log_error_reason)
                
                
                if new_parent in used_bps:
                    used_bps[new_parent].append(new_buildingpoint)
                else:
                    used_bps[new_parent]=[new_buildingpoint]
                
                data = tmp_data

            elif len(op_list)==5:
                new_block_type = str(op_list[0])
                new_parent_a = int(op_list[1])
                new_buildingpoint_a = int(op_list[2])
                new_parent_b = int(op_list[3])
                new_buildingpoint_b = int(op_list[4])
                # print(new_parent_a)

                if new_parent_a>=order_ids:
                    a_head = "can not build on"
                    b_head = "block, this block not exist"
                    return log_error(operation,f"{a_head} order_id {new_parent_a} {b_head}")
                if new_parent_b>=order_ids:
                    a_head = "can not build on"
                    b_head = "block, this block not exist"
                    return log_error(operation,f"{a_head} order_id {new_parent_b} {b_head}")
                
                if new_parent_a in removed_blocks:
                    a_head = "can not build on order_id"
                    b_head = "block, this block has been removed"
                    return log_error(operation,f"{a_head} {new_parent_a} {b_head}")

                if new_parent_b in removed_blocks:
                    a_head = "can not build on order_id"
                    b_head = "block, this block has been removed"
                    return log_error(operation,f"{a_head} {new_parent_b} {b_head}")

                new_parent_type_a = data[new_parent_a]["id"]
                new_parent_bps_a = len(BLOCKPROPERTY[str(new_parent_type_a)]["bc_bp"])

                new_parent_type_b = data[new_parent_b]["id"]
                new_parent_bps_b = len(BLOCKPROPERTY[str(new_parent_type_b)]["bc_bp"])

                if new_buildingpoint_a>=new_parent_bps_a:
                    a_head = "does not exist Constructible point"
                    return log_error(operation,f"new_parent_a {new_parent_a} {a_head} {new_buildingpoint_a}")
                elif new_buildingpoint_b>=new_parent_bps_b:
                    a_head = "does not exist Constructible point"
                    return log_error(operation,f"new_parent_b {new_parent_b} {a_head} {new_buildingpoint_b}")

                data = add_block(new_block_type,new_parent_a,new_buildingpoint_a,
                          new_parent_b,new_buildingpoint_b)
            else:
                print("Wrong ADD format")
        
    return complete_modify(data)

def complete_modify(data):
    new_blocks=[]
    old_orderid_to_new_orderid={-1:-1,0:0}
    for block in data:
        if "parent" in block and block["parent"]!=-100:
            old_order_id = block["order_id"]
            new_order_id = len(new_blocks)
            old_parent = block["parent"]
            new_block = copy.deepcopy(block)
            new_block["order_id"] = new_order_id
            new_block["parent"] = old_parent
            new_blocks.append(new_block)
            old_orderid_to_new_orderid[old_order_id] = new_order_id
        elif "parent_a" in block and  block["parent_a"]!=-100:
            old_order_id = block["order_id"]
            new_order_id = len(new_blocks)
            old_parent_a = block["parent_a"]
            old_parent_b = block["parent_b"]
            new_block = copy.deepcopy(block)
            new_block["order_id"] = new_order_id
            new_block["parent_a"] = old_parent_a
            new_block["parent_b"] = old_parent_b
            new_blocks.append(new_block)
            old_orderid_to_new_orderid[old_order_id] = new_order_id
    for block in new_blocks:
        if "parent" in block:
            old_parent = block["parent"]

            block["parent"] = old_orderid_to_new_orderid[old_parent]
        elif "parent_a" in block:
            old_parent_a = block["parent_a"]
            old_parent_b = block["parent_b"]
            try:
                block["parent_a"] = old_orderid_to_new_orderid[old_parent_a]
                block["parent_b"] = old_orderid_to_new_orderid[old_parent_b]
            except:
                print(data)
                quit()
    return new_blocks,old_orderid_to_new_orderid

def generate_modify_history(modify_step_list, modify_return):
    modify_history = ""
    error_happened = False
    total_ops = 0
    success_ops = 0

    state_tag = ["Error","Success","Unverified"]

    for modify_step in modify_step_list:
        total_ops+=1
        
        if modify_step==modify_return["operation"]:
            reason = modify_return["reason"]
            for tag in state_tag:
                modify_step = modify_step.replace(tag, "")
            modify_history+=f"{modify_step} Error:{reason} \n"
            error_happened=True
        elif not error_happened:
            for tag in state_tag:
                modify_step = modify_step.replace(tag, "")
            modify_history+=f"{modify_step} Success \n"
            success_ops+=1
        else:
            for tag in state_tag:
                modify_step = modify_step.replace(tag, "")
            modify_history+=f"{modify_step} Unverified \n"
    
    return modify_history, success_ops/total_ops

def abl_3djson_to_treejson(output,save_path=None):
    content = output if isinstance(output, list) else extract_json_from_string(output)
    
    machine = ET.Element("Machine", version="1", bsgVersion="1.3", name="gpt")
    global_elem = ET.SubElement(machine, "Global")
    gp = content[0]["GP"]
    gr = content[0]["GR"]
    position = ET.SubElement(global_elem, "Position", x=str(gp[0]), y=str(gp[1]), z=str(gp[2]))
    rotation = ET.SubElement(global_elem, "Rotation", x=str(gr[0]), y=str(gr[1]), z=str(gr[2]), w=str(gr[3]))
    
    data_elem = ET.SubElement(machine, "Data")
    string_array = ET.SubElement(data_elem, "StringArray", key="requiredMods")
    blocks_elem = ET.SubElement(machine, "Blocks")
    content.pop(0)
    
    for info in content:
        
        block_id = info['id']
        
        if info['id']=='18_1':
            block_id ='18'
        
        block = ET.SubElement(blocks_elem, "Block", id=str(block_id), guid=generate_guid())
        transform = ET.SubElement(block, "Transform")
        info_p = info['Position']
        position = ET.SubElement(transform, "Position", x=str(info_p[0]), y=str(info_p[1]), z=str(info_p[2]))
        info_r = info['Rotation']
        rotation = ET.SubElement(transform, "Rotation", x=str(info_r[0]), y=str(info_r[1]), z=str(info_r[2]), w=str(info_r[3]))
        info_s = [1,1,1]
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
            end_rotation = ET.SubElement(block_data,"Vector3",key = "end-rotation")
            ET.SubElement(end_rotation, "X").text = str(0)
            ET.SubElement(end_rotation, "Y").text = str(0)
            ET.SubElement(end_rotation, "Z").text = str(0)
        
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
            q = np.array(info['Rotation'])              
            ref = R.from_quat([0, -0.7071068, 0, 0.7071068])

            equal = (ref == R.from_quat(q) )
            bmt = ET.SubElement(block_data, "Boolean", key="flipped")
            if equal:
                bmt.text = "True"
            else:
                bmt.text = "False"  
            

    
    tree = ET.ElementTree(machine)
    ET.indent(tree, space="\t", level=0)
    xml_str = ET.tostring(machine, encoding="utf-8", method="xml", xml_declaration=True).decode("utf-8")

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)

    try:
        return f"```json{json.dumps(xml2json(xml_str,BLOCKPROPERTYPATH))}```"
    except:
        return f"```json```"


def valid_check(output, return_error_reason=False):
    try:
        content = output if isinstance(output, list) else extract_json_from_string(output)
        if not content:
            raise ValueError("machine is null")
        block_details = convert_to_numpy(deepcopy(content))

        xml_block_details, processed_details, illegal_feedback = llm2xml_filetree(
            block_details,
            BLOCKPROPERTYPATH,
            selected_menu=None
        )
        if illegal_feedback:
            
            error_reason = f"illegal building:{illegal_feedback}"
            
            return (False, error_reason) if return_error_reason else False

        feedback = llm_feedback_3d(
            block_sizes=BLOCKPROPERTYPATH,
            xml_block_details=xml_block_details,
            block_details=processed_details
        )
        env_fail, long, wide, height, overlap_infos = feedback[2], feedback[3], feedback[4], feedback[5], feedback[7]

        
        error_reasons = []
        if env_fail:
            error_reasons.append(f"size error:{long},{wide},{height}")
        if overlap_infos != "no error":
            error_reasons.append(f"overlap error:{overlap_infos}")

        if error_reasons:
            
            full_error_reason = "; ".join(error_reasons)
            
            return (False, full_error_reason) if return_error_reason else False

    except Exception as e:
        
        if return_error_reason:
            return "Reject", "illegal attachable face"
        else:
            return "Reject"

    
    return (True, "无") if return_error_reason else True

def json_to_xml(input_obj,save_path):
    if isinstance(input_obj,str):
        content = extract_json_from_string(input_obj)
    elif isinstance(input_obj,list):
        content = input_obj
    else:
        raise TypeError('Please make sure input type')
    
    block_details = content
    block_details = convert_to_numpy(block_details)
    xml_block_details,block_details,_ = llm2xml_filetree(block_details,
                                                        BLOCKPROPERTYPATH,
                                                        selected_menu=None)
    _,_,_,_,_,_,_,_ = llm_feedback_3d(block_sizes=BLOCKPROPERTYPATH,
                                    xml_block_details=xml_block_details,
                                    block_details = block_details)
    xml_string = create_xml(xml_block_details)
    write_file(save_path,xml_string)
        
    
    return xml_string



def get_bpfacing(type_id,bp_id):
    for block_intro in BLOCKINTRO:
        if int(block_intro["tid"])==int(type_id):
            return block_intro["construable points properties"][int(bp_id)]["relative orientation"]

def update_child_infos(threeDinfo_list,parent_order_id,child_order_id,bp,facing):
    for threeDinfo in threeDinfo_list:
        if int(parent_order_id) == threeDinfo["id"]:
            new_child_info = {"id":child_order_id,"cpt":bp,"relOrient":facing}
            threeDinfo["rel"]["children"].append(new_child_info)
    return threeDinfo_list

def get_3Dinfos_from_json(formatted_json):
    content = deepcopy(formatted_json) if isinstance(formatted_json, list) else extract_json_from_string(formatted_json)

    block_details = convert_to_numpy(content)
    block_details.pop(0)
    gp, gr = np.array([0, 0, 0]), np.array([0, 0, 0, 1])
    block_details = get_3d_from_llm(BLOCKPROPERTY, block_details, gp, gr, log=False)
    
    threeDinfo_list = []

    for block in block_details:
        type_id = str(block["id"])
        threeDinfo = {
            "n": BLOCKPROPERTY[str(type_id)]["name"],
            "tid": int(block["id"]),
            "id": int(block["order_id"]),
            "rel": {"parent": {}, "children": []},
            "abs": {}
        }

        
        
        if type_id in LINEAR_BLOCKS:
            # Handle special blocks (type 7 and 9)
            parent_info = threeDinfo["rel"]["parent"]
            parent_info.update({
                "a OrderID": block["parent_a"],
                "a Constructible Point ID": block["bp_id_a"],
                "b OrderID": block["parent_b"],
                "b Constructible Point ID": block["bp_id_b"]
            })
            
            threeDinfo["abs"].update({
                "a Constructible points coordinate": np.around(block["manu_lp_a"], 1).tolist(),
                "b Constructible points coordinate": np.around(block["manu_lp_b"], 1).tolist()
            })

            if type_id == "9":
                threeDinfo["Special Attributes"] = {}
                a_pos = threeDinfo["abs"]["a Constructible points coordinate"]
                b_pos = threeDinfo["abs"]["b Constructible points coordinate"]
                center_pos = np.mean([b_pos, a_pos], axis=0)
                threeDinfo["Special Attributes"]["Pull Force Direction"] = [
                    np.around(center_pos - a_pos, 1).tolist(),
                    np.around(center_pos - b_pos, 1).tolist()
                ]

        elif type_id != "0":
            # Handle normal blocks
            parent_info = threeDinfo["rel"]["parent"]
            parent_type_id = block_details[block["parent"]]["id"]
            parent_facing = get_bpfacing(parent_type_id, block["bp_id"])
            
            parent_info.update({
                "id": block["parent"],
                "bpid": block["bp_id"],
                "relOrient": parent_facing
            })
            
            threeDinfo_list = update_child_infos(
                threeDinfo_list=threeDinfo_list,
                parent_order_id=block["parent"],
                child_order_id=block["order_id"],
                bp=block["bp_id"],
                facing=parent_facing
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="Gimbal lock detected")
                euler_angles = R.from_quat(block["manu_lr"]).as_euler('xyz', degrees=True)
            
            threeDinfo["abs"].update({
                "buildPt": np.around(block["manu_lp"], 1).tolist(),
                "rot": np.round(euler_angles).astype(int).tolist()
            })

            parent_real_facing = threeDinfo_list[block["parent"]]["abs"]["orient"]
            try:
                threeDinfo["abs"]["orient"] = FACINGMAP[parent_real_facing][parent_facing]
            except KeyError:
                print("absolute information")
                print("abs")
                print("orientation")
                print("orient")
                print("parent_real_facing")
                print(parent_real_facing)
                print("parent_facing")
                print(parent_facing)
                print("parent_type_id")
                print(parent_type_id)
                print("bp_id")
                print(block["bp_id"])
                

            _, bbox_gp = get_bbox(
                block["manu_lp"], block["manu_lr"], 1,
                BLOCKPROPERTY[type_id]['bc_gc'],
                BLOCKPROPERTY[type_id]['bbox_size'],
                gp, gr
            )
            threeDinfo["abs"]["center"] = np.around(np.mean(bbox_gp, axis=0), 1).tolist()

            if type_id in ("2", "46"):
                threeDinfo["Special Attributes"] = {}
                facing = threeDinfo["abs"]["orient"]
                if facing == "z+":
                    threeDinfo["Special Attributes"]["Power Direction for the Machine"] = "x-"
                elif facing == "z-":
                    threeDinfo["Special Attributes"]["Power Direction for the Machine"] = "x+"
                elif facing == "x-":
                    threeDinfo["Special Attributes"]["Power Direction for the Machine"] = "z+"
                elif facing == "x+":
                    threeDinfo["Special Attributes"]["Power Direction for the Machine"] = "z+"
        else:
            # Handle root block (type 0)
            threeDinfo["abs"].update({
                "center": [0, 0, 0],
                "buildPt": [0, 0, 0],
                "orient": "z+",
                "rot": [0, 0, 0]
            })
        
        threeDinfo_list.append(threeDinfo)
    
    # Adjust positions to ground level
    min_y = min(
        min(block["abs"]["a Constructible points coordinate"][1], block["abs"]["b Constructible points coordinate"][1])
        if str(block["tid"]) in LINEAR_BLOCKS else block["abs"]["center"][1]
        for block in threeDinfo_list
    )
    
    for threeDinfo in threeDinfo_list:
        type_id = str(threeDinfo["tid"])
        abs_info = threeDinfo["abs"]
        
        if type_id in LINEAR_BLOCKS:
            for key in ["a Constructible points coordinate", "b Constructible points coordinate"]:
                abs_info[key][1] = round(abs_info[key][1] - min_y, 1)
        else:
            for key in ["center", "buildPt"]:
                abs_info[key][1] = round(abs_info[key][1] - min_y, 1)
            
            threeDinfo["onGround"] = abs_info["center"][1] < 0.5
            
            if type_id in ("2", "46") and not threeDinfo["onGround"]:
                threeDinfo["Special Attributes"] = {}

    return threeDinfo_list

def get_machine_3D_infos(output):
    if isinstance(output,list):
            content = output
    else:
        content = extract_json_from_string(output)
    block_details = deepcopy(content)
    block_details = convert_to_numpy(block_details)
    xml_block_details,block_details,blocks_to_delete_feedback = llm2xml_filetree(block_details,
                                                                                 BLOCKPROPERTYPATH,
                                                                                 selected_menu=None)
    _,_,env_fail,long,wide,height,_,overlap_infos = llm_feedback_3d(block_sizes=BLOCKPROPERTYPATH,
                                                                    xml_block_details=xml_block_details,
                                                                    block_details = block_details)
    threeD_infos = {
        "size" : [long,wide,height],
    }
    return threeD_infos

def check_machine_stat(bsg_machine,game_state_dict, sim_time):

    machine_quats = game_state_dict[0]["quat"][sim_time[0]:sim_time[1]]
    machine_rots = machine_quats.copy()
    machine_facings = [
        get_roughly_facing(R.from_quat(quat).apply([0, 0, 1]))
        for quat in machine_quats
    ]
    sim_time_sec = round(sim_time[1]*LIFECYCLE,2)

    machine_stat_list = [
        {
            "orderid": i,
            "name": obj["name"],
            "state": "damaged" if round(obj["alive"],2) < sim_time_sec else "detached",
            "happened_time": round(obj["alive"],2) if round(obj["alive"],2) < sim_time_sec  else obj["in_machine"]
        }
        for i, obj in enumerate(game_state_dict)
        if "GOAL_" not in obj["name"] and (round(obj["alive"],2) < sim_time_sec or obj["in_machine"] < sim_time_sec)
    ]

    
    machine_lowest_y = []
    for i in range(sim_time[0], sim_time[1]):
        min_y = float('inf')
        for obj in game_state_dict:
            if "GOAL_" in obj["name"]:
                continue
            if i < len(obj["pos"]):
                current_y = obj["pos"][i][1]  
                if current_y < min_y:
                    min_y = current_y
        machine_lowest_y.append(min_y)

    
    machine_power_list = []
    for i, obj in enumerate(game_state_dict):
        if "Wheel" not in obj["name"]:
            continue
            
        init_rot = bsg_machine[i]["rotation"]
        initial_facing = R.from_quat(init_rot).apply([0, 0, 1])
        initial_wheelpower = R.from_quat(init_rot).apply(
            [1, 0, 0] if np.allclose(initial_facing, [-1, 0, 0], atol=0.1)
            else [-1, 0, 0]
        )

        wheel_power = [np.around(R.from_quat(rot).apply(initial_wheelpower), 1) for rot in machine_rots]
        wheel_facing = [get_roughly_facing(R.from_quat(rot).apply(initial_facing)) for rot in machine_rots]

        wheel_have_ground_power = [
            False if (facing in ("y+", "y-") or 
                    j*LIFECYCLE < obj["alive"] or 
                    j*LIFECYCLE < obj["in_machine"])
            else (pos[1] - machine_lowest_y[min(j,len(machine_lowest_y)-1)] < 0.5)
            for j, (facing, pos) in enumerate(zip(wheel_facing, obj["pos"]))
        ]

        wheel_on_ground = [
            False if (j*LIFECYCLE > obj["alive"] or 
                    j*LIFECYCLE > obj["in_machine"])
            else (pos[1] - machine_lowest_y[min(j,len(machine_lowest_y)-1)] < 0.5)
            for j, pos in enumerate(obj["pos"])
        ]

        machine_power_list.append({
            "orderid": i,
            "name": obj["name"],
            "pos": obj["pos"],
            "power": wheel_power,
            "facing": wheel_facing,
            "have_ground_power": wheel_have_ground_power,
            "on_ground": wheel_on_ground,
            "alive": obj["alive"],
            "in_machine": obj["in_machine"],
        })

    return {
        "machine_stat_list": machine_stat_list,
        "machine_power_list": machine_power_list,
        "machine_facings": machine_facings,
        "machine_rots": machine_rots,
        "machine_lowest_y": machine_lowest_y,
    }

def target_block_check(game_state_dict,target_block_name,find_all=False):
    all_target_obj=[]
    for obj_stat in game_state_dict:
        if target_block_name in obj_stat["name"]:
            if not find_all:
                return True, obj_stat
            else:
                all_target_obj.append(obj_stat)
    if not find_all or len(all_target_obj)==0:
        return False,None

    return True,all_target_obj
def target_block_movingcheck(game_state_dict, target_block_name, goal_obj_name):
    
    target_block_infos = [obj for obj in game_state_dict if target_block_name in obj["name"]]
    goal_obj_infos = [obj for obj in game_state_dict if goal_obj_name in obj["name"]]
    
    
    average_goal_pos = np.mean([obj["pos"][0] for obj in goal_obj_infos], axis=0)
    
    target_block_distances = []
    trajectory_descriptions = []
    
    for target_block_info in target_block_infos:
        
        distances = np.round(np.linalg.norm(target_block_info["pos"] - average_goal_pos, axis=1), 2)
        target_block_distances.append(distances)
        
        
        smoothed = gaussian_filter1d(distances, sigma=1)
        deltas = np.diff(smoothed)
        
        states = []
        durations = []
        current_state = None
        duration = LIFECYCLE
        
        for delta in deltas:
            new_state = "Approaching" if delta < -1 else ("Moving away" if delta > 1 else "Stagnationary")
            
            if new_state != current_state:
                if current_state is not None:
                    states.append(f"{current_state}")
                    durations.append(duration)
                current_state = new_state
                duration = LIFECYCLE
            else:
                duration += LIFECYCLE
        
        if current_state is not None:
            last_duration = durations[-1] if durations else 0
            states.append(f"{current_state}({last_duration}-{duration}s)")
            durations.append(duration)
            
        trajectory_descriptions.append({
            "trajectory_description": states,
            "trajectory_description_id": durations
        })
    
    return {
        "target_block_distances": target_block_distances,
        "trajectory_descriptions": trajectory_descriptions
    }
def get_roughly_facing(rotated_vector):
    projections = {
        'x': rotated_vector[0],
        'y': rotated_vector[1],
        'z': rotated_vector[2]
    }
    max_axis = max(projections, key=lambda axis: abs(projections[axis]))
    max_value = projections[max_axis]
    if max_value > 0:
        direction = f'{max_axis}+'
    else:
        direction = f'{max_axis}-'
    return direction



def feedback_overallmachine(bsg_machine,game_state_dict,sim_time,check_time,feedback,check_facing=True,check_damage=True):
    machine_stat = check_machine_stat(bsg_machine,game_state_dict,[0,int(sim_time/LIFECYCLE)])
    if check_facing:
        check_facings = machine_stat["machine_facings"][:int(check_time/LIFECYCLE)]
        facing_changes = len([key for key, group in groupby(check_facings)])
        if facing_changes>((sim_time/LIFECYCLE)/4):
            counter = Counter(check_facings)
            most_common_four = [key for key,group in counter.most_common(4)]
            feedback += ("machine frequently changes in the following orientations"+"\n")
            feedback += str(most_common_four)
            feedback += "\n"
        else:
            counter = Counter(check_facings)
            most_common_one = [key for key,group in counter.most_common(1)][0]
            if most_common_one!="z+":
                feedback += ("machine orientation in long time"+"\n")
                feedback += str(most_common_one)
                feedback += "\n"
    if check_damage:
        for block_error_stat in machine_stat["machine_stat_list"]:
            if block_error_stat["happened_time"]<check_time:
                if block_error_stat["state"]=="damaged":
                    feedback += ("machine damaged"+"\n")
                    feedback += ("machine parts"+"\n")
                    feedback += str(block_error_stat["name"])+" order_id:"+str(block_error_stat["orderid"])
                    feedback += "\n"
                    feedback += ("occurred at"+"\n")
                    feedback += str(block_error_stat["happened_time"])+"sec"
                    feedback += "\n"
    return feedback


def _feedbacktool_get_obj_traj(obj_traj,focus_period):
    sample_interval = int(1/LIFECYCLE)
    
    num_intervals = len(obj_traj) // sample_interval 
    pos_focus_period = range_clip(num_intervals,focus_period)
    for var in pos_focus_period:
        var=var*sample_interval
    positions = obj_traj[pos_focus_period[0]:pos_focus_period[1]]
    positions_str = [f"[{pos[0]}, {pos[1]}, {pos[2]}]" for pos in positions]
    pos_result_str = ", ".join(positions_str)
    return pos_result_str,pos_focus_period

def feedback_distance_between_objs(game_state_dict,feedback,obj_a_name="Boulder",a_costume_name=None,obj_b_name="GOAL_Trigger",b_costume_name=None,
                                   focus_period=[0,5],need_distance=True,need_a_position=True,need_b_position=True):
    has_obj_a,obj_a_stat = target_block_check(game_state_dict,obj_a_name,find_all=True)
    has_obj_b,obj_b_stat = target_block_check(game_state_dict,obj_b_name,find_all=True)
    if not has_obj_a:
        feedback +=f"no necessary block {obj_a_name} \n"
    if not has_obj_b:
        feedback +=f"no necessary block {obj_b_name} \n"
    if (not has_obj_a) or (not has_obj_b):
        return feedback, {
            "name_a":obj_a_name,
            "orderid_a":None,
            "distance":0,
        }
    
    best_distance=-1
    best_obj_a=None
    best_obj_b=None

    for obj_a in obj_a_stat:       
        last_a_pos = obj_a["pos"][-1]
        for obj_b in obj_b_stat:
            last_b_pos = obj_b["pos"][-1]
            distance = np.linalg.norm(last_a_pos - last_b_pos)
            if distance>best_distance:
                best_distance= round(distance, 1)
                best_obj_a = obj_a
                best_obj_b = obj_b   
        
    best_orderid_a = best_obj_a["orderid"]
    best_orderid_b = best_obj_b["orderid"]
    
    best_a_traj = np.array(best_obj_a["pos"])
    best_b_traj = np.array(best_obj_b["pos"])
    
    #pos
    a_pos_result_str,focus_period_a = _feedbacktool_get_obj_traj(best_a_traj,focus_period)
    b_pos_result_str,focus_period_b = _feedbacktool_get_obj_traj(best_b_traj,focus_period)
    
    
    if "GOAL" not in obj_a_name:
        fill_a_orderid=best_orderid_a
    else:
        fill_a_orderid=""
    if "GOAL" not in obj_b_name:
        fill_b_orderid=best_orderid_b
    else:
        fill_b_orderid=""
    if a_costume_name:
        fill_obj_a_name = a_costume_name
    else:
        fill_obj_a_name = obj_a_name
    if b_costume_name:
        fill_obj_b_name = b_costume_name
    else:
        fill_obj_b_name = obj_b_name
    
    if need_distance:
        feedback += f"final distance between {fill_obj_a_name}{fill_a_orderid} and {fill_obj_b_name}{fill_b_orderid}\n{best_distance}\n"
    
    
    if need_a_position: 
        if "GOAL" not in obj_a_name:
            feedback += f"block order_id\n{best_orderid_a}\n"
        else:
            feedback += f"{fill_obj_a_name}, this target is not a block, it's not on the mahcine.\n"
    
        feedback+= f"{fill_obj_a_name} actual position in {focus_period_a} seconds\n{a_pos_result_str}\n"
    if need_b_position: 
        if "GOAL" not in obj_b_name:
            feedback += f"block order_id\n{best_orderid_b}\n"
        else:
            feedback += f"{fill_obj_b_name}, this target is not a block, it's not on the mahcine.\n"
    
        feedback+= f"{fill_obj_b_name} actual position in {focus_period_b} seconds\n{b_pos_result_str}\n"
    
    
    
    return feedback,{
        "name_a":obj_a_name,
        "orderid_a":best_orderid_a,
        "positions_a":a_pos_result_str,
        
        "name_b":obj_b_name,
        "orderid_b":best_orderid_b,
        "positions_b":b_pos_result_str,
        
        "distance":best_distance,
    }


def feedback_get_target(game_state_dict,feedback,block_name="Boulder",costume_name=None,need_distance=True,parabola=True,focus_period=[0,5],
                        need_max_speed=True,need_max_height=True,need_avg_speed=True,need_position=True):
    has_thrownobj,thrownobj_stat = target_block_check(game_state_dict,block_name,find_all=True)
    if not has_thrownobj:
        feedback +=f"no necessary block {block_name} \n"
        return feedback, {
            "name":block_name,
            "orderid":None,
            "distance":0,
        }
    
    
    best_distance=-1
    best_throwobj=None
    for thrownobj in thrownobj_stat:
        init_pos = thrownobj["pos"][0]            
        last_pos = thrownobj["pos"][-1]
        distance = np.linalg.norm(init_pos - last_pos)
        if distance>best_distance:
            best_distance= round(distance, 1)
            best_throwobj = thrownobj
    best_orderid = best_throwobj["orderid"]
    if parabola:
        try:
            best_distance,land_idx = fit_throw_traj(best_throwobj["pos"],get_land_idx=True)
        except:
            best_distance=0
            land_idx=0
    
    best_traj = np.array(best_throwobj["pos"])
    velocities = (best_traj[1:] - best_traj[:-1]) / LIFECYCLE
    speeds = np.linalg.norm(velocities, axis=1)
    try:
        max_speed = round(np.max(speeds),1)
    except:
        print("best_traj may only have one position")
        print(best_traj)
        max_speed = 0.0
    max_height = round(np.max(best_traj[:, 1]),2)
    
    sample_interval = int(1/LIFECYCLE)
    #velocity
    num_intervals = len(speeds) // sample_interval 
    speed_focus_period = range_clip(num_intervals,focus_period)
    speeds_reshaped = speeds[:num_intervals * sample_interval].reshape(num_intervals, sample_interval)
    average_speeds = np.mean(speeds_reshaped[speed_focus_period[0]:speed_focus_period[1],:], axis=1)
    average_speeds = np.round(average_speeds, 1)
    average_speeds = average_speeds.tolist()
    #pos
    num_intervals = len(best_traj) // sample_interval 
    pos_focus_period = range_clip(num_intervals,focus_period)
    for var in pos_focus_period:
        var=var*sample_interval
    first_five_positions = best_traj[pos_focus_period[0]:pos_focus_period[1]]
    positions_str = [f"[{pos[0]}, {pos[1]}, {pos[2]}]" for pos in first_five_positions]
    pos_result_str = ", ".join(positions_str)
    
    if "GOAL" not in block_name:
        feedback += f"block order_id\n{best_orderid}\n"
    else:
        feedback += f"This target is not a block, it's not on the mahcine.\n"
    if costume_name:
        block_name = costume_name
    if need_distance:
        feedback += f"{block_name} moving distance\n{best_distance}\n"
    if need_max_speed:
        feedback+= f"{block_name} max speed\n{max_speed}\n"
    if need_max_height:
        feedback+= f"{block_name} max height\n{max_height}\n"
    if need_avg_speed:
        feedback+= f"{block_name} average speed per second in {speed_focus_period} seconds\n{average_speeds}\n"
    if need_position:
        feedback+= f"{block_name} actual position in {pos_focus_period} seconds\n{pos_result_str}\n"
    if parabola:
        feedback += f"land_idx\n{land_idx}\n"
    
    
    return feedback,{
        "name":block_name,
        "orderid":best_orderid,
        "distance":best_distance,
        "max_speed":max_speed,
        "velocity":speeds, #velocity per LIFECYCLE
        "max_height":max_height,
        "avg_velocity":average_speeds, #velocity per seconds
        "positions":pos_result_str
    }


def get_env_feedback(sim_time, game_state_dict, simulate_menu, bsg_machine,is_win=False, **kwargs):
    """
    kargs:
    machine_json=None,
    required_feedback=None,
    use_querier = True,
    designer_output = None,
    threeDinfo = None,
    block_limitations=[],
    save_path=None,
    simulate_loop = None,
    return_scores=False,
    agentic_pipeline = None
    """

    machine_json = kwargs.get('machine_json', None)
    required_feedback = kwargs.get('required_feedback', None)
    use_querier = kwargs.get('use_querier', True)
    designer_output = kwargs.get('designer_output', None)
    threeDinfo = kwargs.get('threeDinfo', None)
    block_limitations = kwargs.get('block_limitations', [])
    save_path = kwargs.get('save_path', None   )
    simulate_loop = kwargs.get('simulate_loop', None)
    return_scores = kwargs.get('return_scores', True)
    agentic_pipeline = kwargs.get('agentic_pipeline', None)
    
    win_condition = simulate_menu["win_condition"]
    target_name = simulate_menu.get("target_name")
                
    feedback=""
    feedback+=f"Task type:{win_condition}\n"
    
    # print("win_condition:",win_condition)
    if win_condition== "Boulder_throw":
        check_time = sim_time
        feedback = feedback_overallmachine(bsg_machine,game_state_dict,sim_time,check_time,feedback)
        
        
        feedback,target_block_infos= feedback_get_target(game_state_dict,feedback,"Boulder")
        scores = target_block_infos["distance"]

        feedback,_= feedback_get_target(game_state_dict,feedback,"startingBlock"
                                                         ,need_distance=False,parabola=False,need_max_speed=False,need_max_height=False,
                                                         need_avg_speed=False,need_position=True)
        
    elif win_condition== "Car_distance":
        check_time = sim_time
        feedback = feedback_overallmachine(bsg_machine,game_state_dict,sim_time,check_time,feedback)
        
        feedback,starting_block_infos= feedback_get_target(game_state_dict,feedback,"startingBlock"
                                                         ,costume_name="car",need_distance=True,parabola=False,need_max_speed=True,need_max_height=True,
                                                         need_avg_speed=True,need_position=False)
        scores = starting_block_infos["distance"]
    elif win_condition=="Target_deliver":
        feedback+=f"Target name:GOAL_{target_name}\n"
        check_time = sim_time
        feedback = feedback_overallmachine(bsg_machine,game_state_dict,sim_time,check_time,feedback)
        feedback,target_block_infos= feedback_get_target(game_state_dict,feedback,f"GOAL_{target_name}",parabola=False,focus_period=[0,21])
        scores = target_block_infos["distance"]
    elif win_condition=="Boulder_throw_with_target":
        check_time = sim_time
        feedback = feedback_overallmachine(bsg_machine,game_state_dict,sim_time,check_time,feedback)
        
        feedback,selected_obj_infos= feedback_distance_between_objs(game_state_dict,feedback,"Boulder",obj_b_name=f"GOAL_{target_name}",need_a_position=True,need_b_position=False)
        scores = selected_obj_infos["distance"]     
    else:
        raise TypeError("unknown task type")
     
    if is_win:
        feedback+="Game Win!\nTrue\n"
    
    if use_querier:
        # kargs
        # {task_definition}
        # {jsoninfo}
        # {threedinfo}
        # {env_feedback}
        # block_limitations
        task_definition = extract_json_from_string(designer_output["designer_output"])["definition"]
        env_feedback = feedback
        threedinfo = threeDinfo
        block_limitations =list({int(block['tid']) for block in threeDinfo 
                        if isinstance(block, dict) and 'tid' in block})
        block_limitations = list(set(block_limitations+block_limitations))
        json_info = machine_json
        
        kargs = {
            "task_definition":task_definition,
            "jsoninfo":json_info,
            "threedinfo":threeDinfo,
            "env_feedback":env_feedback,
            "block_limitations":block_limitations
        }
        total_input,result,kargs = agentic_pipeline.ask_env_querier(kargs=kargs)
        
        querier_input_save_path = os.path.join(save_path,f"env_querier_input_{simulate_loop}.txt")
        querier_result_save_path = os.path.join(save_path,f"env_querier_output_{simulate_loop}.txt")
        txt_file_paths = [querier_input_save_path,querier_result_save_path]
        txt_contents = [total_input,result]
        for j,txt_file_path in enumerate(txt_file_paths):
            write_file(txt_file_path,str(txt_contents[j]))
        required_feedback = agentic_pipeline.get_required_list([result])[0]



    
    """
    [
        {
            "order_id": int,
            "duration": [float,float],
            "properties": ["position", "rotation", "velocity","length"],
        },
        ...
    ]
    """
    if required_feedback and len(required_feedback) > 0:
        feedback += f"{'builder required feedback'}\n"
        
        for attn_block in required_feedback:
            attn_block_orderid = int(attn_block["order_id"])
            if attn_block_orderid >= len(machine_json):
                continue
                
            attn_block_type = machine_json[attn_block_orderid]["id"]
            duration = [min(t, sim_time) for t in attn_block["duration"]]
            
            if "length" in attn_block["properties"] and str(attn_block_type) not in LINEAR_BLOCKS:
                attn_block["properties"].remove("length")
            
            obj_info = game_state_dict[attn_block_orderid]
            obj_info["alive"] = round(obj_info["alive"] - LIFECYCLE, 2)
            
            
            total_length_record = []
            if str(attn_block_type) in LINEAR_BLOCKS:
                parent_a = machine_json[attn_block_orderid]["parent_a"]
                parent_b = machine_json[attn_block_orderid]["parent_b"]
                a_data = game_state_dict[parent_a]
                b_data = game_state_dict[parent_b]
                total_length_record = [
                    round(np.linalg.norm(
                        easy_get_bppos(machine_json[parent_a]["id"], a_data["pos"][i], a_data["quat"][i], 
                        machine_json[attn_block_orderid]["bp_id_a"]) -
                        easy_get_bppos(machine_json[parent_b]["id"], b_data["pos"][i], b_data["quat"][i],
                        machine_json[attn_block_orderid]["bp_id_b"])
                    ), 1)  
                    for i in range(min(len(a_data["pos"]), len(b_data["pos"])))
                ]

            
            def get_slice(data, start, end):
                start_idx = min(int(start/LIFECYCLE), len(data))
                end_idx = min(int(end/LIFECYCLE), len(data))
                return data[start_idx:end_idx]

            broken_time = obj_info["alive"] if obj_info["alive"] < duration[1] else -1
            parent_broken_time = len(total_length_record) if total_length_record and len(total_length_record) < duration[1] else -1
            
            position = get_slice(obj_info["pos"], duration[0], broken_time if broken_time != -1 else duration[1])
            rotation = get_slice(obj_info["quat"], duration[0], broken_time if broken_time != -1 else duration[1])
            velocity = get_slice(obj_info["vel"], duration[0], broken_time if broken_time != -1 else duration[1])
            length = get_slice(total_length_record, duration[0], parent_broken_time if parent_broken_time != -1 else duration[1]) if total_length_record else []

            
            feedback += f"{'block order_id'}\n{attn_block_orderid}\n"
            feedback += f"{'block type_id'}\n{attn_block_type}\n"
            feedback += f"{'block info duration(s)'}\n{duration}\n"
            
            if broken_time != -1:
                feedback += f"{'block broken before below time(s)'}\n{broken_time}\n"
            
            for prop in attn_block["properties"]:
                prop_data = {
                    "position": (position, broken_time, "block"),
                    "rotation": (rotation, broken_time, "block"),
                    "velocity": (velocity, broken_time, "block"), 
                    "length": (length, parent_broken_time, "some parent block")
                }.get(prop)
                
                if prop_data:
                    feedback += f"block {prop}\n{prop_data[0]}\n"
                    if prop_data[1] != -1:
                        feedback += f"because of {prop_data[2]} damaged, {prop} incomplete or not recorded\n"

    if return_scores:
        scores = round(scores, 2)
        return feedback, scores
    else:
        return feedback

def process_bsg(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    blocks_info = []
    for orderid, block in enumerate(root.find('Blocks')):
        position = block.find('Transform/Position')
        x = float(position.attrib['x'])
        y = float(position.attrib['y'])
        z = float(position.attrib['z'])
        rotation = block.find('Transform/Rotation')
        rx = float(rotation.attrib['x'])
        ry = float(rotation.attrib['y'])
        rz = float(rotation.attrib['z'])
        rw = float(rotation.attrib['w'])
        block_info = {
            'orderid': orderid,
            'position': [x,y,z],
            'rotation': [rx, ry, rz, rw]
        }
        blocks_info.append(block_info)
        start_block_pos = np.array(blocks_info[0]['position'])
        curr_block_pos = np.array(block_info["position"])
        startblock_dis = np.abs(start_block_pos - curr_block_pos)
        startblock_dis = np.round(startblock_dis, 2)
        startblock_dis = round(np.linalg.norm(startblock_dis), 2)
        blocks_info[-1]["startblock_dis"] = startblock_dis
    return blocks_info

from scipy.optimize import curve_fit
def fit_throw_traj(traj,get_land_idx=False):
    boulder_x, boulder_y, boulder_z, boulder_t = [], [], [], []

    t, dt = 0.0, LIFECYCLE
    for rec in traj:
        x, y, z = rec[0],rec[1],rec[2]
        boulder_x.append(x)
        boulder_y.append(y)
        boulder_z.append(z)
        boulder_t.append(t)
        t += dt

    boulder_x = np.array(boulder_x)
    boulder_y = np.array(boulder_y)
    boulder_z = np.array(boulder_z)
    boulder_t = np.array(boulder_t)

    xz = np.column_stack((boulder_x, boulder_z))
    start_xz = xz[0]
    s = np.linalg.norm(xz - start_xz, axis=1)


    win = 5
    dy = np.convolve(np.gradient(boulder_y, boulder_t),
                    np.ones(win)/win, mode='same')
    thresh = 0.05
    land_idx = np.where((dy[:-1] < -thresh) & (dy[1:] > -thresh))[0]
    if len(land_idx) == 0:
        flight_mask = np.ones_like(boulder_t, dtype=bool)
    else:
        land_idx = land_idx[0] + 1
        flight_mask = boulder_t <= boulder_t[land_idx]
    flight_mask = flight_mask | (boulder_y >= 3.0)   # y ≥ 3 m is flying
    if flight_mask.sum() > 3:
        def para_y_s(s, a, b, c):
            return a * s**2 + b * s + c
        popt, _ = curve_fit(para_y_s,
                            s[flight_mask],
                            boulder_y[flight_mask])
        a, b, c = popt
        # smooth curve
        s_fit = np.linspace(0, s[flight_mask].max(), 200)
        y_fit = para_y_s(s_fit, a, b, c)
    else:
        s_fit = y_fit = [0]

    idx_land = np.argmin(np.abs(boulder_t - boulder_t[flight_mask][-1]))
    if get_land_idx:
        return round(s_fit[-1],2),idx_land
    else:
        return round(s_fit[-1],2)

def get_facing_from_json(response):
    content = deepcopy(response) if isinstance(response, list) else extract_json_from_string(response)
    order_facing_list = []
    
    
    for info in content:
        facing_dict = {}
        facing_dict["order_id"] = info["order_id"]
        if str(info["id"]) in LINEAR_BLOCKS:
            continue
        
        if int(info["order_id"])==0:
            facing_dict["facing"]="z+"
        else:
            parent_type = int(content[info["parent"]]["id"])
            parent_order = int(info["parent"])
            relative_facing_str = get_bpfacing(parent_type, info["bp_id"])
            
            parent_real_facing_str="z+"
            for facing_info in order_facing_list:
                if int(facing_info["order_id"])==parent_order:
                    parent_real_facing_str = facing_info["facing"]
                    break
            
            facing_dict["facing"]=FACINGMAP[parent_real_facing_str][relative_facing_str]
        order_facing_list.append(facing_dict)
    return order_facing_list
def get_machine_block_num(response):
    content = deepcopy(response) if isinstance(response, list) else extract_json_from_string(response)
    num_count_dict={}
    
    for info in content:
        block_type = str(info["id"])
        if block_type in num_count_dict:
            num_count_dict[block_type]+=1
        else:
            num_count_dict[block_type]=1
    return num_count_dict


if __name__ == '__main__':
    pass