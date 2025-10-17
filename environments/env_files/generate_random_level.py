import os
import shutil
import uuid
import random
import json
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R


def copy_template(template_path: str, output_path: str) -> None:
    if not os.path.isfile(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    shutil.copyfile(template_path, output_path)


def ensure_objects_root(tree: ET.ElementTree) -> ET.Element:
    root = tree.getroot()
    objects = root.find('./Objects')
    if objects is None:
        objects = ET.SubElement(root, 'Objects')
    return objects


def _new_object_element(prefab_id: int, position, scale=None, rotation=None, object_id: str | None = None, enable_physics: bool | None = None, lel_global: bool | None = None) -> ET.Element:
    obj = ET.Element('Object')
    if object_id is not None:
        obj.set('ID', str(object_id))
    else:
        obj.set('ID', _generate_object_id())
    obj.set('Prefab', str(prefab_id))

    pos = ET.SubElement(obj, 'Position')
    pos.set('x', f"{position[0]}")
    pos.set('y', f"{position[1]}")
    pos.set('z', f"{position[2]}")

    # Only add Rotation element if rotation is specified
    if rotation is not None:
        rot = ET.SubElement(obj, 'Rotation')
        rot.set('x', f"{rotation[0]}")
        rot.set('y', f"{rotation[1]}")
        rot.set('z', f"{rotation[2]}")
        rot.set('w', f"{rotation[3]}")

    # Only add Scale element if scale is specified
    if scale is not None:
        scl = ET.SubElement(obj, 'Scale')
        scl.set('x', f"{scale[0]}")
        scl.set('y', f"{scale[1]}")
        scl.set('z', f"{scale[2]}")

    # Create Data element with physics and global settings if any are specified
    if enable_physics is not None or lel_global is not None:
        data = ET.SubElement(obj, 'Data')
        if enable_physics is not None:
            boolean_elem = ET.SubElement(data, 'Boolean')
            boolean_elem.set('key', 'bmt-lel-enable-physics')
            boolean_elem.text = 'True' if enable_physics else 'False'

        if lel_global is not None:
            global_elem = ET.SubElement(data, 'Boolean')
            global_elem.set('key', 'bmt-lel-global')
            global_elem.text = 'True' if lel_global else 'False'
    else:
        # Always include empty Data element for consistency
        ET.SubElement(obj, 'Data')

    return obj


def _generate_object_id() -> str:
    """
    Generate a signed 64-bit integer ID using UUID4's lower 64 bits, to match
    the style seen in existing .blv files (which can be negative or positive).
    """
    low64 = uuid.uuid4().int & ((1 << 64) - 1)
    if low64 >= (1 << 63):
        low64 -= (1 << 64)
    return str(low64)


def add_fixed_object(tree: ET.ElementTree, prefab_id: int, position, scale=None, rotation=None, enable_physics: bool | None = None, lel_global: bool | None = None) -> None:
    objects = ensure_objects_root(tree)
    obj = _new_object_element(prefab_id=prefab_id, position=position, scale=scale, rotation=rotation, enable_physics=enable_physics, lel_global=lel_global)
    objects.append(obj)


# Removed external merge: template determines base objects


def add_random_objects(
    tree: ET.ElementTree,
    region_ranges,
    items: list,
) -> None:
    """
    items: list of dicts with keys:
      - prefab: int
      - count: int
      - scale_min: (sx, sy, sz) optional - if not provided, no Scale element will be added
      - scale_max: (sx, sy, sz) optional - if not provided, no Scale element will be added
      - enable_physics: bool optional
      - rotation_range: [[rx_min, rx_max], [ry_min, ry_max], [rz_min, rz_max]] optional (Euler angles in degrees) - if not provided, no Rotation element will be added
    region_ranges: ((xmin,xmax), (ymin,ymax), (zmin,zmax))
    Positions are uniformly sampled within the axis-aligned box.
    """
    for item in items:
        prefab = int(item['prefab'])
        count = int(item['count'])
        scale_min = item.get('scale_min')
        scale_max = item.get('scale_max')
        enable_physics = item.get('enable_physics')
        rotation_range = item.get('rotation_range')
        lel_global = item.get('lel_global')

        for _ in range(count):
            px = random.uniform(region_ranges[0][0], region_ranges[0][1])
            py = random.uniform(region_ranges[1][0], region_ranges[1][1])
            pz = random.uniform(region_ranges[2][0], region_ranges[2][1])

            # Calculate scale only if both scale_min and scale_max are provided
            scale = None
            if scale_min is not None and scale_max is not None:
                sx = random.uniform(scale_min[0], scale_max[0])
                sy = random.uniform(scale_min[1], scale_max[1])
                sz = random.uniform(scale_min[2], scale_max[2])
                scale = (sx, sy, sz)

            # Generate random rotation if rotation_range is specified
            rotation_quat = None
            if rotation_range:
                rx = random.uniform(rotation_range[0][0], rotation_range[0][1])
                ry = random.uniform(rotation_range[1][0], rotation_range[1][1])
                rz = random.uniform(rotation_range[2][0], rotation_range[2][1])
                # Convert Euler angles (degrees) to quaternion using scipy
                rot = R.from_euler('xyz', [rx, ry, rz], degrees=True)
                quat = rot.as_quat()  # Returns [x, y, z, w]
                rotation_quat = (quat[0], quat[1], quat[2], quat[3])

            add_fixed_object(tree, prefab, (px, py, pz), scale=scale, rotation=rotation_quat, enable_physics=enable_physics, lel_global=lel_global)


def add_group(
    tree: ET.ElementTree,
    group_definition: dict,
    anchor_position,
    anchor_uniform_scale: float = 1.0,
    group_rotation: tuple | None = None,
) -> None:
    """
    group_definition: {
        'members': [
            { 'prefab': int, 'position': (dx, dy, dz), 'scale': (sx, sy, sz), 'rotation': (rx, ry, rz) (optional, Euler degrees), 'enable_physics': bool (optional) }
        ]
    }
    Members' relative transforms are preserved; only translated by anchor and scaled uniformly by anchor_uniform_scale.
    Group rotation is applied to the entire group, preserving relative positions and rotations.
    """
    # Create group rotation matrix if specified
    group_rot_matrix = None
    if group_rotation:
        group_rot_matrix = R.from_quat([group_rotation[0], group_rotation[1], group_rotation[2], group_rotation[3]])
    
    for member in group_definition.get('members', []):
        prefab = member['prefab']
        rel_pos = member.get('position', (0, 0, 0))
        rel_scale = member.get('scale', (1.0, 1.0, 1.0))
        enable_physics = member.get('enable_physics')
        member_euler_rotation = member.get('rotation')  # Euler angles in degrees

        # Apply group rotation to relative position if group is rotated
        if group_rot_matrix:
            rotated_pos = group_rot_matrix.apply([rel_pos[0], rel_pos[1], rel_pos[2]])
            world_pos = (
                anchor_position[0] + rotated_pos[0],
                anchor_position[1] + rotated_pos[1],
                anchor_position[2] + rotated_pos[2],
            )
        else:
            world_pos = (
                anchor_position[0] + rel_pos[0],
                anchor_position[1] + rel_pos[1],
                anchor_position[2] + rel_pos[2],
            )
        
        world_scale = (
            anchor_uniform_scale * rel_scale[0],
            anchor_uniform_scale * rel_scale[1],
            anchor_uniform_scale * rel_scale[2],
        )
        
        # Calculate final rotation (group rotation * member rotation)
        final_rotation = None
        if member_euler_rotation or group_rotation:
            if member_euler_rotation and group_rotation:
                # Combine group rotation with member's local rotation
                member_rot = R.from_euler('xyz', member_euler_rotation, degrees=True)
                combined_rot = group_rot_matrix * member_rot
                quat = combined_rot.as_quat()  # [x, y, z, w]
                final_rotation = (quat[0], quat[1], quat[2], quat[3])
            elif group_rotation:
                final_rotation = group_rotation
            else:  # member_euler_rotation only
                member_rot = R.from_euler('xyz', member_euler_rotation, degrees=True)
                quat = member_rot.as_quat()  # [x, y, z, w]
                final_rotation = (quat[0], quat[1], quat[2], quat[3])
        
        add_fixed_object(tree, prefab, world_pos, world_scale, rotation=final_rotation, enable_physics=enable_physics)


def add_random_groups(
    tree: ET.ElementTree,
    region_ranges,
    groups: list,
) -> None:
    """
    groups: list of dicts with keys:
      - definition: group_definition (see add_group)
      - count: int
      - uniform_scale_min: float optional
      - uniform_scale_max: float optional
      - rotation_range: [[rx_min, rx_max], [ry_min, ry_max], [rz_min, rz_max]] optional (Euler angles in degrees)
    """
    for g in groups:
        definition = g['definition']
        count = int(g['count'])
        smin = float(g.get('uniform_scale_min', 1.0))
        smax = float(g.get('uniform_scale_max', 1.0))
        rotation_range = g.get('rotation_range')

        for _ in range(count):
            px = random.uniform(region_ranges[0][0], region_ranges[0][1])
            py = random.uniform(region_ranges[1][0], region_ranges[1][1])
            pz = random.uniform(region_ranges[2][0], region_ranges[2][1])
            s = random.uniform(smin, smax)
            
            # Generate random group rotation if rotation_range is specified
            group_rotation = None
            if rotation_range:
                rx = random.uniform(rotation_range[0][0], rotation_range[0][1])
                ry = random.uniform(rotation_range[1][0], rotation_range[1][1])
                rz = random.uniform(rotation_range[2][0], rotation_range[2][1])
                # Convert Euler angles (degrees) to quaternion
                rot = R.from_euler('xyz', [rx, ry, rz], degrees=True)
                quat = rot.as_quat()  # Returns [x, y, z, w]
                group_rotation = (quat[0], quat[1], quat[2], quat[3])
            
            add_group(tree, definition, (px, py, pz), s, group_rotation=group_rotation)


def save_tree(tree: ET.ElementTree, output_path: str) -> None:
    _indent_with_tabs(tree.getroot())
    tree.write(output_path, encoding='utf-8', xml_declaration=False)


def _indent_with_tabs(elem: ET.Element, level: int = 0) -> None:
    """
    Pretty-print XML using tab indentation to mirror template.blv style.
    """
    indent_str = "\n" + ("\t" * level)
    child_indent_str = "\n" + ("\t" * (level + 1))

    if len(elem):
        if elem.text is None or not elem.text.strip():
            elem.text = child_indent_str
        for child in list(elem):
            _indent_with_tabs(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = indent_str
    else:
        if elem.text is not None and not elem.text.strip():
            elem.text = None
    # Ensure tail for siblings
    if level > 0:
        if elem.tail is None or not elem.tail.strip():
            elem.tail = indent_str


def load_items_library(items_file: str = 'items_library.json') -> dict:
    """
    Load the items library from JSON file.
    
    Args:
        items_file: Path to the JSON file containing item definitions
    
    Returns:
        Dictionary containing all item definitions
    """
    base_dir = os.path.dirname(__file__)
    items_path = os.path.join(base_dir, items_file)
    
    with open(items_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('items', {})


def load_groups_library(groups_file: str = 'groups_library.json') -> dict:
    """
    Load the groups library from JSON file.
    
    Args:
        groups_file: Path to the JSON file containing group definitions
    
    Returns:
        Dictionary containing all group definitions
    """
    base_dir = os.path.dirname(__file__)
    groups_path = os.path.join(base_dir, groups_file)
    
    with open(groups_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('groups', {})


def resolve_items(items_configs: list, items_library: dict) -> list:
    """
    Resolve item references to actual item configurations.
    
    Args:
        items_configs: List of item configurations with item_id references
        items_library: Dictionary of available items
    
    Returns:
        List of resolved item configurations with prefab IDs and scales
    """
    resolved = []
    
    for item_config in items_configs:
        item_id = item_config.get('item_id')
        
        if not item_id:
            # If no item_id, assume it's already a complete config (backward compatibility)
            resolved.append(item_config)
            continue
        
        if item_id not in items_library:
            available = ', '.join(items_library.keys())
            raise ValueError(f"Item '{item_id}' not found in library. Available items: {available}")
        
        item_def = items_library[item_id]

        # Build resolved item config with optional scale parameters
        resolved_item = {
            'prefab': item_def['prefab'],
            'count': item_config['count'],
        }

        # Add scale_min and scale_max if provided in preset
        if 'scale_min' in item_config:
            resolved_item['scale_min'] = item_config['scale_min']
        if 'scale_max' in item_config:
            resolved_item['scale_max'] = item_config['scale_max']
        
        # Check if enable_physics is specified in preset, otherwise use library default
        if 'enable_physics' in item_config:
            resolved_item['enable_physics'] = item_config['enable_physics']
        elif 'enable_physics' in item_def:
            resolved_item['enable_physics'] = item_def['enable_physics']

        # Check if lel_global is specified in preset, otherwise use library default
        if 'lel_global' in item_config:
            resolved_item['lel_global'] = item_config['lel_global']
        elif 'lel_global' in item_def:
            resolved_item['lel_global'] = item_def['lel_global']
        
        # Pass through rotation_range if specified in preset
        if 'rotation_range' in item_config:
            resolved_item['rotation_range'] = item_config['rotation_range']
        
        resolved.append(resolved_item)
    
    return resolved


def resolve_groups(groups_configs: list, groups_library: dict) -> list:
    """
    Resolve group references to actual group configurations.
    
    Args:
        groups_configs: List of group configurations with group_id references
        groups_library: Dictionary of available groups
    
    Returns:
        List of resolved group configurations with definitions
    """
    resolved = []
    
    for group_config in groups_configs:
        group_id = group_config.get('group_id')
        
        if not group_id:
            # If no group_id, assume it's already a complete config (backward compatibility)
            resolved.append(group_config)
            continue
        
        if group_id not in groups_library:
            available = ', '.join(groups_library.keys())
            raise ValueError(f"Group '{group_id}' not found in library. Available groups: {available}")
        
        group_def = groups_library[group_id]
        
        # Build resolved group config
        resolved_group = {
            'definition': {
                'members': group_def['members']
            },
            'count': group_config['count'],
            'uniform_scale_min': group_config.get('uniform_scale_min', 1.0),
            'uniform_scale_max': group_config.get('uniform_scale_max', 1.0),
        }
        
        # Pass through rotation_range if specified in preset
        if 'rotation_range' in group_config:
            resolved_group['rotation_range'] = group_config['rotation_range']
        
        resolved.append(resolved_group)
    
    return resolved


def load_preset(preset_name: str = 'default', preset_file: str = 'level_presets.json') -> dict:
    """
    Load a preset configuration from the JSON file and resolve item/group references.
    
    Args:
        preset_name: Name of the preset to load (e.g., 'default', 'easy_obstacle_course')
        preset_file: Path to the JSON file containing presets
    
    Returns:
        Dictionary containing the preset configuration with resolved items and groups
    """
    base_dir = os.path.dirname(__file__)
    preset_path = os.path.join(base_dir, preset_file)
    
    with open(preset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if preset_name not in data['presets']:
        available = ', '.join(data['presets'].keys())
        raise ValueError(f"Preset '{preset_name}' not found. Available presets: {available}")
    
    preset = data['presets'][preset_name]
    
    # Load libraries
    items_library = load_items_library()
    groups_library = load_groups_library()
    
    # Resolve item and group references
    if 'random_items' in preset and preset['random_items']:
        preset['random_items'] = resolve_items(preset['random_items'], items_library)
    
    if 'random_groups' in preset and preset['random_groups']:
        preset['random_groups'] = resolve_groups(preset['random_groups'], groups_library)
    
    # Convert region_ranges from list to tuple format
    if 'random_region_ranges' in preset and preset['random_region_ranges']:
        preset['random_region_ranges'] = tuple(tuple(r) for r in preset['random_region_ranges'])
    
    return preset


def generate_level(
    template_path: str,
    output_path: str,
    random_region_ranges: tuple | None = None,
    random_items: list | None = None,
    random_groups: list | None = None,
) -> None:
    copy_template(template_path, output_path)
    tree = ET.parse(output_path)


    if random_region_ranges and random_items:
        add_random_objects(tree, random_region_ranges, random_items)

    if random_region_ranges and random_groups:
        add_random_groups(tree, random_region_ranges, random_groups)

    save_tree(tree, output_path)


def generate_level_from_preset(
    preset_name: str,
    output_path: str,
    preset_file: str = 'level_presets.json',
) -> None:
    """
    Generate a level using a preset configuration from JSON.
    
    Args:
        preset_name: Name of the preset to use
        output_path: Path where the generated level will be saved
        preset_file: Path to the JSON file containing presets
    """
    base_dir = os.path.dirname(__file__)
    preset = load_preset(preset_name, preset_file)
    
    # Resolve template path relative to script directory
    template_path = os.path.join(base_dir, preset['template_path'])
    
    generate_level(
        template_path=template_path,
        output_path=output_path,
        random_region_ranges=preset.get('random_region_ranges'),
        random_items=preset.get('random_items'),
        random_groups=preset.get('random_groups'),
    )


if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    output = os.path.join(base_dir, 'Generated_Level.blv')

    # Method 1: Use preset from JSON (recommended)
    # Try different presets: 'default', 'rotation_test', 'physics_test', 'easy_obstacle_course', 'zero_g_challenge', 'obstacle_maze'
    generate_level_from_preset(
        preset_name='default',  # Testing rotation functionality
        output_path=output,
    )
    
    print(f"Level generated successfully using preset: {output}")
