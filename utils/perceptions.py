import numpy as np
from robobopy.utils.BlobColor import BlobColor
import math


def distance_between(r, o):
    dx = o['x'] - r['x']
    dz = o['z'] - r['z']
    return math.hypot(dx, dz)


# def angle_to_target(robot_pos, robot_yaw_deg, target_pos):
#     dx = target_pos['x'] - robot_pos['x']
#     dz = target_pos['z'] - robot_pos['z']
    
#     # Ángulo absoluto desde el robot hacia el objetivo
#     angle_to_target_rad = math.atan2(dz, dx)
#     angle_to_target_deg = math.degrees(angle_to_target_rad)
    
#     # Diferencia cruda entre la orientación del robot y el objetivo
#     raw = angle_to_target_deg - robot_yaw_deg
    
#     # Normalizamos a (-180, 180]
#     relative_angle = (raw + 180) % 360 - 180
    
#     return relative_angle

# def angle_to_target(robot_pos, robot_yaw_deg, target_pos):
#     dx = target_pos['x'] - robot_pos['x']
#     dz = target_pos['z'] - robot_pos['z']

#     bearing_rad = math.atan2(dx, dz)
#     bearing_deg = math.degrees(bearing_rad)

#     diff = (bearing_deg - robot_yaw_deg + 180) % 360 - 180
#     return diff


def angle_to_target(robot_pos, robot_yaw_deg, target_pos):

    dx = target_pos['x'] - robot_pos['x']
    dz = target_pos['z'] - robot_pos['z']

    bearing = math.atan2(dx, dz)
    yaw = math.radians(robot_yaw_deg)
    raw = bearing - yaw
    rel = (raw + math.pi) % (2*math.pi) - math.pi

    return math.sin(rel), math.cos(rel)


def angle_from_sin_cos(s, c, wrap_360=False):

    theta_rad = math.atan2(s, c)
    theta_deg = math.degrees(theta_rad)
    if wrap_360:
        theta_deg = theta_deg % 360
    return theta_deg


def get_simple_perceptions(sim):

    objects = sim.getObjects()
    P_t = {}

    if objects != None and len(objects) > 0:
        for obj in objects:
            obj_str = str(obj).lower()

            loc_robot = sim.getRobotLocation(0)
            robot_rotation = loc_robot["rotation"]
            robot_position = loc_robot["position"]

            if "redcylinder" in obj_str:
                loc_red = sim.getObjectLocation(obj)
                pos_red = loc_red["position"]

                P_t["red_sin"], P_t["red_cos"] = angle_to_target(robot_position, robot_rotation["y"], pos_red)
                P_t["red_dist"] = distance_between(robot_position, pos_red)

            elif "greencylinder" in obj_str:
                loc_green = sim.getObjectLocation(obj)
                pos_green = loc_green["position"]

                P_t["green_sin"], P_t["green_cos"] = angle_to_target(robot_position, robot_rotation["y"], pos_green)
                P_t["green_dist"] = distance_between(robot_position, pos_green)

            elif "bluecylinder" in obj_str:
                loc_blue = sim.getObjectLocation(obj)
                pos_blue = loc_blue["position"]

                P_t["blue_sin"], P_t["blue_cos"] = angle_to_target(robot_position, robot_rotation["y"], pos_blue)
                P_t["blue_dist"] = distance_between(robot_position, pos_blue)

    return P_t


def get_cylinder_positions(sim):

    objects = sim.getObjects()
    positions = {}

    if objects != None and len(objects) > 0:
        for obj in objects:
            obj_str = str(obj).lower()

            # Comprobamos si es el cilindro rojo
            if "redcylinder" in obj_str:
                loc_red = sim.getObjectLocation(obj)
                positions["red"] = loc_red["position"]

            elif "greencylinder" in obj_str:
                loc_green = sim.getObjectLocation(obj)
                positions["green"] = loc_green["position"]

            elif "bluecylinder" in obj_str:
                loc_blue = sim.getObjectLocation(obj)
                positions["blue"] = loc_blue["position"]

    return positions


def get_perception_vector(sim):
    P = get_simple_perceptions(sim)
    return np.array([
        P['red_rotation'],
        P['red_position'],
        P['green_rotation'],
        P['green_position'],
        P['blue_rotation'],
        P['blue_position']
    ], dtype=np.float32)