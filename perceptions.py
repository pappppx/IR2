from robobopy.utils.BlobColor import BlobColor
import math


def distance_between(r, o):
    dx = o['x'] - r['x']
    dz = o['z'] - r['z']
    return math.hypot(dx, dz)

def angle_to_target(robot_pos, robot_yaw_deg, target_pos):
    dx = target_pos['x'] - robot_pos['x']
    dz = target_pos['z'] - robot_pos['z']
    
    angle_to_target_rad = math.atan2(dz, dx)
    angle_to_target_deg = math.degrees(angle_to_target_rad)
    
    relative_angle = (angle_to_target_deg - robot_yaw_deg + 360) % 360
    
    return relative_angle

def get_simple_perceptions(sim):
    # Obtenemos la lista de objetos de la escena
    objects = sim.getObjects()

    # Inicializamos un diccionario para el vector P(t) = (red, green, blue)
    P_t = {}

    if objects != None and len(objects) > 0:
        for obj in objects:
            obj_str = str(obj).lower()

            # Obtenemos la posición del robot
            loc_robot = sim.getRobotLocation(0)
            robot_rotation = loc_robot["rotation"]
            robot_position = loc_robot["position"]

            # Comprobamos si es el cilindro rojo
            if "redcylinder" in obj_str:
                loc_red = sim.getObjectLocation(obj)
                pos_red = loc_red["position"]

                P_t["red_rotation"] = angle_to_target(robot_position, robot_rotation["y"], pos_red)
                P_t["red_position"] = distance_between(robot_position, pos_red)

            elif "greencylinder" in obj_str:
                loc_green = sim.getObjectLocation(obj)
                pos_green = loc_green["position"]

                P_t["green_rotation"] = angle_to_target(robot_position, robot_rotation["y"], pos_green)
                P_t["green_position"] = distance_between(robot_position, pos_green)

            elif "bluecylinder" in obj_str:
                loc_blue = sim.getObjectLocation(obj)
                pos_blue = loc_blue["position"]

                P_t["blue_rotation"] = angle_to_target(robot_position, robot_rotation["y"], pos_blue)
                P_t["blue_position"] = distance_between(robot_position, pos_blue)

    # Imprimimos el vector P(t) con las distancias encontradas
    return P_t

# ————— protocol de scan para complejo —————
def scan_for_cylinders(robot, wheel_speed=10):
    P = {'red':None,'green':None,'blue':None}
    robot.moveTiltTo(100,8)
    robot.wait(0.2)
    # t0 = time.time()
    robot.moveWheels(wheel_speed, -wheel_speed)
    while all(P[c] is not None for c in P) == False:
        blob_r = robot.readColorBlob(BlobColor.RED)
        blob_g = robot.readColorBlob(BlobColor.GREEN)
        blob_b = robot.readColorBlob(BlobColor.BLUE)

        # Para cada color, si detecta y aún no lo había guardado, lo almacena
        if blob_r.size > 0 and P['red'] is None:
            P['red'] = {'x': blob_r.posx, 'y': blob_r.posy, 'size': blob_r.size}
        if blob_g.size > 0 and P['green'] is None:
            P['green'] = {'x': blob_g.posx, 'y': blob_g.posy, 'size': blob_g.size}
        if blob_b.size > 0 and P['blue'] is None:
            P['blue'] = {'x': blob_b.posx, 'y': blob_b.posy, 'size': blob_b.size}

    robot.stopMotors()
    return P