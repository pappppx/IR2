import numpy as np
import random
from robobopy.utils.IR import IR
from utils.perceptions import get_perception_vector, get_simple_perceptions

SAMPLE_DT       = 1.0
AVOID_THRESHOLD = 15


def perform_main_action(robot, sim, angle, duration=0.5, evade_thresh=18):

    perform_action(robot, sim, angle, duration)
    
    P = get_perception_vector(sim)

    loc = sim.getRobotLocation(0)["position"]
    
    evaded = go_back_if_needed(robot, angle, duration=duration, threshold=evade_thresh)
    return P, evaded, loc

def perform_action(robot, sim, angle, duration=0.5):

    spin_speed = 20
    forward_speed = 20

    t_turn = abs(angle) / 180.0 * 1.75
    if angle > 0:
        robot.moveWheelsByTime(-spin_speed, spin_speed, t_turn)
    elif angle < 0:
        robot.moveWheelsByTime(spin_speed, -spin_speed, t_turn)

    robot.moveWheelsByTime(forward_speed, forward_speed, duration)


def perform_random_action(rob, spin_speed=20, forward_speed=20, min_duration=1, max_duration=3):

    rand_angle = random.uniform(0.0, 360.0)
    t_turn = abs(rand_angle) / 180.0 * 1.75

    if rand_angle > 0:
        rob.moveWheelsByTime(-spin_speed, spin_speed, t_turn)
    elif rand_angle < 0:
        rob.moveWheelsByTime(spin_speed, -spin_speed, t_turn)

    rob.wait(0.1)

    rand_duration = random.uniform(min_duration, max_duration)
    rob.moveWheelsByTime(forward_speed, forward_speed, rand_duration)
    rob.wait(0.1)


def perform_simple_action(robot, angle, duration=0.5):

    spin_speed = 20
    forward_speed = 20

    t_turn = abs(angle) / 180.0 * 1.75

    if angle > 0:
        robot.moveWheelsByTime(-spin_speed, spin_speed, t_turn)
        robot.wait(0.1)
        if avoid_if_needed(robot):
            return None
        
    elif angle < 0:
        robot.moveWheelsByTime(spin_speed, -spin_speed, t_turn)
        robot.wait(0.1)
        if avoid_if_needed(robot):
            return None

    robot.moveWheelsByTime(forward_speed, forward_speed, duration)
    robot.wait(0.5)
    return angle
    
    
def avoid_if_needed(robot, threshold=12):

    front_sensors = [IR.FrontC, IR.FrontLL, IR.FrontRR]
    readings = {}
    for s in front_sensors:
        val = robot.readIRSensor(s) or 0
        readings[s] = val

    detected = {s: v for s, v in readings.items() if v > threshold}
    if not detected:
        return False

    sensor = max(detected, key=detected.get)
    val = detected[sensor]

    if sensor == IR.FrontC:
        angle = 180
    elif sensor == IR.FrontRR:
        angle = 180 - 45
    else:
        angle = 180 + 45

    duration = angle / 180.0 * 1.75

    print(f"  ¡Obstáculo detectado en {sensor.name} ({val})! Girando {angle}° " +
          f"({duration:.2f}s)")

    robot.moveWheelsByTime(20, -20, duration)
    robot.moveWheelsByTime(20, 20, 1.5)
    return True

def go_back_if_needed(robot, angle, duration=0.5, threshold=18):

    front_sensors = [IR.FrontC, IR.FrontLL, IR.FrontRR]
    readings = {s: robot.readIRSensor(s) or 0 for s in front_sensors}

    detected = {s: v for s, v in readings.items() if v > threshold}
    if not detected:
        return False

    spin_speed = 20
    forward_speed = 20

    robot.moveWheelsByTime(-forward_speed, -forward_speed, duration)
    robot.wait(0.1)

    t_turn = abs(angle) / 180.0 * 1.75

    if angle > 0:
        robot.moveWheelsByTime(spin_speed, -spin_speed, t_turn)
    elif angle < 0:
        robot.moveWheelsByTime(-spin_speed, spin_speed, t_turn)

    return True