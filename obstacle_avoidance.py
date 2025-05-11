from robobopy.utils.IR import IR


def needs_avoidance(robot, threshold=12):
    front_sensors = [IR.FrontC, IR.FrontLL, IR.FrontRR]
    for s in front_sensors:
        if (robot.readIRSensor(s) or 0) > threshold:
            return True
    return False

def undo_if_needed(robot, angle, duration=1.0):
    if not needs_avoidance(robot):
        return False
    
    print("Undo last movement")
    spin_speed = -20
    forward_speed = -20

    robot.moveWheelsByTime(forward_speed, forward_speed, duration)

    t_turn = abs(-angle) / 180.0 * 1.75
    if angle > 0:
        robot.moveWheelsByTime(-spin_speed, spin_speed, t_turn)
    elif angle < 0:
        robot.moveWheelsByTime(spin_speed, -spin_speed, t_turn)
    
    robot.wait(0.1)

    return True