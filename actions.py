import random
from robobopy.utils.IR import IR


# ————— Parámetros —————
SAMPLE_DT       = 1.0      # segundos entre muestras
AVOID_THRESHOLD = 15      # valor IR a partir del cual hay obstáculo

from perceptions import get_simple_perceptions
import numpy as np

def perform_main_action(robot, sim, angle, duration=0.5):
    """
    Ejecuta giro + avance. 
    Devuelve (S_main, evaded), donde:
      - S_main: estado justo tras la maniobra principal
                [red_rot, red_pos, green_rot, green_pos, blue_rot, blue_pos]
      - evaded: True si luego tuvo que retroceder, False en caso contrario.
    """
    spin_speed = 20
    forward_speed = 20
    # 1) Giro
    t_turn = abs(angle) / 180.0 * 1.75
    if angle > 0:
        robot.moveWheelsByTime(-spin_speed, spin_speed, t_turn)
    elif angle < 0:
        robot.moveWheelsByTime(spin_speed, -spin_speed, t_turn)
    # 2) Avance
    robot.moveWheelsByTime(forward_speed, forward_speed, duration)
    robot.wait(0.1)

    # 3) Leer percepción tras la maniobra principal
    P = get_simple_perceptions(sim)
    S_main = np.array([
        P['red_rotation'],  P['red_position'],
        P['green_rotation'],P['green_position'],
        P['blue_rotation'], P['blue_position']
    ], dtype=np.float32)

    loc = sim.getRobotLocation(0)["position"]
    
    # 4) Ver si hay que retroceder
    evaded = go_back_if_needed(robot, angle, duration)
    robot.wait(0.1)
    return S_main, evaded, loc


def perform_random_action(rob, spin_speed=20, forward_speed=20):

    rand_angle = random.uniform(0.0, 360.0)
    t_turn = abs(rand_angle) / 180.0 * 1.75

    if rand_angle > 0:
        rob.moveWheelsByTime(-spin_speed, spin_speed, t_turn)
    elif rand_angle < 0:
        rob.moveWheelsByTime(spin_speed, -spin_speed, t_turn)

    rob.wait(0.1)

    # Avance aleatorio entre 0.5 y 1.5 segundos
    rand_duration = random.uniform(0.5, 2.5)
    rob.moveWheelsByTime(forward_speed, forward_speed, rand_duration)
    rob.wait(0.1)


def perform_simple_action(robot, angle, duration=0.5):
    # tu diccionario ACCIONES de antes
    spin_speed = 20
    forward_speed = 20

    # Calcular tiempo de giro proporcional
    t_turn = abs(angle) / 180.0 * 1.75

    # Giro sobre sí mismo
    if angle > 0:
        robot.moveWheelsByTime(-spin_speed, spin_speed, t_turn)
        robot.wait(0.1)
        if avoid_if_needed(robot):
            return None # Que quede reflejado en el dataset que hizo un movimiento evasivo
        
    elif angle < 0:
        robot.moveWheelsByTime(spin_speed, -spin_speed, t_turn)
        robot.wait(0.1)
        if avoid_if_needed(robot):
            return None # Que quede reflejado en el dataset que hizo un movimiento evasivo
    # angle == 0: no gira

    # Avanzar en línea recta
    robot.moveWheelsByTime(forward_speed, forward_speed, duration)
    robot.wait(0.5)
    return angle
    
def perform_continuous_action(robot, left_power, right_power, duration=1.0):
    """
    Ejecuta una acción compleja: potencia continua en cada rueda.
    - left_power, right_power: valores típicamente en [-100,100] o el rango que admita tu robot.
    - duration: tiempo en segundos que se mantiene ese comando antes de frenar.
    """
    robot.moveWheels(left_power, right_power)
    robot.wait(duration)
    robot.stopMotors()
    
def sample_random_continuous_action(max_power=15):
    """
    Devuelve una tupla (left, right) con potencias muestreadas uniformemente
    en [-max_power, +max_power].
    """
    return (random.randint(-max_power, max_power),
            random.randint(-max_power, max_power))

def avoid_if_needed(robot, threshold=12):
    """
    Lee solo los sensores frontales. Si alguno está por encima del umbral,
    prioriza el que dé mayor valor y gira en consecuencia:
      FrontC  -> 180°
      FrontRR -> 135°
      FrontLL -> 225°
    usando moveWheelsByTime(20, -20, duration).
    Devuelve True si hizo una maniobra, False en caso contrario.
    """
    # Sensores frontales
    front_sensors = [IR.FrontC, IR.FrontLL, IR.FrontRR]
    readings = {}
    for s in front_sensors:
        val = robot.readIRSensor(s) or 0
        readings[s] = val

    # Filtrar los que sobrepasan el umbral
    detected = {s: v for s, v in readings.items() if v > threshold}
    if not detected:
        return False

    # Elegir el sensor con mayor lectura
    sensor = max(detected, key=detected.get)
    val = detected[sensor]

    # Determinar ángulo de giro
    if sensor == IR.FrontC:
        angle = 180
    elif sensor == IR.FrontRR:
        angle = 180 - 45    # 135°
    else:  # IR.FrontLL
        angle = 180 + 45    # 225°

    # Calcular duración proporcional (1.75 s = 180°)
    duration = angle / 180.0 * 1.75

    print(f"  ¡Obstáculo detectado en {sensor.name} ({val})! Girando {angle}° " +
          f"({duration:.2f}s)")

    # Ejecutar giro sobre sí mismo
    robot.moveWheelsByTime(20, -20, duration)
    robot.moveWheelsByTime(20, 20, 1.5)
    robot.wait(0.2)
    return True

def go_back_if_needed(robot, angle, duration, threshold=12):
    """
    Si un sensor detecta un obstáculo tras perform_main_action,
    retrocede el mismo tiempo que avanzó (duration) y gira en sentido
    contrario el mismo tiempo que giró (t_turn).
    Devuelve True si se ejecutó la maniobra de retroceso, False en caso contrario.
    """
    from robobopy.utils.IR import IR

    # Leer solo sensores frontales
    front_sensors = [IR.FrontC, IR.FrontLL, IR.FrontRR]
    readings = {s: robot.readIRSensor(s) or 0 for s in front_sensors}

    # Filtrar sensores sobre umbral
    detected = {s: v for s, v in readings.items() if v > threshold}
    if not detected:
        return False

    # Maniobra de retroceso
    spin_speed = 20
    forward_speed = 20

    # 1) Retroceder mismo tiempo que avanzó
    robot.moveWheelsByTime(-forward_speed, -forward_speed, duration)
    robot.wait(0.1)

    # 2) Girar en sentido contrario
    t_turn = abs(angle) / 180.0 * 1.75
    if angle > 0:
        # para deshacer giro positivo, invertimos signos
        robot.moveWheelsByTime(spin_speed, -spin_speed, t_turn)
    elif angle < 0:
        # para deshacer giro negativo
        robot.moveWheelsByTime(-spin_speed, spin_speed, t_turn)
    # angle == 0: sin giro, no hace nada

    robot.wait(0.2)
    return True