import random
from robobopy.utils.IR import IR


# ————— Parámetros —————
SAMPLE_DT       = 1.0      # segundos entre muestras
AVOID_THRESHOLD = 15      # valor IR a partir del cual hay obstáculo

from perceptions import get_simple_perceptions
import numpy as np

def perform_main_action(robot, sim, angle, duration=1.0):
    """
    Ejecuta SOLO el giro + avance (tu acción cognitiva) y devuelve
    el estado medido justo después de esa maniobra, antes de cualquier evitado.
    """
    spin_speed = 20
    forward_speed = 20
    # 1) Girar proporcional a angle
    t_turn = abs(angle) / 180.0 * 1.75
    if angle > 0:
        robot.moveWheelsByTime(-spin_speed, spin_speed, t_turn)
    elif angle < 0:
        robot.moveWheelsByTime(spin_speed, -spin_speed, t_turn)
    # 2) Avanzar recto
    robot.moveWheelsByTime(forward_speed, forward_speed, duration)
    robot.wait(0.1)
    # 3) Leer percepción justo tras la maniobra principal
    P = get_simple_perceptions(sim)
    S_main = np.array([
        P['red_rotation'],  P['red_position'],
        P['green_rotation'],P['green_position'],
        P['blue_rotation'], P['blue_position']
    ], dtype=np.float32)
    # 4) Ejecutar evitado si es necesario, pero sin volver a leer
    if avoid_if_needed(robot):
        robot.wait(0.1)
    # 5) Devolver estado tras la acción cognitiva
    return S_main




def perform_simple_action(robot, angle, duration=1.0):
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
