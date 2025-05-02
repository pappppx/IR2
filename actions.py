import random
from robobopy.utils.IR import IR
import random


# ————— Parámetros —————
SAMPLE_DT       = 1.0      # segundos entre muestras
AVOID_THRESHOLD = 200      # valor IR a partir del cual hay obstáculo
AVOID_SPEED     = 20       # velocidad de retroceso y giro
AVOID_TIME      = 0.3      # tiempo de cada maniobra de evitación

IR_SENSORS = [
    IR.BackC,
    IR.BackL,
    IR.BackR,
    IR.FrontC,
    IR.FrontL,
    IR.FrontLL,
    IR.FrontR,
    IR.FrontRR
]

AVOID_ACTIONS = {
    IR.FrontC:  (-AVOID_SPEED,       -AVOID_SPEED),
    IR.FrontL:  (-AVOID_SPEED//2,    -AVOID_SPEED),   # back + turn right
    IR.FrontLL: (-AVOID_SPEED//2,    -AVOID_SPEED),   # idem FrontL
    IR.FrontR:  (-AVOID_SPEED,       -AVOID_SPEED//2),# back + turn left
    IR.FrontRR: (-AVOID_SPEED,       -AVOID_SPEED//2),# idem FrontR

    IR.BackC:   (AVOID_SPEED,        AVOID_SPEED),
    IR.BackL:   (AVOID_SPEED//2,     AVOID_SPEED),    # forward + turn right
    IR.BackR:   (AVOID_SPEED,        AVOID_SPEED//2), # forward + turn left
}

DISCRETE_ACTIONS = {
    -90:{'l':10,'r':-10},
    -45:{'l':8,'r':4},
    0:{'l':10,'r':10},
    45:{'l':4,'r':8},
    90:{'l':-10,'r':10}
    }

def perform_simple_action(robot, angulo, duration=2.0):
    # tu diccionario ACCIONES de antes
    v = DISCRETE_ACTIONS[angulo]
    robot.moveWheels(v['l'], v['r'])
    robot.wait(duration)
    robot.stopMotors()
    
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

def avoid_if_needed(robot, threshold=AVOID_THRESHOLD):
    """
    Lee todos los sensores IR. Si alguno detecta obstáculo (valor > threshold),
    ejecuta la maniobra mapeada en AVOID_ACTIONS y sale.
    """
    for s in IR_SENSORS:
        val = robot.readIRSensor(s)
        if val and val > threshold:
            print(f"  ¡Obstáculo detectado en {s.name} ({val})! Maniobra de evitación:")
            # obtener acción según sensor; por defecto retrocede
            left_spd, right_spd = AVOID_ACTIONS.get(
                s,
                (-AVOID_SPEED, -AVOID_SPEED)  # fallback: retroceder recto
            )
            # ejecutar maniobra
            robot.stopMotors()
            robot.moveWheels(left_spd, right_spd)
            robot.wait(AVOID_TIME)
            robot.stopMotors()
            break