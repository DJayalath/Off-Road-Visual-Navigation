################################################################################
#
# This work is all the original work of the author and is distributed under the
# GNU General Public License v3.0.
#
# Created By : Dulhan Jayalath
# Date : 2022/04/27
# Project : Real-time Neural Visual Navigation for Autonomous Off-Road Robots
#
################################################################################

# ------------------------------------------------------------------------------
# Moves the rover to follow predicted trajectories using a serial connection
# via the UART port of the rover's Pixhawk 4 autopilot system.
# Recommended videos:
# - https://www.youtube.com/watch?v=paPXfGOhqfo
# - https://www.youtube.com/watch?v=kB9YyG2V-nA
#-------------------------------------------------------------------------------

# Useful links:
# https://ardupilot.org/rover/docs/common-radio-control-calibration.html
# https://ardupilot.org/rover/docs/common-rcmap.html#rover-notes
# https://dronekit-python.readthedocs.io/en/latest/examples/channel_overrides.html
# https://ardupilot.org/rover/docs/rover-steering-input-type-and-reversing-behaviour.html#rover-steering-input-type-and-reversing-behaviour

# Note: Servos are stable (not moving) at 1500 PWM. They range between 1000 (max reverse) and 2000 (max forward).
# Recommend starting slow (e.g. at 1500 +/- 100)

# "Zero" position for servo PWM
SERVO_STABLE = 1500

# The amount to add to PWM for servo movement
SERVO_FW_INC = 100

# Radians to which turns must be accurate
TURN_MARGIN = 0.1

# Seconds per turn increment
TURN_RESOLUTION = 0.04

# Amount to turn for diagonals e.g. NW (depends on mapping of graph to ground plane and perspective)
# FIXME: Calibrate this
DIAG_ANGLE = 0.7

# Durations of forward movements depending on rank
# Movements at higher ranks are at a further perspective so map to longer distances
# FIXME: Calibrate this
FW_DURATIONS = [0.5, 0.8, 0.8, 0.9]

# Angle to turn if starting position is from one of the two central bottom rank nodes
BINARY_START_ANGLE = 0.1

from dronekit import connect, VehicleMode
from pymavlink import mavutil
import time
import math

vehicle = None
north_yaw = 0 # Note: yaw will be in radians

def connect_vehicle():

    global vehicle

    # Connect to vehicle via UART
    vehicle = connect('/dev/ttyAMA0', wait_ready=True, baud=57600)
    print("Connected")


    # --- Set parameters ---

    # Set Servo 1 (left motor?) and 3 (right motor) to disabled (0) state.
    # This enables control by MAVLink servo commands.
    # Note: Flight controller reboot required after change
    vehicle.parameters['SERVO1_FUNCTION'] = 0
    vehicle.parameters['SERVO3_FUNCTION'] = 0

    # Wait for parameter set
    time.sleep(1)
    
    print("Parameters (probably) set")

    # Switch to manual mode in order to accept commands
    while vehicle.mode!='MANUAL':
        print("Current mode", vehicle.mode)
        vehicle.mode = VehicleMode("MANUAL")
        print("Waiting for drone to enter MANUAL flight mode")
        time.sleep(5)
    print("Vehicle now in MANUAL MODE")

    # Arm motors (not necessarily required but good practice)
    print("Arming vehicle")
    while vehicle.armed is False:
        print("Arming...")
        vehicle.armed = True
        time.sleep(1)
    print("Vehicle ARMED")

    # Initial north yaw direction
    north_yaw = vehicle.attitude.yaw
    print(f"Calibrated NORTH YAW = {north_yaw} rad.")

    # Print status
    status()

def status():
    print("Battery", vehicle.battery)
    print("Is armed?", vehicle.armed)

    # EKF says if GPS has a fix. Not required for visual navigation since we override servos.
    print("EKF", vehicle.ekf_ok)

# Set a response for both servos.
def set_servos(left_pwm, right_pwm):

    # Servo instance numbers
    RIGHT = 1
    LEFT = 3

    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target_system, target_component
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO, #command
        0, #confirmation
        LEFT,    # servo number
        left_pwm,          # servo position between 1000 and 2000
        0, 0, 0, 0, 0)    # param 3 ~ 7 not used

    msg2 = vehicle.message_factory.command_long_encode(
        0, 0,    # target_system, target_component
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO, #command
        0, #confirmation
        RIGHT,    # servo number
        right_pwm,          # servo position between 1000 and 2000
        0, 0, 0, 0, 0)    # param 3 ~ 7 not used
    
    # Send command to vehicle
    vehicle.send_mavlink(msg)
    vehicle.send_mavlink(msg2)

    # Flush message buffer to force message send
    vehicle.flush()

def stop_servos():
    set_servos(SERVO_STABLE, SERVO_STABLE)
    print("Servos STOPPED")

# Set servo PWM to 1500us to prep.
# If servos are not prepped, they will not respond to
# other PWM values.
# SERVOx_FUNCTION must be 0 (disabled) to respond to
# these MAVLink commands
def prep_servos():
    stop_servos()
    time.sleep(1)
    print("Servos PREPPED")

# Travel forward for *duration*.
# time.sleep() is unreliable -> use alternative for more accuracy
def go_forward(duration):
    set_servos(SERVO_STABLE + SERVO_FW_INC,
        SERVO_STABLE + SERVO_FW_INC)
    time.sleep(duration)
    stop_servos()
    print("Travelled forward")

# !: Zero turns mean the vehicle should turn almost on the spot.

def zero_turn_right(duration):
    set_servos(SERVO_STABLE + SERVO_FW_INC, SERVO_STABLE - SERVO_FW_INC)
    time.sleep(duration)
    stop_servos()
    print("Zero turned left")

def zero_turn_left(duration):
    set_servos(SERVO_STABLE - SERVO_FW_INC, SERVO_STABLE + SERVO_FW_INC)
    time.sleep(duration)
    stop_servos()
    print("Zero turned right")

def setup():
    connect_vehicle()
    prep_servos()

# Turn to target orientation
def turn_to(target):

    delta = target - vehicle.attitude.yaw # Amount to turn

    while abs(delta) > TURN_MARGIN:

        # Turn
        if delta > 0:
            zero_turn_right(TURN_RESOLUTION)
        else:
            zero_turn_left(TURN_RESOLUTION)

        current = vehicle.attitude.yaw
        print(f"Current yaw: {current}, target: {target}")

        # New delta
        delta = target - current

# Follow a set of headings to destination
def follow_headings(headings):

    north_yaw = vehicle.attitude.yaw

    for rank, hd in enumerate(headings):

        print(f"NORTH: {north_yaw}")

        # Set target heading
        if hd == 'L_START':
            target = north_yaw - BINARY_START_ANGLE
        elif hd == 'R_START':
            target = north_yaw + BINARY_START_ANGLE
        elif hd == 'N':
            target = north_yaw
        elif hd == 'W':
            target = north_yaw - (math.pi / 2.0)
        elif hd == 'E':
            target = north_yaw + (math.pi / 2.0)
        elif hd == 'NW':
            target = north_yaw - DIAG_ANGLE
        elif hd == 'NE':
            target = north_yaw + DIAG_ANGLE

        # Value should be between -pi and +pi

        # Correct for negative orientation
        if target < -math.pi:
            target += 2 * math.pi

        # Correct for orientation > 2pi
        if target >= math.pi:
            target -= 2 * math.pi

        print(f"Correcting heading -- CURRENT = {vehicle.attitude.yaw}, TARGET = {target}")
        turn_to(target)
        
        # Go in new direction
        go_forward(FW_DURATIONS[rank])