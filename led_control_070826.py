#simulate natural irradiance
#MJR 6/22/26

import pandas as pd
import numpy as np

#these are needed for RS232 communication
import serial
import time

#pvlib is a library for calculation of irradiance parameters
import pvlib
from pvlib import clearsky, atmosphere, solarposition
from pvlib.location import Location
from pvlib.iotools import read_tmy3

###############
#if you want the code to run through the program in an accelerated fashion increase this factor
accel_factor = 1

###############
#settings for RS232 communication through USB
#this is what the port was created by on my macbook, will need to be changed for the raspberry pi
PORT = '/dev/tty.usbserial-A9AQD7P5'

prefix = 'S1'  #which fluorostat to control

ser = serial.Serial(PORT, baudrate=19200, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=1)

# Allow time for the connection to initialize
time.sleep(2)

# this function converts W / m^2 into controller units, for now
# 1 W / m^2 = 1 unit, capped at 1000
def scale_to_controller(I):
    return min(I,1000)   #prevent intensity from exceeding 1000 in controller units

#time step
deltaT = 10 #min

#latitude and longitude for Chicago, but could pick somewhere else 
site = Location(41.9,-87.6)  

#generate times for the month of july starting 8 AM sampled every 10 minutes (arbitrary choice)
times = pd.date_range(start='2025-07-01 08:00:00', end='2025-08-01 08:00:00', freq='10min', tz=site.tz)

cs = site.get_clearsky(times)
ghi = cs['ghi']

tlist = []
cslist = []
t=0
for i in range(len(cs)):
    tlist.append(t)
    cslist.append(scale_to_controller(ghi[i]))
    t += deltaT*60  //t is in seconds

#main loop for waveform control
start = datetime.datetime.now()
done = False
index = 0
while not done:
    current = datetime.datetime.now()
    elapsed = (current - start).total_seconds()
    update = False
    while elapsed > tlist[index] / accel_factor:
        index += 1
        update = True
        if index >= len(tlist):
            done = True
            break
    if update:
        value = cslist[index]
        string = prefix + f"{int(value):04d}"
        ser.write(string.encode())
    #don't work too hard
    time.sleep(10)
