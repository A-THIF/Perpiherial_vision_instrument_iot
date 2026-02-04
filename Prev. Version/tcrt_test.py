import RPi.GPIO as GPIO
import time

SENSOR_PIN = 21  # GPIO17 (Pin 11)

GPIO.setmode(GPIO.BCM)
GPIO.setup(SENSOR_PIN, GPIO.IN)

print("TCRT5000 Test Started (Press Ctrl+C to stop)")

try:
    while True:
        if GPIO.input(SENSOR_PIN) == GPIO.LOW:
            print("Object Detected!")
        else:
            print("No Object")
        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nExiting...")
    GPIO.cleanup()
