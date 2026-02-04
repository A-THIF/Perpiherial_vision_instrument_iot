import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)

print("Turning LED ON for 5 seconds")
GPIO.output(17, GPIO.HIGH)
time.sleep(5)

print("Turning LED OFF")
GPIO.output(17, GPIO.LOW)

GPIO.cleanup()
