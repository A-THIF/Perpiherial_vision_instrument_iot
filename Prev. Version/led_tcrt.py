import RPi.GPIO as GPIO
import time

# === BCM Pin Mode ===
GPIO.setmode(GPIO.BCM)

# === Pin Setup ===
LED_PIN = 17          # LED control pin
SENSOR_POWER = 27     # TCRT V+ control pin
SENSOR_SIGNAL = 22    # TCRT signal input pin

GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(SENSOR_POWER, GPIO.OUT)
# Enable pull-up so input is stable when idle (adjust if logic is inverted)
GPIO.setup(SENSOR_SIGNAL, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("LED/TCRT test started")
print("Press Ctrl+C to exit")

try:
    while True:
        # Turn ON LED & Sensor
        GPIO.output(LED_PIN, GPIO.HIGH)
        GPIO.output(SENSOR_POWER, GPIO.HIGH)
        
        print("\nLED and sensor ON - Wave your hand in front of the sensor...")
        start_time = time.time()
        
        while time.time() - start_time < 5:
            sensor_value = GPIO.input(SENSOR_SIGNAL)
            if sensor_value == 0:  # Active LOW output typical for TCRT modules
                print("Object detected!")
            else:
                print("No object")
            time.sleep(0.2)

        # Turn OFF LED & Sensor
        GPIO.output(LED_PIN, GPIO.LOW)
        GPIO.output(SENSOR_POWER, GPIO.LOW)
        print("LED and sensor OFF")
        
        # Off period
        time.sleep(5)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    GPIO.cleanup()
    print("GPIO cleaned up.")
