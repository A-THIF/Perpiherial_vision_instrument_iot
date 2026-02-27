import RPi.GPIO as GPIO
import time

# Use BCM numbering
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# --- PIN MAPPING ---
# Physical Pin -> BCM Number
# Pin 8  -> GPIO 14
# Pin 10 -> GPIO 15
# Pin 12 -> GPIO 18
# Pin 16 -> GPIO 23
# Pin 18 -> GPIO 24
# Pin 22 -> GPIO 25
# Pin 24 -> GPIO 8
# Pin 26 -> GPIO 7

# The list of LED pins
led_pins = [14, 15, 18, 23, 24, 25, 8, 7]

# --- HELPER FUNCTION (Moved to Top) ---
def pin_to_physical(bcm):
    mapping = {14:8, 15:10, 18:12, 23:16, 24:18, 25:22, 8:24, 7:26}
    return mapping.get(bcm, "?")

print("--- LED SEQUENCE TEST ---")
print("Blinking LEDs one by one...")
print("Press CTRL+C to stop")

# 1. Setup all pins as OUTPUT and turn them OFF immediately
for pin in led_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

try:
    while True:
        # Loop through each LED in the list
        for i, pin in enumerate(led_pins):
            print(f"Blinking LED {i+1} (Physical Pin {pin_to_physical(pin)})")
            
            # Turn ON
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(0.5) # ON for 0.5s
            
            # Turn OFF
            GPIO.output(pin, GPIO.LOW)
            time.sleep(0.1) # OFF for 0.1s

except KeyboardInterrupt:
    print("\nStopping... Turning off all LEDs.")
    GPIO.cleanup()
