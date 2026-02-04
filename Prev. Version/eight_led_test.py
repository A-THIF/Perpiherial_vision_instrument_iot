import RPi.GPIO as GPIO
import time

# === BCM Mode ===
GPIO.setmode(GPIO.BCM)

# === LED GPIO Pins for 8 Channels ===
led_pins = [17, 18, 27, 5, 16, 13, 19, 26]

# === Setup all pins as outputs ===
GPIO.setup(led_pins, GPIO.OUT, initial=GPIO.LOW)

print("8-LED Test: Each LED will turn ON for 5 seconds in sequence.")
print("Press Ctrl+C to stop.")

try:
    while True:
        for i, pin in enumerate(led_pins, start=1):
            print(f"\nChannel {i} - LED on GPIO{pin} -> ON")
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(5)  # keep LED ON for 5 sec
            GPIO.output(pin, GPIO.LOW)
            print(f"Channel {i} - LED OFF")
            time.sleep(0.5)  # short gap before next LED

except KeyboardInterrupt:
    print("\nTest stopped by user.")

finally:
    GPIO.cleanup()
    print("GPIO cleaned up. All LEDs OFF.")
