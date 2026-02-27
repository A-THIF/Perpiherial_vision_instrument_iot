import RPi.GPIO as GPIO
import time

# Use BCM numbering
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# --- PIN MAPPING ---
# LED 1 -> Pin 8  (GPIO 14)
# LED 2 -> Pin 10 (GPIO 15)
# LED 3 -> Pin 12 (GPIO 18)
# LED 4 -> Pin 16 (GPIO 23)
# LED 5 -> Pin 18 (GPIO 24)
# LED 6 -> Pin 22 (GPIO 25)
# LED 7 -> Pin 24 (GPIO 8)
# LED 8 -> Pin 26 (GPIO 7)

led_pins = [14, 15, 18, 23, 24, 25, 8, 7]

print("--- MANUAL LED TESTER ---")
print("Type a number (1-8) to turn ON that LED.")
print("Type 'q' to quit.")

# Setup all pins as OUTPUT and OFF
for pin in led_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

try:
    while True:
        # Ask user for input
        choice = input("\nEnter LED Number (1-8): ")
        
        if choice == 'q':
            break
            
        try:
            index = int(choice) - 1 # Convert 1-8 to 0-7 list index
            
            if 0 <= index < 8:
                target_pin = led_pins[index]
                print(f">>> Turning ON LED {choice} (Pin {target_pin}) for 3 seconds...")
                
                GPIO.output(target_pin, GPIO.HIGH)
                time.sleep(3) # Stay on for 3 seconds
                GPIO.output(target_pin, GPIO.LOW)
                
                print(">>> LED OFF.")
            else:
                print("Error: Please enter a number between 1 and 8.")
                
        except ValueError:
            print("Error: Invalid input.")

except KeyboardInterrupt:
    pass

print("\nExiting...")
GPIO.cleanup()
