import RPi.GPIO as GPIO
import time

# --- HELPER FUNCTION (Must be at the top!) ---
def pin_to_physical_map(bcm):
    mapping = {14:8, 15:10, 18:12, 23:16, 24:18, 25:22, 8:24, 7:26}
    return mapping.get(bcm, "?")

# Use BCM numbering
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# --- PIN MAPPING ---
led_pins = [14, 15, 18, 23, 24, 25, 8, 7]

# Setup all pins as OUTPUT and OFF
for pin in led_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

print("--- CUSTOM LED TESTER ---")
print("Options:")
print("  'q' = Quit program")
print("  'd' = Change duration")

try:
    while True:
        # --- OUTER LOOP: ASK FOR DURATION ---
        print("\n-----------------------------")
        dur_input = input("Step 1: Enter Duration (seconds): ")
        
        if dur_input.lower() == 'q': 
            break
            
        try:
            duration = float(dur_input)
            print(f"--> Duration set to {duration} seconds.")
            
            # --- INNER LOOP: ASK FOR LED ---
            while True:
                led_input = input(f"\n[Duration: {duration}s] Enter LED (1-8) or 'd' to change: ")
                
                if led_input.lower() == 'q':
                    raise KeyboardInterrupt # Quick way to exit everything
                
                if led_input.lower() == 'd':
                    break # Break inner loop, go back to ask duration
                
                try:
                    index = int(led_input) - 1
                    if 0 <= index < 8:
                        target_pin = led_pins[index]
                        phys_pin = pin_to_physical_map(target_pin)
                        
                        print(f">>> Turning ON LED {led_input} (Pin {phys_pin})...")
                        GPIO.output(target_pin, GPIO.HIGH)
                        
                        time.sleep(duration)
                        
                        GPIO.output(target_pin, GPIO.LOW)
                        print(">>> LED OFF")
                    else:
                        print("Error: Number must be 1-8")
                except ValueError:
                    print("Error: Invalid number.")
                    
        except ValueError:
            print("Error: Please enter a number for duration (e.g. 0.5)")

except KeyboardInterrupt:
    pass

print("\nExiting...")
GPIO.cleanup()
