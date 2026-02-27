import smbus2 as smbus
import RPi.GPIO as GPIO
import time
import sys

# Combine Right Side (Even physical) and Left Side (Odd physical)
LED_PINS = [14, 15, 18, 23, 24, 25, 8, 7, 4, 17, 27, 22, 5, 6, 13, 19]

MCP_1 = 0x20 # Right side sensors (0-7)
MCP_2 = 0x21 # Left side sensors (8-15)

print("Initializing GPIO...")
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

print("Initializing I2C Bus...")
try:
    bus = smbus.SMBus(1)
    
    # Setup MCP 1 (0x20)
    bus.write_byte_data(MCP_1, 0x00, 0xFF) # All Inputs
    bus.write_byte_data(MCP_1, 0x0C, 0xFF) # Enable Pull-ups
    print("✅ MCP23017 (0x20) Connected!")
    
    # Setup MCP 2 (0x21)
    bus.write_byte_data(MCP_2, 0x00, 0xFF) # All Inputs
    bus.write_byte_data(MCP_2, 0x0C, 0xFF) # Enable Pull-ups
    print("✅ MCP23017 (0x21) Connected!")
    
except Exception as e:
    print(f"❌ HARDWARE ERROR: Could not find MCP chips. ({e})")
    sys.exit(1)

print("\n==================================================")
print(" DIAGNOSTIC MODE ACTIVE")
print(" Touch any sensor to light up its matching LED.")
print(" Press CTRL+C to quit.")
print("==================================================\n")

try:
    while True:
        # Read from 0x20 (Invert logic because Active LOW)
        val_1 = (~bus.read_byte_data(MCP_1, 0x12)) & 0xFF
        
        # Read from 0x21
        val_2 = (~bus.read_byte_data(MCP_2, 0x12)) & 0xFF
        
        # Combine into a single 16-bit number
        all_sensors = (val_2 << 8) | val_1

        # Check every bit from 0 to 15
        for i in range(16):
            if all_sensors & (1 << i):
                GPIO.output(LED_PINS[i], GPIO.HIGH) # Turn LED ON
            else:
                GPIO.output(LED_PINS[i], GPIO.LOW)  # Turn LED OFF

        # Print to terminal if something is being touched
        if all_sensors != 0:
            # zfill(16) ensures it always prints exactly 16 zeroes/ones
            print(f"Sensors touched: {bin(all_sensors)[2:].zfill(16)}")

        # Small delay to prevent CPU overload
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nCleaning up LEDs...")
    for pin in LED_PINS:
        GPIO.output(pin, GPIO.LOW)
    print("Test Ended.")
    sys.exit(0)