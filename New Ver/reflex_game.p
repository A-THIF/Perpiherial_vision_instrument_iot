import smbus
import time
import random

bus = smbus.SMBus(1)
MCP = 0x20  # Make sure i2cdetect shows 20 (not 21)

# Register Addresses
IODIRA = 0x00 # Input (Sensors)
IODIRB = 0x01 # Output (LEDs)
GPIOA  = 0x12 # Read Sensors
GPIOB  = 0x13 # Write LEDs
GPPUA  = 0x0C # Pull-up Resistors

# --- CONFIGURATION ---
# We are using Sensor 1 (A1) and LED 1 (B1) based on your wiring.
# If you want A0, change SENSOR_BIT to 1 and LED_BIT to 1.
SENSOR_BIT = 2  # Value 2 = Pin A1
LED_BIT    = 2  # Value 2 = Pin B1

def setup_chip():
    try:
        bus.write_byte_data(MCP, IODIRA, 0xFF) # Port A = Inputs
        bus.write_byte_data(MCP, IODIRB, 0x00) # Port B = Outputs
        bus.write_byte_data(MCP, GPPUA, 0xFF)  # Enable Pull-ups
        bus.write_byte_data(MCP, GPIOB, 0)     # Turn off all LEDs to start
        return True
    except OSError:
        print("ERROR: Chip disconnected! Fix the Red/Black wires.")
        return False

# --- MAIN GAME LOOP ---
if setup_chip():
    print("--- REFLEX GAME STARTED ---")
    print("Wait for the LED... Put your hand in when it lights up!")

    while True:
        try:
            # 1. Wait for a random time (between 2 and 5 seconds)
            delay = random.uniform(2, 5)
            print(f"\nWaiting {delay:.1f} seconds...")
            time.sleep(delay)

            # 2. TURN ON LED
            print("GO! LED ON!")
            bus.write_byte_data(MCP, GPIOB, LED_BIT)
            
            # 3. Start the 5-Second Timer
            start_time = time.time()
            detected = False
            
            while (time.time() - start_time) < 5.0:
                # Read Sensors
                inputs = bus.read_byte_data(MCP, GPIOA)
                
                # Check if Sensor is BLOCKED (Value becomes 0)
                if (inputs & SENSOR_BIT) == 0:
                    print(">>> SUCCESS! Object Detected! <<<")
                    detected = True
                    break # Exit the timer loop immediately
                
                time.sleep(0.05) # Small delay to save CPU

            # 4. End of Round
            bus.write_byte_data(MCP, GPIOB, 0) # Turn LED OFF
            
            if not detected:
                print("--- FAILED: Time's Up! ---")
            
            time.sleep(1) # Short pause before next round

        except OSError:
            print("CRITICAL: Wire loose. Reconnecting...")
            setup_chip()
            time.sleep(1)
