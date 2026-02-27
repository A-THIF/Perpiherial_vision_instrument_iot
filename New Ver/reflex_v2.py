import smbus
import time
import random

bus = smbus.SMBus(1)
MCP = 0x20  # Targeting your first chip

# Register Addresses
IODIRA = 0x00 # Input Direction (Port A)
IODIRB = 0x01 # Output Direction (Port B)
GPIOA  = 0x12 # Read Inputs (Port A)
GPIOB  = 0x13 # Write Outputs (Port B)
GPPUA  = 0x0C # Pull-up Resistors (Port A)

# --- CONFIGURATION (UPDATED FOR A0 and B0) ---
SENSOR_BIT = 1  # Value 1 selects Pin A0
LED_BIT    = 1  # Value 1 selects Pin B0

def setup_chip():
    try:
        # 1. Set Port A to INPUT (Sensors)
        bus.write_byte_data(MCP, IODIRA, 0xFF)
        # 2. Set Port B to OUTPUT (LEDs)
        bus.write_byte_data(MCP, IODIRB, 0x00)
        # 3. Enable Pull-up Resistors for Sensors
        bus.write_byte_data(MCP, GPPUA, 0xFF)
        # 4. Turn off LED to start
        bus.write_byte_data(MCP, GPIOB, 0)
        return True
    except OSError:
        print("ERROR: Chip 0x20 not found. Check Red Power Wire!")
        return False

# --- GAME START ---
if setup_chip():
    print("--- REFLEX GAME (A0 & B0) ---")
    print("1. Wait for LED B0 to light up.")
    print("2. Quickly trigger Sensor A0!")
    print("-----------------------------")

    while True:
        try:
            # 1. RANDOM DELAY (Wait 2 to 5 seconds)
            delay = random.uniform(2, 5)
            print(f"\nWaiting {delay:.1f} seconds...")
            time.sleep(delay)

            # 2. TURN ON LED B0
            print(">>> GO! LED ON! <<<")
            bus.write_byte_data(MCP, GPIOB, LED_BIT)
            
            # 3. START TIMER (You have 3 seconds to react)
            start_time = time.time()
            success = False
            
            while (time.time() - start_time) < 3.0:
                # Read all sensors
                inputs = bus.read_byte_data(MCP, GPIOA)
                
                # Check if A0 is triggered (Value becomes 0)
                if (inputs & SENSOR_BIT) == 0:
                    print("SUCCESS! Reflex Time: {:.3f}s".format(time.time() - start_time))
                    success = True
                    break # Exit loop
                
                # Tiny delay to prevent CPU overload
                time.sleep(0.01)

            # 4. TURN OFF LED B0
            bus.write_byte_data(MCP, GPIOB, 0)
            
            if not success:
                print("TOO SLOW! (Time's Up)")
            
            # Pause before next round
            time.sleep(2)

        except OSError:
            print("CRITICAL: Connection Lost. Retrying...")
            setup_chip()
            time.sleep(1)
