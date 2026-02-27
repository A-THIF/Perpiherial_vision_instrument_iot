import smbus
import time
import random

bus = smbus.SMBus(1)
MCP = 0x20  # We are using the first chip

# --- SETTINGS (CHANGE THESE IF NEEDED) ---
# Sensor is on Port A.
# Value 1 = Pin A0 (Chip Pin 21)
# Value 2 = Pin A1 (Chip Pin 22)
SENSOR_VAL = 1  

# LED is on Port B.
# Value 1 = Pin B0 (Chip Pin 1)
# Value 2 = Pin B1 (Chip Pin 2)
LED_VAL = 2     

# --- REGISTERS ---
IODIRA = 0x00 # Input Direction (Port A)
IODIRB = 0x01 # Output Direction (Port B)
GPIOA  = 0x12 # Read Inputs
GPIOB  = 0x13 # Write Outputs
GPPUA  = 0x0C # Pull-up Resistors

def setup():
    try:
        # 1. Set Port A as Input (Sensors)
        bus.write_byte_data(MCP, IODIRA, 0xFF)
        # 2. Set Port B as Output (LEDs)
        bus.write_byte_data(MCP, IODIRB, 0x00)
        # 3. Enable Pull-up Resistors on Port A
        bus.write_byte_data(MCP, GPPUA, 0xFF)
        # 4. Turn OFF all LEDs initially
        bus.write_byte_data(MCP, GPIOB, 0x00)
        return True
    except OSError:
        print("ERROR: Chip 0x20 not found! Check Red/Black wires.")
        return False

# --- GAME LOOP ---
if setup():
    print("--- REFLEX GAME READY ---")
    print(f"Watch LED (Value {LED_VAL})... Trigger Sensor (Value {SENSOR_VAL})")

    while True:
        try:
            # STEP 1: Wait for a random time (2 to 6 seconds)
            wait_time = random.uniform(2, 6)
            print(f"\nWaiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)

            # STEP 2: TURN LED ON
            print(">>> GO! LED ON! <<<")
            bus.write_byte_data(MCP, GPIOB, LED_VAL)

            # STEP 3: Start 5-Second Timer
            start_time = time.time()
            success = False

            while (time.time() - start_time) < 5.0:
                # Read Sensors
                inputs = bus.read_byte_data(MCP, GPIOA)

                # Check if Sensor is TRIGGERED (Logic 0)
                if (inputs & SENSOR_VAL) == 0:
                    print("SUCCESS: Object Detected!")
                    success = True
                    break  # Stop the timer immediately
                
                # Small delay to keep CPU cool
                time.sleep(0.01)

            # STEP 4: Turn LED OFF
            bus.write_byte_data(MCP, GPIOB, 0)

            # Result Message
            if not success:
                print("FAILED: Too slow (Time's Up)")

            time.sleep(2) # Pause before next round

        except OSError:
            print("Connection Lost... Retrying...")
            setup()
            time.sleep(1)
