import smbus
import time
import random

bus = smbus.SMBus(1)
MCP = 0x20  # Address 20

# Registers
IODIRA = 0x00 # Port A Direction (Inputs)
IODIRB = 0x01 # Port B Direction (Outputs)
GPIOA  = 0x12 # Read Sensors
GPIOB  = 0x13 # Write LEDs
GPPUA  = 0x0C # Pull-up Resistors

# --- TARGET: UNIT 0 (A0 and B0) ---
# In binary, Pin 0 is the first bit (Value 1)
TARGET_BIT = 1 

def setup_game():
    try:
        # Reset Chip
        bus.write_byte_data(MCP, IODIRA, 0xFF) # Port A Input
        bus.write_byte_data(MCP, IODIRB, 0x00) # Port B Output
        bus.write_byte_data(MCP, GPPUA, 0xFF)  # Pull-ups
        bus.write_byte_data(MCP, GPIOB, 0)     # Turn OFF all LEDs
        return True
    except OSError:
        print("Error: Chip not connected.")
        return False

# --- GAME LOOP ---
if setup_game():
    print("--- REFLEX GAME: UNIT 0 ---")
    print("1. Watch LED B0.")
    print("2. When it lights up, block Sensor A0!")
    print("-----------------------------")
    time.sleep(1)

    while True:
        try:
            # 1. Random Wait (2 to 5 seconds)
            wait_time = random.uniform(2, 5)
            print(f"\nWaiting {wait_time:.1f}s...")
            time.sleep(wait_time)

            # 2. LED ON
            print(">>> GO! <<<")
            bus.write_byte_data(MCP, GPIOB, TARGET_BIT)
            
            # 3. Timer Start
            start = time.time()
            won = False
            
            while (time.time() - start) < 3.0: # 3 Second Limit
                # Read Sensors
                inputs = bus.read_byte_data(MCP, GPIOA)
                
                # Check A0 (Bit 1). If it is 0, object detected.
                if (inputs & TARGET_BIT) == 0:
                    reaction = time.time() - start
                    print(f"WINNER! Time: {reaction:.3f}s")
                    won = True
                    break
                
            # 4. Cleanup
            bus.write_byte_data(MCP, GPIOB, 0) # LED OFF
            
            if not won:
                print("TOO SLOW!")

            time.sleep(2) # Pause before next round

        except OSError:
            setup_game()
