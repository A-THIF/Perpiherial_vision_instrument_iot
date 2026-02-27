import smbus
import time

bus = smbus.SMBus(1)
MCP = 0x20  # Your Chip Address

# Registers
IODIRB = 0x01 # Port B Direction
GPIOB  = 0x13 # Port B Value (LEDs)

# --- CONFIGURATION ---
# We want B0, B1, B2, B3, B4 to turn ON.
# That is value 31 (Binary 00011111)
ALL_ON = 31 

def setup():
    try:
        # Set Port B to Output (0 = Output)
        bus.write_byte_data(MCP, IODIRB, 0x00)
        # Start with them OFF
        bus.write_byte_data(MCP, GPIOB, 0)
        return True
    except OSError:
        print("ERROR: Chip not found. Check Red Power Wire!")
        return False

# --- BLINK LOOP ---
if setup():
    print("--- BLINKING 5 LEDS (B0-B4) ---")
    print("Press CTRL+C to stop")

    while True:
        try:
            # Turn ON (Send 31)
            bus.write_byte_data(MCP, GPIOB, ALL_ON)
            print("ON")
            time.sleep(0.5)

            # Turn OFF (Send 0)
            bus.write_byte_data(MCP, GPIOB, 0)
            print("OFF")
            time.sleep(0.5)

        except OSError:
            print("Connection flicker... retrying")
