import smbus
import time

bus = smbus.SMBus(1)
MCP_ADDRESS = 0x20

# Registers
IODIRA = 0x00
GPPUA  = 0x0C
GPIOA  = 0x12

print("--- FULL RACK MONITOR (A1 - A7) ---")
print("Waiting for Connection...")

# Setup Loop
setup_done = False
while not setup_done:
    try:
        bus.write_byte_data(MCP_ADDRESS, IODIRA, 0xFF) # All Inputs
        bus.write_byte_data(MCP_ADDRESS, GPPUA, 0xFF)  # Pull-Ups
        print(">>> SUCCESS: Chip Active!")
        setup_done = True
    except OSError:
        time.sleep(1)

while True:
    try:
        # Read all 8 pins
        data = bus.read_byte_data(MCP_ADDRESS, GPIOA)
        
        # Build the status line
        # We start loop from 1 to 7 (Skipping A0)
        output = "Rack A: |"
        for pin in range(1, 8):
            
            # Check if THIS specific pin is triggered
            # We shift '1' to the left by 'pin' amount
            # e.g., for Pin 3, we check binary 00001000
            if (data & (1 << pin)) == 0:
                output += f" A{pin}: PHONE |"
            else:
                output += f" A{pin}:  .    |"
        
        print(output)
        time.sleep(0.2)
        
    except OSError:
        print("!!! WIRE LOOSE - Wiggle SDA/SCL !!!")
        time.sleep(0.5)
