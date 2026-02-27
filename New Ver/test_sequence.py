import smbus
import time

bus = smbus.SMBus(1)
MCP = 0x20  # Address 20

# Registers
IODIRB = 0x01 # Port B Direction
GPIOB  = 0x13 # Port B Output

# Setup
try:
    # Set Port B to Output (0)
    bus.write_byte_data(MCP, IODIRB, 0x00)
    # Turn ALL OFF to start
    bus.write_byte_data(MCP, GPIOB, 0)
    print("--- SAFETY CHECK PASSED ---")
except OSError:
    print("ERROR: Chip 20 not found. Check wires.")
    exit()

print("Testing LEDs B0 to B4 one by one...")
print("If the screen freezes, UNPLUG IMMEDIATELY.")

while True:
    for i in range(0, 5): # Loop from 0 to 4
        
        # Calculate the binary value for just this ONE pin
        # 0 = 1, 1 = 2, 2 = 4, 3 = 8, 4 = 16
        pin_value = 2 ** i 
        
        print(f"Testing LED B{i} (Value {pin_value})...")
        
        try:
            # Turn ON just this one LED
            bus.write_byte_data(MCP, GPIOB, pin_value)
            time.sleep(0.5) # Keep it on for 0.5 seconds
            
            # Turn OFF
            bus.write_byte_data(MCP, GPIOB, 0)
            time.sleep(0.2) # Short pause
            
        except OSError:
            print(f"CRITICAL ERROR at LED B{i}! Wiring short detected!")
            break
            
    print("--- Sequence Complete. Restarting... ---")
    time.sleep(1)
