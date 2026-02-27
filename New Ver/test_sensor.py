import smbus
import time

bus = smbus.SMBus(1)
MCP_ADDRESS = 0x20

# Registers
IODIRA = 0x00
GPPUA  = 0x0C
GPIOA  = 0x12

print("--- TESTING PIN A1 ---")

# Setup (Same as before)
bus.write_byte_data(MCP_ADDRESS, IODIRA, 0xFF) # All Inputs
bus.write_byte_data(MCP_ADDRESS, GPPUA, 0xFF)  # Pull-Ups

try:
    while True:
        # Read the whole Port A (8 pins at once)
        data = bus.read_byte_data(MCP_ADDRESS, GPIOA)
        
        # --- THIS IS THE CHANGE ---
        # We use '& 2' to check the SECOND pin (A1)
        # (A0 would be '& 1')
        pin_state = data & 2 
        
        if pin_state == 0:
            print(">>> OBJECT DETECTED! (Pin A1)")
        else:
            print("... Path Clear")
            
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopped.")
except OSError:
    print("Error: Check Wires")
