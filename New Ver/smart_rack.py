import smbus
import time

bus = smbus.SMBus(1)
MCP = 0x20  # This targets the "20" we saw on screen

# Registers
IODIRA = 0x00 # Port A -> Inputs
IODIRB = 0x01 # Port B -> Outputs
GPIOA  = 0x12 # Read Inputs
GPIOB  = 0x13 # Write Outputs
GPPUA  = 0x0C # Pull-ups

print("--- SMART RACK ACTIVE ---")
print("Sensor A1 controls LED B1")

# Setup
try:
    bus.write_byte_data(MCP, IODIRA, 0xFF) # Port A is INPUT
    bus.write_byte_data(MCP, IODIRB, 0x00) # Port B is OUTPUT
    bus.write_byte_data(MCP, GPPUA, 0xFF)  # Pull-ups on
except OSError:
    print("Error: Chip 20 not found. Check wires.")

while True:
    try:
        # Read Sensors
        inputs = bus.read_byte_data(MCP, GPIOA)
        
        # Logic: If Sensor A1 (Bit 1, Value 2) is Triggered (Low/0)
        if (inputs & 2) == 0:
            # Turn ON LED B1 (Bit 1, Value 2)
            bus.write_byte_data(MCP, GPIOB, 2)
            print("PHONE DETECTED! [LED ON]")
        else:
            # Turn OFF LED
            bus.write_byte_data(MCP, GPIOB, 0)
            print("Empty... [LED OFF]")

        time.sleep(0.1)
        
    except OSError:
        print("Connection unstable...")
        time.sleep(0.5)
