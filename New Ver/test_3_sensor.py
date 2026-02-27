import smbus
import time

bus = smbus.SMBus(1)
MCP_ADDRESS = 0x20

# Registers
IODIRA = 0x00
GPPUA  = 0x0C
GPIOA  = 0x12

print("--- 3 SENSOR MONITOR (A1, A2, A3) ---")
print("Waiting for I2C Connection...")

# --- SETUP LOOP (Keeps trying until connected) ---
connected = False
while not connected:
    try:
        bus.write_byte_data(MCP_ADDRESS, IODIRA, 0xFF) # Set as Inputs
        bus.write_byte_data(MCP_ADDRESS, GPPUA, 0xFF)  # Turn on Pull-Ups
        print(">>> SUCCESS: MCP23017 Connected!")
        connected = True
    except OSError:
        time.sleep(1)

# --- MAIN LOOP ---
while True:
    try:
        # Read the whole Port A
        data = bus.read_byte_data(MCP_ADDRESS, GPIOA)
        
        # Check specific pins using Bitwise Logic
        # A1 is Bit 1 (Value 2)
        # A2 is Bit 2 (Value 4)
        # A3 is Bit 3 (Value 8)
        
        s1 = "DETECTED" if (data & 2) == 0 else "......"
        s2 = "DETECTED" if (data & 4) == 0 else "......"
        s3 = "DETECTED" if (data & 8) == 0 else "......"
        
        # Print status on one line (updates constantly)
        print(f"A1: {s1}  |  A2: {s2}  |  A3: {s3}")
        
        time.sleep(0.2)
        
    except OSError:
        print("!!! CONNECTION LOST - CHECK WIRES !!!")
        time.sleep(0.5)
