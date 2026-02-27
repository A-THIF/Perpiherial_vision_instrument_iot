import smbus
import time

bus = smbus.SMBus(1)
MCP = 0x20  # The address you saw in i2cdetect

# Registers
IODIRB = 0x01 # Output Direction (Port B)
GPIOB  = 0x13 # Write Outputs (Port B)

print("--- LED TEST START ---")
print("Trying to turn ON ALL Port B pins...")

try:
    # 1. Set all Port B pins to OUTPUT (0 = Output)
    bus.write_byte_data(MCP, IODIRB, 0x00)
    
    # 2. Turn ALL Port B pins HIGH (255 = 11111111 in binary)
    bus.write_byte_data(MCP, GPIOB, 0xFF) 
    print("COMMAND SENT: All LEDs should be ON now.")
    
    # Keep them on for 10 seconds so you can check
    time.sleep(10)
    
    # 3. Turn them OFF
    bus.write_byte_data(MCP, GPIOB, 0x00)
    print("Test Complete.")

except OSError:
    print("ERROR: Chip 20 not reachable. Check that Red Power Wire!")
