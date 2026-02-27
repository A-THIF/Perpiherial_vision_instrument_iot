import smbus
import time

bus = smbus.SMBus(1)
MCP = 0x20  # Your Chip Address

# Registers
IODIRB = 0x01 # Direction Port B
GPIOB  = 0x13 # Output Port B

print("--- TESTING 3 LEDs (B0, B1, B2) ---")

try:
    # 1. Set Port B to OUTPUT
    bus.write_byte_data(MCP, IODIRB, 0x00) 

    while True:
        # 2. Turn ON (Send value 7 -> Binary 00000111)
        # This turns on Bit 0, Bit 1, and Bit 2
        bus.write_byte_data(MCP, GPIOB, 7)
        print("ON  [***]")
        time.sleep(0.5)

        # 3. Turn OFF (Send value 0)
        bus.write_byte_data(MCP, GPIOB, 0)
        print("OFF [   ]")
        time.sleep(0.5)

except KeyboardInterrupt:
    # Clean up (Turn off lights before quitting)
    bus.write_byte_data(MCP, GPIOB, 0)
    print("\nTest Stopped.")

except OSError:
    print("\nERROR: Chip not reachable. Check the Red Power Wire again!")
