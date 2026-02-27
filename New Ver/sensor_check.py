import smbus
import time

bus = smbus.SMBus(1)
MCP = 0x20  # Address of the MCP Chip

# Registers
IODIRA = 0x00 # Direction Port A
GPIOA  = 0x12 # Read Port A
GPPUA  = 0x0C # Pull-up Resistors

print("--- SENSOR DIAGNOSTIC (A0-A7) ---")
print("1 = Empty (Sensor Off)")
print("0 = DETECTED (Sensor On)")
print("-----------------------------------")

def setup():
    try:
        # Set Port A as INPUT
        bus.write_byte_data(MCP, IODIRA, 0xFF)
        # Turn on Pull-up Resistors (Helps with floating signals)
        bus.write_byte_data(MCP, GPPUA, 0xFF)
        return True
    except OSError:
        print("CRITICAL ERROR: Chip 0x20 not found.")
        print("Check: 1. Red Power Wire  2. Black Ground Wire  3. SDA/SCL")
        return False

if setup():
    while True:
        try:
            # Read all 8 sensors at once
            data = bus.read_byte_data(MCP, GPIOA)
            
            # Print status for each pin
            # We loop from 0 to 7
            output = ""
            for i in range(8):
                # Check if the 'i-th' bit is 0 or 1
                is_detected = not (data & (1 << i)) 
                
                if is_detected:
                    output += f"A{i}:[DETECT]  "
                else:
                    output += f"A{i}:........  "
            
            # \r prints on the same line (like an animation)
            print(output, end='\r')
            time.sleep(0.1)

        except OSError:
            print("\nConnection Lost! Did a wire shake loose?")
            time.sleep(1)
            setup() # Try to reconnect
