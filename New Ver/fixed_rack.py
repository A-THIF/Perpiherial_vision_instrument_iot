import smbus
import time

bus = smbus.SMBus(1)
MCP = 0x20  # Address 20

# Registers
IODIRA = 0x00 # Port A Direction (Inputs)
IODIRB = 0x01 # Port B Direction (Outputs)
GPIOA  = 0x12 # Read Sensors
GPIOB  = 0x13 # Write LEDs
GPPUA  = 0x0C # Pull-up Resistors (Port A)

print("--- FINAL RACK SYSTEM ---")
print("Logic: Direct Mapping (0->0, 1->1)")

def force_setup():
    try:
        # 1. FORCE Port B to be OUTPUTS (This fixes the 'Dim Light')
        bus.write_byte_data(MCP, IODIRB, 0x00) 
        
        # 2. Set Port A to INPUTS
        bus.write_byte_data(MCP, IODIRA, 0xFF)
        
        # 3. Enable Pull-ups for Sensors
        bus.write_byte_data(MCP, GPPUA, 0xFF)
        return True
    except OSError:
        return False

# Initial Setup
if not force_setup():
    print("Check Connections!")

while True:
    try:
        # 1. READ SENSORS
        # If Sensor sees object, it sends 0.
        # If Sensor is empty, it sends 1.
        sensor_state = bus.read_byte_data(MCP, GPIOA)
        
        # 2. LOGIC FIX
        # We are sending the signal DIRECTLY to the LEDs now.
        # If your wiring is "Active Low" (VCC->LED->Pin), this works perfectly.
        # If the previous code was "Reverse", this will be "Correct".
        led_state = sensor_state
        
        # 3. WRITE TO LEDS
        bus.write_byte_data(MCP, GPIOB, led_state)
        
        # 4. SAFETY RE-APPLY (Fixes the "Reset" issue)
        # Every 10 loops, we remind the chip "Hey, Port B is OUTPUT!"
        # This prevents the LEDs from going dim if the power flickers.
        bus.write_byte_data(MCP, IODIRB, 0x00) 

        time.sleep(0.05)

    except OSError:
        force_setup()
