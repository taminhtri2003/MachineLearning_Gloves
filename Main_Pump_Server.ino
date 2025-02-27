#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>

// See the following for generating UUIDs:
// https://www.uuidgenerator.net/

#define SERVICE_UUID        "e77cdcbe-3f93-4d59-b2be-f468aba8eabe"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// Pin definitions for valves and pumps
const int valve_pump1 = 8;
const int valve_pump2 = 4;
const int valve_finger1 = 5;
const int valve_finger2 = 6;
const int valve_finger3 = 7;
const int valve_finger4 = 15;
const int valve_finger5 = 16;
const int pump1 = 47;
const int pump2 = 48;

// Transition time in milliseconds
const unsigned long transitionTime = 500;

// State variables to track the current and desired states
int currentState = -1; // -1 represents an initial "unknown" state
int desiredState = -1;
void setupPins() {
  // Set all pins to OUTPUT mode
  pinMode(valve_pump1, OUTPUT);
  pinMode(valve_pump2, OUTPUT);
  pinMode(valve_finger1, OUTPUT);
  pinMode(valve_finger2, OUTPUT);
  pinMode(valve_finger3, OUTPUT);
  pinMode(valve_finger4, OUTPUT);
  pinMode(valve_finger5, OUTPUT);
  pinMode(pump1, OUTPUT);
  pinMode(pump2, OUTPUT);
}

void resetOutputs() {
  // Turn off all pumps and close all valves
  digitalWrite(pump1, LOW);
  digitalWrite(valve_pump1, HIGH);
  digitalWrite(pump2, LOW);
  digitalWrite(valve_pump2, HIGH);
  digitalWrite(valve_finger1, LOW);
  digitalWrite(valve_finger2, LOW);
  digitalWrite(valve_finger3, LOW);
  digitalWrite(valve_finger4, LOW);
  digitalWrite(valve_finger5, LOW);
  Serial.println("Outputs reset to OFF.");
}

void setState0() {
  // Configure valves and pumps for state 0
  digitalWrite(valve_pump2, HIGH); // Close pump2 valve
  digitalWrite(valve_pump1, LOW);  // Open pump1 valve
  digitalWrite(pump1, HIGH);      // Turn on pump1
  digitalWrite(valve_finger1, HIGH);
  digitalWrite(valve_finger2, HIGH);
  digitalWrite(valve_finger3, HIGH);
  digitalWrite(valve_finger4, HIGH);
  digitalWrite(valve_finger5, HIGH);

  Serial.println("State set to 0.");
}

void setState1() {
  // Configure valves and pumps for state 1
  digitalWrite(valve_pump1, HIGH); // Close pump1 valve
  digitalWrite(valve_pump2, LOW);  // Open pump2 valve
  digitalWrite(pump2, HIGH);      // Turn on pump2
  digitalWrite(valve_finger1, LOW);
  digitalWrite(valve_finger2, LOW);
  digitalWrite(valve_finger3, LOW);
  digitalWrite(valve_finger4, LOW);
  digitalWrite(valve_finger5, LOW);

  Serial.println("State set to 1.");
}

void updateState() {
  Serial.print("Changing state to: ");
  Serial.println(desiredState);

  // Transition to a neutral "all OFF" state for safety
  resetOutputs();
  delay(transitionTime);

  // Apply the new state
  if (desiredState == 0) {
    setState0();
  } else if (desiredState == 1) {
    setState1();
  }

  // Update the current state and print confirmation
  currentState = desiredState;
  Serial.print("State updated to: ");
  Serial.println(currentState);
}

class MyCallbacks: public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pCharacteristic) {
        std::string value = pCharacteristic->getValue();

        if (value.length() > 0) {
            Serial.println("*********");
            Serial.print("New value: ");
            for (int i = 0; i < value.length(); i++)
                Serial.print(value[i]);

            // Check if the entire string is a number
            bool isNumber = true;
            for (int i = 0; i < value.length(); i++) {
                if (!isdigit(value[i])) {
                    isNumber = false;
                    break;
                }
            }

            if (isNumber) {
                // Convert the string to an integer
                desiredState = std::stoi(value); 
                Serial.println();
                Serial.print("Desired state set to: ");
                Serial.println(desiredState);
            } else {
                Serial.println("Invalid input. Please enter a number.");
            }

            Serial.println("*********");
        }
    }
};

void setup() {
  Serial.begin(115200);
  Serial.println("Starting BLE work!");

  // Configure pin modes for valves and pumps
  setupPins();

  // Ensure all outputs are in a known "OFF" state initially
  resetOutputs();

  BLEDevice::init("Capstone_TaMinhTri");
  BLEServer *pServer = BLEDevice::createServer();
  BLEService *pService = pServer->createService(SERVICE_UUID);
  BLECharacteristic *pCharacteristic = pService->createCharacteristic(
                                         CHARACTERISTIC_UUID,
                                         BLECharacteristic::PROPERTY_READ |
                                         BLECharacteristic::PROPERTY_WRITE
                                       );

  pCharacteristic->setCallbacks(new MyCallbacks());
  pService->start();
  // BLEAdvertising *pAdvertising = pServer->getAdvertising();  // this still is working for backward compatibility
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);  // functions that help with iPhone connections issue
  pAdvertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();
  Serial.println("Characteristic defined! Now you can read it in your phone!");

  
  // Print initial state for debugging
  Serial.println("System initialized. Current state: OFF (-1).");
}

void loop() {
  // Update system state if the desired state has changed
  if (desiredState != currentState) {
    updateState();
  } 
}
