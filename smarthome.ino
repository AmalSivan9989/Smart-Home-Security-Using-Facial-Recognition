#include <Servo.h> // Include the Servo library

Servo myServo;  // Create a Servo object to control the servo motor
int servoPin = 9;  // Pin connected to the servo signal wire

void setup() {
  Serial.begin(9600);  // Start serial communication at 9600 baud rate
  myServo.attach(servoPin);  // Attach the servo to the pin
  Serial.println("Enter angle (0-180) to move the servo:");
}

void loop() {
  if (Serial.available() > 0) {
    int angle = Serial.parseInt();  // Read the angle from the serial monitor

    if (angle >= 0 && angle <= 180) {
      myServo.write(angle);  // Set the servo to the specified angle
      Serial.print("Servo moved to: ");
      Serial.println(angle);
    } else {
      Serial.println("Please enter a valid angle (0-180).");
    }
  }
}
