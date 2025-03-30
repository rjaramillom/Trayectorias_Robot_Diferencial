/*
  Envío del valor de 2 ADC cada 10 ms. Voltaje y Corriente. 
*/

#include <WiFi.h>
#include <Firebase_ESP_Client.h>

#include "addons/TokenHelper.h"
#include "addons/RTDBHelper.h"

#define WIFI_SSID "tp-link 312a"
#define WIFI_PASSWORD "12345678"
#define API_KEY "---------llave de base de datos --------"
#define DATABASE_URL "---------url de base de datos --------"

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

// Se establece base de tiempo y pines de medición
unsigned long sendDataPrevMillis = 0;
const int ADC1_PIN = 1;   
const int ADC2_PIN = 2;  
bool signupOK = false;

// Inicialización
void setup() {
  Serial.begin(115200);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(300);
  }
  Serial.println();
  Serial.print("Connected with IP: ");
  Serial.println(WiFi.localIP());

  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;

  if (Firebase.signUp(&config, &auth, "", "")) {
    Serial.println("Firebase SignUp OK");
    signupOK = true;
  } else {
    Serial.printf("Firebase SignUp Error: %s\n", config.signer.signupError.message.c_str());
  }

  config.token_status_callback = tokenStatusCallback;
  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
}

void loop() {
  if (Firebase.ready() && signupOK && (millis() - sendDataPrevMillis > 10)) {
    sendDataPrevMillis = millis();
    int adc1_value = analogRead(ADC1_PIN);
    int adc2_value = analogRead(ADC2_PIN);

    if (Firebase.RTDB.setInt(&fbdo, "/sensors/voltaje", adc1_value)) {
      // Solo para depuracion Serial.printf("voltaje: %d sent\n", adc1_value);
    } else {
      Serial.printf("Error al enviar voltaje: %s\n", fbdo.errorReason().c_str());
    }

    if (Firebase.RTDB.setInt(&fbdo, "/sensors/corriente", adc2_value)) {
      // Solo para depuracion Serial.printf("Corriente: %d sent\n", adc2_value);
    } else {
      Serial.printf("Error al enviar corriente: %s\n", fbdo.errorReason().c_str());
    }
  }
}
