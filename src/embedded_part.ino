#include "esp_camera.h"
#include <WiFi.h>
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

// ============================================================================
//  Analog Display Recognition System - ESP32-CAM Node
//
//  This firmware turns an AI Thinker ESP32-CAM into a simple image capture node.
//  The Python recognition system requests a fresh frame from:
//
//      http://<esp32-ip-address>/capture
//
//  The board responds with raw JPEG data, which the Python pipeline decodes
//  with OpenCV before running object detection.
// ============================================================================


// -----------------------------------------------------------------------------
// Wi-Fi credentials
// -----------------------------------------------------------------------------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";


// -----------------------------------------------------------------------------
// AI Thinker ESP32-CAM pin layout
// -----------------------------------------------------------------------------
namespace CameraPin {
  constexpr int PWDN  = 32;
  constexpr int RESET = -1;
  constexpr int XCLK  = 0;

  constexpr int SIOD = 26;
  constexpr int SIOC = 27;

  constexpr int Y9 = 35;
  constexpr int Y8 = 34;
  constexpr int Y7 = 39;
  constexpr int Y6 = 36;
  constexpr int Y5 = 21;
  constexpr int Y4 = 19;
  constexpr int Y3 = 18;
  constexpr int Y2 = 5;

  constexpr int VSYNC = 25;
  constexpr int HREF  = 23;
  constexpr int PCLK  = 22;
}


// -----------------------------------------------------------------------------
// Capture server
// -----------------------------------------------------------------------------
WiFiServer captureServer(80);


// -----------------------------------------------------------------------------
// Camera configuration
// -----------------------------------------------------------------------------
camera_config_t buildCameraConfig() {
  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0 = CameraPin::Y2;
  config.pin_d1 = CameraPin::Y3;
  config.pin_d2 = CameraPin::Y4;
  config.pin_d3 = CameraPin::Y5;
  config.pin_d4 = CameraPin::Y6;
  config.pin_d5 = CameraPin::Y7;
  config.pin_d6 = CameraPin::Y8;
  config.pin_d7 = CameraPin::Y9;

  config.pin_xclk  = CameraPin::XCLK;
  config.pin_pclk  = CameraPin::PCLK;
  config.pin_vsync = CameraPin::VSYNC;
  config.pin_href  = CameraPin::HREF;

  config.pin_sscb_eda = CameraPin::SIOD;
  config.pin_sscb_scl = CameraPin::SIOC;

  config.pin_pwdn  = CameraPin::PWDN;
  config.pin_reset = CameraPin::RESET;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  /*
    The Python pipeline resizes each received frame to 640x640 before inference.
    SVGA gives the model enough detail while keeping transfer time reasonable.
  */
  if (psramFound()) {
    config.frame_size   = FRAMESIZE_SVGA;  // 800 x 600
    config.jpeg_quality = 10;              // Lower value = better JPEG quality
    config.fb_count     = 2;
  } else {
    config.frame_size   = FRAMESIZE_CIF;   // Safer fallback for boards without PSRAM
    config.jpeg_quality = 12;
    config.fb_count     = 1;
  }

  return config;
}


bool startCamera() {
  camera_config_t config = buildCameraConfig();

  esp_err_t status = esp_camera_init(&config);

  if (status != ESP_OK) {
    Serial.printf("Camera initialization failed. Error: 0x%x\n", status);
    return false;
  }

  Serial.println("Camera ready.");
  return true;
}


// -----------------------------------------------------------------------------
// Wi-Fi connection
// -----------------------------------------------------------------------------
void connectToWiFi() {
  Serial.printf("Connecting to Wi-Fi: %s", WIFI_SSID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Wi-Fi connected.");
  Serial.print("ESP32-CAM IP address: ");
  Serial.println(WiFi.localIP());
}


// -----------------------------------------------------------------------------
// HTTP response helpers
// -----------------------------------------------------------------------------
void sendTextResponse(WiFiClient& client, const char* status, const char* message) {
  client.println(status);
  client.println("Content-Type: text/plain");
  client.println("Connection: close");
  client.println();
  client.println(message);
}


void sendCapture(WiFiClient& client) {
  camera_fb_t* frame = esp_camera_fb_get();

  if (!frame) {
    Serial.println("Capture failed.");
    sendTextResponse(
      client,
      "HTTP/1.1 500 Internal Server Error",
      "Camera capture failed."
    );
    return;
  }

  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: image/jpeg");
  client.print("Content-Length: ");
  client.println(frame->len);
  client.println("Connection: close");
  client.println();

  client.write(frame->buf, frame->len);

  esp_camera_fb_return(frame);
}


// -----------------------------------------------------------------------------
// Request parsing
// -----------------------------------------------------------------------------
String readRequestLine(WiFiClient& client) {
  String requestLine = client.readStringUntil('\n');
  requestLine.trim();
  return requestLine;
}


void skipRequestHeaders(WiFiClient& client) {
  while (client.connected()) {
    String headerLine = client.readStringUntil('\n');
    headerLine.trim();

    if (headerLine.length() == 0) {
      break;
    }
  }
}


void handleRequest(WiFiClient& client) {
  String requestLine = readRequestLine(client);
  skipRequestHeaders(client);

  Serial.print("Request: ");
  Serial.println(requestLine);

  if (requestLine.startsWith("GET /capture ")) {
    sendCapture(client);
    return;
  }

  sendTextResponse(
    client,
    "HTTP/1.1 404 Not Found",
    "ADRS camera node is running. Use /capture to request a frame."
  );
}


// -----------------------------------------------------------------------------
// Arduino lifecycle
// -----------------------------------------------------------------------------
void setup() {
  /*
    ESP32-CAM boards can be sensitive to power dips during camera startup.
    Disabling the brownout detector avoids unwanted resets on unstable supplies.
  */
  WRITE_PERI_REG(RTC_CNTL_BROWNOUT_REG, 0);

  Serial.begin(115200);
  Serial.setDebugOutput(false);

  if (!startCamera()) {
    Serial.println("Startup stopped. Check the camera module and board selection.");
    return;
  }

  connectToWiFi();

  captureServer.begin();

  Serial.println();
  Serial.println("ADRS camera node is online.");
  Serial.print("Python capture URL: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/capture");
}


void loop() {
  WiFiClient client = captureServer.available();

  if (!client) {
    return;
  }

  handleRequest(client);
  client.stop();
}