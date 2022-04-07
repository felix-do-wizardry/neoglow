#include <math.h>
#include <string.h>

// #include <WiFi.h>
#include <ESP8266WiFi.h>
#include <WebSocketsServer.h>

#include <iostream>
#include "ArduinoJson.h"

#include <Adafruit_NeoPixel.h>

#define PIN_pixel D2

// Constants
const String name = "node3";
char* ssid = "Felix_AX3Pro";
char* password = "WinterBoySummerGirl@wifi6";
//const int NUM_PIXELS = 8;
//const int NUM_PIXELS = 40;
//const int NUM_PIXELS = 60;
//const int NUM_PIXELS = 120;
//const int NUM_PIXELS = 144;
const int NUM_PIXELS = 60 * 2 + 144;
//const int NUM_PIXELS = 256;

const int NUM_SUBPIXELS = NUM_PIXELS * 3;
const int ws_port = 8908;

// int rgbValues[] = {0, 120, 255};
// int rgbValues[NUM_SUBPIXELS];

// int pixel_values[];
int pixel_mode = 1; // 0=default blink | 1=controled RGB
int color_updated = 1;
int loop_index = 0;
int wifi_state = 0;
float fps = 0.0;
unsigned long time_start = millis();
unsigned long time_current = time_start;
unsigned long time_last = time_start;
unsigned long time_last_fps = time_start;
unsigned long time_elapsed = 0;
unsigned long time_since_last = 0;
long time_delay = 10;

int pixel_indices[NUM_PIXELS];
float pixel_pos_a[NUM_PIXELS];

Adafruit_NeoPixel pixels = Adafruit_NeoPixel(NUM_PIXELS, PIN_pixel, NEO_GRB + NEO_KHZ800);
// Adafruit_NeoPixel pixels;

// json
//StaticJsonDocument<3600> doc;
// StaticJsonDocument<3000> doc;
// ws
WebSocketsServer webSocket = WebSocketsServer(ws_port);


// nn (NodeNeo) WS data fields
int nn_time_offset[NUM_PIXELS];
float nn_hue_base = 0.;
float nn_hue_delta = 1.;
float nn_saturation = 1.;
float nn_value = 0.5;
unsigned long nn_duration = 4000;
unsigned long nn_time_start = 0;


typedef struct {
    double r;       // a fraction between 0 and 1
    double g;       // a fraction between 0 and 1
    double b;       // a fraction between 0 and 1
} rgb;
typedef struct {
    double h;       // angle in degrees
    double s;       // a fraction between 0 and 1
    double v;       // a fraction between 0 and 1
} hsv;
typedef struct {
    uint16_t h;      // hue [0, 65535]
    uint8_t s;       // sat [0, 255]
    uint8_t v;       // val [0, 255]
} hsvP;
static hsv   rgb2hsv(rgb in);
static rgb   hsv2rgb(hsv in);
rgb cc;
rgb colors[NUM_PIXELS];
hsv colors_hsv[NUM_PIXELS];
hsvP colors_hsvP[NUM_PIXELS];

rgb hsv2rgb(hsv in)
{
    double      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch(i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;     
}

// Called when receiving any WebSocket message
void onWebSocketEvent(uint8_t num,
                      WStype_t type,
                      uint8_t * payload,
                      size_t length) {

  // Figure out the type of WebSocket event
  switch(type) {

    // Client has disconnected
    case WStype_DISCONNECTED:
//      Serial.printf("[%u] Disconnected!\n", num);
      break;

    // New client has connected
    case WStype_CONNECTED:
      {
        IPAddress ip = webSocket.remoteIP(num);
//        Serial.printf("[%u] Connection from ", num);
//        Serial.println(ip.toString());
      }
      break;

    case WStype_TEXT:
    
      // Serial.printf("[%u] Text: %s\n", num, payload);
      // Echo text message back to client
      // webSocket.sendTXT(num, payload);
      
      
      // Serial.printf("MSG received\n");
      
      // TODO: enable for processing of payloads
      if (processPayload(payload) > 0) {
        String _txt = "{ \"fps\" : " + String(fps) + " , \"name\" : " + name + " }";
        webSocket.sendTXT(num, _txt);
      }
      break;

    // For everything else: do nothing
    case WStype_BIN:
    case WStype_ERROR:
    case WStype_FRAGMENT_TEXT_START:
    case WStype_FRAGMENT_BIN_START:
    case WStype_FRAGMENT:
    case WStype_FRAGMENT_FIN:
    default:
      break;
  }
}

void setup_wifi() {
  wifi_state = 0;
  Serial.print("\n\nConnecting to Wifi");
  WiFi.begin(ssid, password);
  for (int i=0; i<20; i++) {
    delay(500);
    Serial.print(".");
    if ( WiFi.status() == WL_CONNECTED ) {
      wifi_state = 1;
      Serial.print("\nWifi Connected @ ");
      Serial.println(WiFi.localIP());
      return;
    }
  }
  Serial.println("Wifi Failed");
}

Adafruit_NeoPixel setup_pixels(uint16_t n, uint16_t p) {
  Adafruit_NeoPixel _pixels = Adafruit_NeoPixel(n, p, NEO_GRB + NEO_KHZ800);
  // _pixels.begin();
  // _pixels.clear();
  // _pixels.show();
  return _pixels;
}

void setup() {
  time_start = millis();
  
  Serial.println("");
  Serial.println("");
  Serial.print("BOARD[");
  Serial.print(name);
  Serial.println("]");

  // Start Serial port
  Serial.begin(115200);
  
  // Connect to access point
  setup_wifi();
  
  // pixels = setup_pixels(NUM_PIXELS, PIN_pixel);
  pixels.begin();
  pixels.clear();
  pixels.show();
  
  // Start WebSocket server and assign callback
  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);
  Serial.print("WS starting @ ");
  Serial.print(WiFi.localIP());
  Serial.print(":");
  Serial.println(ws_port);
  
  pinMode(2, OUTPUT);
  digitalWrite(2, HIGH);
  
  for (int i=0; i<NUM_PIXELS; i++) {
    // colors[i].r = 0.;
    // colors[i].g = 0.;
    // colors[i].b = 0.;
    // colors_hsv[i].h = 0.;
    // colors_hsv[i].s = 0.;
    // colors_hsv[i].v = 0.;
    
    colors_hsvP[i].h = 0;
    colors_hsvP[i].s = 255;
    colors_hsvP[i].v = 0;
    
    // rgbValues[i * 3 + 0] = 0;
    // rgbValues[i * 3 + 1] = 0;
    // rgbValues[i * 3 + 2] = 0;
    
    nn_time_offset[i] = (int)((float)i / NUM_PIXELS * nn_duration);
  }
  
  Serial.println("creating pixels");
  for (int i=0; i<NUM_PIXELS; i++) {
    pixel_indices[i] = i;
    pixel_pos_a[i] = (float)(i % NUM_PIXELS) / NUM_PIXELS;
  }
  Serial.println("pixels created");
  
}

void loop() {
  if (loop_index % 3 == 0) {
    digitalWrite(2, LOW);
    // delay(1);
    webSocket.loop();
    digitalWrite(2, HIGH);
  }
  
  // delay(1);
  time_current = millis();
  time_since_last = time_current - time_last;
  time_elapsed = time_current - time_start;
  
  if (time_since_last < time_delay) {
    delay(time_delay - time_since_last);
  }
  time_last = time_current;
  
  int time_elapsed_fps = time_current - time_last_fps;
  // if (time_elapsed_fps >= 2000) {
  if (loop_index % 500 == 0 && loop_index > 0) {
    time_last_fps = time_current;
    float _fps = 200. / max(time_elapsed_fps, 1) * 1000;
    Serial.print("fps[");
    Serial.print(_fps);
    Serial.println("]");
  }
  
  
  unsigned long nn_time_elapsed = time_current - nn_time_start;
  // nn_time_elapsed = nn_time_elapsed % nn_duration;
  // hsv color_hsv_temp;
  // color_hsv_temp.s = nn_saturation;
  // color_hsv_temp.v = nn_value;
  
  for (int i=0; i<NUM_PIXELS; i++) {
    // if (loop_index % 500 == 0) {
    //   if (i == 0) {
    //     Serial.print("\npixel time offset: ");
    //   }
    //   Serial.print(nn_time_offset[i]);
    //   Serial.print(" ");
    // }
    
    float _state = (float)(nn_time_elapsed + nn_time_offset[i]) / nn_duration;
    float h = nn_hue_base + _state * nn_hue_delta;
    h = h - floor(h);
    
    // color_hsv_temp.h = h * 360.;
    
    // colors[i] = hsv2rgb(color_hsv_temp);
    
    colors_hsvP[i].h = (int)floor(min(max(h * 65536., 0.), 65535.));
    colors_hsvP[i].s = (int)(nn_saturation * 255);
    colors_hsvP[i].v = (int)(nn_value * 255);
    
  }
  
  color_updated = 1;
  
  if (color_updated >= 1) {
    // color_updated = color_updated - 1;
    for (int i=0; i<NUM_PIXELS; i++) {
      pixels.setPixelColor(
        i,
        // (int)round(colors[i].r * 255),
        // (int)round(colors[i].g * 255),
        // (int)round(colors[i].b * 255)
        // pixels.Color(20, 120, 100)
        pixels.ColorHSV(colors_hsvP[i].h, colors_hsvP[i].s, colors_hsvP[i].v)
      );
    }
    pixels.show();
  }
  
  loop_index = loop_index + 1;
  if (loop_index >= 1000000000000) {
    loop_index = 0;
    time_start = millis();
  }
  
}

int processPayload(uint8_t* payload) {
    // String str = (char*)buff;
    process_msg((char*)payload);
    int _ping = 0;
    return _ping;
}

int parse_hex_single(String s) {
  const char* chars = s.c_str();
  int char_len = strlen(chars);
  
  char subbuff[char_len + 1];
  
  memcpy(subbuff, &chars[0], char_len);
  subbuff[char_len] = '\0';
  int v = strtol(subbuff, NULL, 16);
  return v;
}

int parse_hex(String s, int char_count, int * out, int out_len) {
  const char* chars = s.c_str();
  // Serial.println(chars);
  // int char_len = sizeof(chars);
  int char_len = strlen(chars);
  
  char_len = char_len - (char_len % char_count);
  int num_count = floor(char_len / char_count);
  int nums[num_count];
  
  char subbuff[char_count + 1];
  
  // Serial.println("");
  // Serial.print("<parse_hex> s=");
  // Serial.print(s);
  // Serial.print(" | num_count=");
  // Serial.print(num_count);
  // Serial.println("");
  // Serial.print("fill to len: ");
  // Serial.print(sizeof(out));
  // Serial.println("");
  // Serial.print("nums:");
  
  // Serial.printf("processing pixels color from payload: ");
  for (int i = 0; i < num_count; i++) {
    if (i >= out_len) {
      break;
    }
    memcpy(subbuff, &chars[i * char_count], char_count);
    subbuff[char_count] = '\0';
    int v = strtol(subbuff, NULL, 16);
    // nums[i] = v;
    out[i] = v;
    
    // Serial.print(" ");
    // Serial.print(v);
  }
  // Serial.println("");
  // return nums;
  return 0;
}

int process_msg(String str) {
  // String str = "A1N c2 t120 d4 t0 b5 t0 a2 t368 e2 t452 1c t0 e1 t600";
  String strs[20];
  int StringCount = 0;
  String _str = str;
  
  while (str.length() > 0) {
    int index = str.indexOf('|');
    if (index == -1) {
      strs[StringCount++] = str;
      break;
    } else {
      strs[StringCount++] = str.substring(0, index);
      str = str.substring(index+1);
    }
  }
  
  // print results
  // Serial.print("[");
  // Serial.print(_str);
  // Serial.print("] -> ");
  nn_time_start = millis();
  for (int i = 0; i < StringCount; i++) {
    String s = strs[i];
    if (s.length() < 1) {continue;}
    // if (i > 0) {
    //   Serial.print(" + ");
    // }
    // Serial.print("[");
    // Serial.print(strs[i]);
    // Serial.print("]");
    
    // Serial.println("");
    // Serial.print("grabbing the str: ");
    // Serial.println(s);
    
    // if (i >= 1) {
    //   Serial.println("");
    //   // int * _values = parse_hex(s, 3);
    //   Serial.print("value[");
    //   // int _value = _values[0];
    //   int _value = parse_hex_single(s);
    //   Serial.print(_value);
    //   Serial.print("] - v [");
    //   float v = (float)(_value) / pow(16, 3);
    //   Serial.print(v);
    //   Serial.print("]");
      
    //   Serial.println("");
    // }
    // Serial.println("");
    
    // int nn_time_offset[NUM_PIXELS];
    // float nn_hue_base = 0.;
    // float nn_hue_delta = 1.;
    // float nn_saturation = 1.;
    // float nn_value = 0.5;
    // unsigned long nn_duration = 4000;
    // unsigned long nn_time_start = 0;
    
    switch(i) {
    case 0: { // offset - 4
      // auto _values = parse_hex(s, 4, nn_time_offset);
      // Serial.print("\n[UPDATED] offset: ");
      // Serial.print("str_len=");
      // Serial.print(s.length());
      // Serial.print(" | val_len=");
      // Serial.print(sizeof(_values));
      // Serial.print(" | ");
      // if (sizeof(_values) == 1) {
      if (s.length() <= 4) {
        // Serial.print("(single) ");
        // Serial.print(_values[0]);
        int _value = parse_hex_single(s);
        for (int j; j < NUM_PIXELS; j++) {
          nn_time_offset[j] = _value;
        }
      } else {
        // Serial.print("(multi) ");
        auto _ = parse_hex(s, 4, nn_time_offset, NUM_PIXELS);
        // for (int j; j < sizeof(_values); j++) {
        //   Serial.print(_values[j]);
        //   Serial.print(",");
        //   if (j >= NUM_PIXELS) {
        //     break;
        //   }
        //   nn_time_offset[j] = _values[j];
        // }
      }
    } break;
    case 1: { // hue_base - 3
      nn_hue_base = (float)parse_hex_single(s) / pow(16, 3);
    } break;
    case 2: { // hue_delta - 3
      nn_hue_delta = (float)parse_hex_single(s) / pow(16, 3);
    } break;
    case 3: { // sat - 2
      nn_saturation = (float)parse_hex_single(s) / pow(16, 2);
    } break;
    case 4: { // val - 2
      nn_value = (float)parse_hex_single(s) / pow(16, 2);
    } break;
    case 5: { // duration - ?
      nn_duration = parse_hex_single(s);
    } break;
    default:
      break;
    }
  }
  
  return StringCount;
}
