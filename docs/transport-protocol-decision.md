# Transport Protocol Decision: HTTP over MQTT

## Decision

The ESP32 nodes transmit thermal frames to the Azure API using **HTTP/1.1 chunked
transfer encoding** rather than MQTT, despite MQTT being the protocol named in the
original project proposal.

## Why MQTT was proposed

MQTT is the standard recommendation for IoT device-to-cloud communication because:

- Minimal per-message overhead (2-byte fixed header vs ~200-byte HTTP headers)
- Persistent TCP connection avoids per-message handshake latency
- Pub/sub decoupling: multiple cloud consumers can subscribe to a topic without the
  device knowing about them
- Well-supported by cloud IoT brokers (Azure IoT Hub, AWS IoT Core, etc.)

These advantages are real, but they apply most strongly to **high-frequency,
low-payload** sensors (e.g. a thermostat publishing a single float every few seconds).

## Why HTTP is the right choice here

### 1. Payload size makes header overhead irrelevant

Each MLX90640 frame is 768 float32 temperatures, serialised to approximately **3.8–4.2 KB
of JSON**. At that payload size, the ~200-byte difference between HTTP and MQTT headers
is less than 5% of the total bytes on the wire. The protocol choice has no measurable
effect on bandwidth or latency.

### 2. MQTT requires the full payload in a single buffer — HTTP does not

`adafruit_minimqtt` (the CircuitPython MQTT library) requires the complete message
payload to be assembled as a single `bytes` or `str` object before calling `publish()`.
Allocating a contiguous ~4 KB buffer on top of the ESP32's already-fragmented heap
(WiFi stack consumes ~40 KB; CircuitPython VM adds further overhead from a 520 KB total)
caused repeated `MemoryError` crashes in early versions of the firmware.

The current HTTP implementation uses **chunked transfer encoding** to stream the frame
in 32-pixel (~224-byte) chunks, writing temperatures directly from the MLX90640 frame
buffer into a pre-allocated bytearray without ever materialising the full JSON string.
This was the key engineering change that made the device stable. Switching to MQTT
would reintroduce the exact allocation pattern the chunked approach was designed to
avoid.

### 3. TLS (required for Azure IoT Hub) has a large memory footprint on CircuitPython

Azure IoT Hub's MQTT endpoint requires TLS. The CircuitPython TLS stack on the
ESP32-WROOM-32 consumes an additional ~20–30 KB of heap at connection time and
significantly increases handshake duration. This further reduces the headroom available
for the frame buffer and increases the risk of `MemoryError` on `getFrame()`.

The current HTTP endpoint is plain HTTP (port 80), which avoids this overhead. The
deployment is on an internal/private network where this is acceptable.

### 4. Azure IoT Hub message quotas

Azure IoT Hub's free tier allows 8,000 device-to-cloud messages per day. At the
current upload rate of one frame per 15 seconds across five sensors, the system
generates approximately **28,800 messages/day** — more than three times the free-tier
limit. The HTTP App Service endpoint has no per-message quota.

### 5. No pub/sub fan-out is needed

MQTT's primary architectural advantage — multiple independent subscribers receiving
the same message — is not currently needed. The only consumer of thermal frames is
the FastAPI server. If a second consumer (e.g. a real-time alerting service) were
added in the future, Azure Service Bus or Event Grid would be a better fit than raw
MQTT anyway, and both can be fed from the existing HTTP endpoint via server-side
forwarding.

## When MQTT would be the right choice

For future nodes that transmit lighter payloads (e.g. a single ambient temperature
float, a PIR motion boolean, a door contact state), MQTT would be preferable:

- Payload fits in a tiny buffer, eliminating the heap-fragmentation risk
- Persistent connection meaningfully reduces per-reading latency at high publish rates
- Pub/sub fan-out becomes useful as the number of downstream consumers grows

A hybrid architecture — MQTT for lightweight auxiliary sensors, HTTP chunked streaming
for the thermal camera nodes — would be reasonable at larger scale.

## Report language

When addressing the MQTT deviation in the written report, suggested framing:

> "The project proposal specified MQTT for device-to-cloud transport. After
> implementation, HTTP/1.1 chunked transfer encoding was selected instead due to a
> specific constraint of the MLX90640 sensor: each frame is approximately 4 KB of
> serialised data, and the CircuitPython MQTT library requires this payload to be
> assembled as a single contiguous heap allocation before transmission. On the
> ESP32-WROOM-32, whose available heap is reduced to under 100 KB after the WiFi stack
> initialises, this caused consistent MemoryError failures. Chunked HTTP streaming
> avoids this by writing pixel data incrementally from a pre-allocated 256-byte buffer.
> For lower-bandwidth sensor nodes, MQTT remains the preferred protocol and would be
> used in a production deployment."
