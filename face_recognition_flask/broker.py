from paho.mqtt import client as mqtt

# Configure MQTT broker
MQTT_PORT = 1883
MQTT_BROKER = "0.0.0.0"  # Listen on all network interfaces

# Start broker and listen for messages
def on_connect(client, userdata, flags, rc):
    print("Broker connected with result code", str(rc))

def on_message(client, userdata, msg):
    print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")

broker = mqtt.Client("JetsonNanoBroker")
broker.on_connect = on_connect
broker.on_message = on_message

broker.bind(MQTT_BROKER, MQTT_PORT)
broker.loop_forever()
