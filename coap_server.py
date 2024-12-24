import asyncio
import logging
import aiocoap.resource as resource
import aiocoap
import json
import random



def get_data_as_json():
    data = {
        "temperature": round(random.uniform(20.0, 30.0), 2),
        "humidity": round(random.uniform(30.0, 50.0), 2),
        "anomaly": random.choice([False, True])
    }
    return json.dumps(data)


class SensorDataResource(resource.Resource):

    async def render_get(self, request):
        payload = get_data_as_json().encode('utf-8')
        return aiocoap.Message(payload=payload, content_format=50)



async def main():
    logging.basicConfig(level=logging.INFO)
    root = resource.Site()
    root.add_resource(['sensor'], SensorDataResource())
    await aiocoap.Context.create_server_context(root, bind=('0.0.0.0', 5683))
    await asyncio.get_running_loop().create_future()

if __name__ == "__main__":
    asyncio.run(main())
