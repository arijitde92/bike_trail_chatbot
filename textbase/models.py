# textbase/models.py
import json
import openai
import requests
import time
import typing
from textbase.trail import Trail
from textbase.message import Message


class OpenAI:
    api_key = None

    # The function list that will be passed for "functions" argument in create function of open ai chatbot
    my_custom_functions = [
        {
            'name': 'get_lat_long',
            'description': 'Get latitude and longitude of the input location',
            'parameters': {
                'type': 'object',
                'properties': {
                    'location': {
                        'type': 'string',
                        'description': 'The name of the location'
                    }
                }
            }
        }
    ]

    @classmethod
    def get_lat_long(cls, location: str) -> typing.Tuple[str, str]:
        """
        Uses the geocode.maps.co API to get the latitude and longitude information about an area.
        Inputs:
            location (str): Name of the location whose latitude and longitude needs to be found.

        Returns:
            latitude (str): The latitude of the location
            longitude (str): The longitude of the location
        """
        response = requests.get(f"https://geocode.maps.co/search?q={location}")
        my_json = response.content.decode('utf8').replace("'", '"')

        # Load the JSON to a Python list & dump it back out as formatted JSON
        data = json.loads(my_json)
        latitude = data[0]['lat']
        longitude = data[0]['lon']
        return latitude, longitude

    @classmethod
    def get_trail_json(cls, latitude: str, longitude: str) -> str:
        """
        Uses the latitude and longitude information to find bike trail information using the trailapi explore endpoint.
        More about trailapi can be found here - https://rapidapi.com/trailapi/api/trailapi
        Inputs:
            latitude (str): latitude value
            longitude (str): longitude value
        Returns:
            response (json): The trail information in json format
        """
        url = "https://trailapi-trailapi.p.rapidapi.com/trails/explore/"

        querystring = {"lat": latitude, "lon": longitude}

        headers = {
            "X-RapidAPI-Key": "f4f9e121c1msha704ca037d6b542p1931a2jsn49d927804b5f",
            "X-RapidAPI-Host": "trailapi-trailapi.p.rapidapi.com"
        }
        response = requests.get(url, headers=headers, params=querystring)
        return response.json()

    @classmethod
    def get_trail_info(cls, trail_json: typing.Dict) -> list[Trail]:
        """
        Extracts the trail information from the trail data in json format returned from get_trail_json
        Inputs:
            trail_json (Dict): trail information in json format
        Returns:
            trails (list[Trail]): List of Trail objects containing trail information
        """
        trails = []
        for trail_data in trail_json['data']:
            id = trail_data['id']
            name = trail_data['name']
            url = trail_data['url']
            length = trail_data['length']
            desc = trail_data['description']
            direction = trail_data['directions']
            city = trail_data['city']
            region = trail_data['region']
            country = trail_data['country']
            difficulty = trail_data['difficulty']
            features = trail_data['features']
            rating = trail_data['rating']
            # Creates a Trail pydantic model using the above fields
            trail = Trail(id=id, name=name, url=url, length=length, description=desc, directions=direction,
                          city=city, region=region, country=country, difficulty=difficulty,
                          features=features, rating=rating)
            trails.append(trail)
        return trails

    @classmethod
    def generate(
            cls,
            system_prompt: str,
            message_history: list[Message],
            model="gpt-3.5-turbo",
            max_tokens=3000,
            temperature=0.7,

    ):
        assert cls.api_key is not None, "OpenAI API key is not set"
        openai.api_key = cls.api_key

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                *map(dict, message_history),
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            functions=OpenAI.my_custom_functions,
            function_call='auto'
        )
        print(response)
        try:    # If the model intends to call the function, the content will be null
            function_args = json.loads(response["choices"][0]["message"]["function_call"]["arguments"])
            latitude, longitude = OpenAI.get_lat_long(function_args['location'])
            trail_json = OpenAI.get_trail_json(latitude, longitude)
            trail_infos = OpenAI.get_trail_info(trail_json)
            return_string = ""
            # Will show only one trail information, it can be changed to show multiple trail information
            for trail_info in trail_infos[:1]:
                name = trail_info.name
                description = trail_info.description
                url = trail_info.url
                return_string += (f'Name: {name}. Description: {description}.'
                                  f' You can check more about this trail at: {url}')
            print(return_string)
            return return_string
        except KeyError:    # If the model doesn't intend to call a function, the content will be displayed in chat
            return response["choices"][0]["message"]["content"]



class HuggingFace:
    api_key = None

    @classmethod
    def generate(
            cls,
            system_prompt: str,
            message_history: list[Message],
            model: typing.Optional[str] = "microsoft/DialoGPT-small",
            max_tokens: typing.Optional[int] = 3000,
            temperature: typing.Optional[float] = 0.7,
            min_tokens: typing.Optional[int] = None,
            top_k: typing.Optional[int] = None
    ) -> str:
        try:
            assert cls.api_key is not None, "Hugging Face API key is not set"

            headers = {"Authorization": f"Bearer {cls.api_key}"}
            API_URL = "https://api-inference.huggingface.co/models/" + model
            inputs = {
                "past_user_inputs": [system_prompt],
                "generated_responses": [
                    f"ok I will answer according to the context, where context is '{system_prompt}'"],
                "text": ""
            }

            for message in message_history:
                if message.role == "user":
                    inputs["past_user_inputs"].append(message.content)
                else:
                    inputs["generated_responses"].append(message.content)

            inputs["text"] = inputs["past_user_inputs"].pop(-1)
            payload = {
                "inputs": inputs,
                "max_length": max_tokens,
                "temperature": temperature,
                "min_length": min_tokens,
                "top_k": top_k,
            }
            data = json.dumps(payload)
            response = requests.request("POST", API_URL, headers=headers, data=data)
            response = json.loads(response.content.decode("utf-8"))

            if response.get("error", None) == "Authorization header is invalid, use 'Bearer API_TOKEN'":
                print("Hugging Face API key is not correct")

            if response.get("estimated_time", None):
                print(f"Model is loading please wait for {response.get('estimated_time')}")
                time.sleep(response.get("estimated_time"))
                response = requests.request("POST", API_URL, headers=headers, data=data)
                response = json.loads(response.content.decode("utf-8"))

            return response["generated_text"]
        except Exception as ex:
            print(f"Error occured while using this model, please try using another model, Exception was {ex}")


class BotLibre:
    application = None
    instance = None

    @classmethod
    def generate(
            cls,
            message_history: list[Message],
    ):
        request = {"application": cls.application, "instance": cls.instance, "message": message_history[-1].content}
        response = requests.post('https://www.botlibre.com/rest/json/chat', json=request)
        data = json.loads(response.text)  # parse the JSON data into a dictionary
        message = data['message']
        return message
