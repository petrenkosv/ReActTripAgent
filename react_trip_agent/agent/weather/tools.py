import requests
from typing import Dict, Any, Tuple
from autogen_core.tools import FunctionTool

from ...config import OPEN_WHEATHER_KEY, GEOCODING_URL, WHEATHER_BASE_URL


def geocode_city(city_name: str) -> Tuple[float, float]:
    """
    Convert a city name to latitude and longitude coordinates using OpenWeatherMap Geocoding API.

    Args:
        city_name (str): The name of the city to geocode
        api_key (str, optional): OpenWeatherMap API key. If not provided, will try to use environment variable.

    Returns:
        Tuple[float, float]: A tuple containing (latitude, longitude) coordinates

    Raises:
        ValueError: If API key is not provided and not found in environment variables,
                   or if the city cannot be found
        ConnectionError: If there's an issue connecting to the geocoding API
        Exception: For any other errors during the API request
    """
    try:
        # Set up parameters for the geocoding request
        params = {
            "q": city_name,
            "limit": 1,  # Get only the first (most relevant) result
            "appid": OPEN_WHEATHER_KEY,
        }

        # Make the request to the geocoding API
        response = requests.get(GEOCODING_URL, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the response
        geocoding_data = response.json()

        # Check if we got any results
        if not geocoding_data:
            raise ValueError(f"City not found: {city_name}")

        # Extract latitude and longitude from the first result
        first_result = geocoding_data[0]
        latitude = first_result.get("lat")
        longitude = first_result.get("lon")

        if latitude is None or longitude is None:
            raise ValueError(f"Incomplete geocoding data for city: {city_name}")

        return float(latitude), float(longitude)

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Failed to connect to the geocoding API. Please check your internet connection."
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise ValueError(
                "Invalid API key. Please provide a valid OpenWeatherMap API key."
            )
        else:
            raise Exception(f"HTTP error occurred: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while geocoding city name: {e}")


def get_weather_forecast(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Get weather forecast for a specific location based on latitude and longitude.

    Args:
        latitude (float): The latitude of the location
        longitude (float): The longitude of the location
        api_key (str, optional): OpenWeatherMap API key. If not provided, will try to use environment variable.

    Returns:
        Dict[str, Any]: Weather forecast data including current conditions and 5-day forecast

    Raises:
        ValueError: If API key is not provided and not found in environment variables
        ConnectionError: If there's an issue connecting to the weather API
        Exception: For any other errors during the API request
    """
    try:
        # Get current weather data
        current_weather_url = f"{WHEATHER_BASE_URL}/weather"
        current_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": OPEN_WHEATHER_KEY,
            "units": "metric",
        }

        current_response = requests.get(current_weather_url, params=current_params)
        current_response.raise_for_status()
        current_data = current_response.json()

        # Get 5-day forecast data
        forecast_url = f"{WHEATHER_BASE_URL}/forecast"
        forecast_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": OPEN_WHEATHER_KEY,
            "units": "metric",
        }

        forecast_response = requests.get(forecast_url, params=forecast_params)
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        # Combine the data
        weather_data = {
            "current": current_data,
            "forecast": forecast_data,
            "location": {
                "latitude": latitude,
                "longitude": longitude,
                "name": current_data.get("name", "Unknown"),
                "country": current_data.get("sys", {}).get("country", "Unknown"),
            },
            "units": "metric",
        }

        return weather_data

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Failed to connect to the weather API. Please check your internet connection."
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise ValueError(
                "Invalid API key. Please provide a valid OpenWeatherMap API key."
            )
        elif e.response.status_code == 404:
            raise ValueError(
                f"Weather data not found for coordinates: {latitude}, {longitude}"
            )
        else:
            raise Exception(f"HTTP error occurred: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while fetching weather data: {e}")


def get_weather_by_city(city_name: str) -> Dict[str, Any]:
    """
    Get weather forecast for a city by name.

    This is a convenience function that combines geocoding and weather forecast retrieval.

    Args:
        city_name (str): The name of the city to get weather for
        api_key (str, optional): OpenWeatherMap API key. If not provided, will try to use environment variable.

    Returns:
        Dict[str, Any]: Weather forecast data including current conditions and 5-day forecast

    Raises:
        ValueError: If API key is not provided and not found in environment variables,
                   or if the city cannot be found
        ConnectionError: If there's an issue connecting to the API
        Exception: For any other errors during the API request
    """
    # First, geocode the city name to get coordinates
    latitude, longitude = geocode_city(city_name)

    # Then, get the weather forecast using those coordinates
    return get_weather_forecast(latitude, longitude)


weather_tool = FunctionTool(
    get_weather_by_city,
    "Tool to get weather by city name. Returns current weather and 4 day forecast",
)
