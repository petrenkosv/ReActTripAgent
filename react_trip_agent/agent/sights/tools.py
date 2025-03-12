import requests
import math
from typing import Dict, Any, Tuple, List
from autogen_core.tools import FunctionTool

from react_trip_agent.config import GEOAPIFY_API_KEY
from react_trip_agent.agent.weather.tools import geocode_city


def calculate_bounding_box(
    lat: float, lon: float, distance_km: float = 30
) -> Tuple[float, float, float, float]:
    """
    Calculate a bounding box around a point given a distance in kilometers.

    Args:
        lat (float): Latitude of the center point
        lon (float): Longitude of the center point
        distance_km (float): Distance in kilometers from the center point

    Returns:
        Tuple[float, float, float, float]: A tuple containing (min_lon, min_lat, max_lon, max_lat)
    """
    # Earth's radius in kilometers
    earth_radius = 6371.0

    # Convert distance to radians
    angular_distance = distance_km / earth_radius

    # Convert latitude and longitude to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # Calculate min and max latitudes
    min_lat = lat_rad - angular_distance
    max_lat = lat_rad + angular_distance

    # Calculate min and max longitudes
    # The width of the box depends on the latitude
    delta_lon = math.asin(math.sin(angular_distance) / math.cos(lat_rad))
    min_lon = lon_rad - delta_lon
    max_lon = lon_rad + delta_lon

    # Convert back to degrees
    min_lat_deg = math.degrees(min_lat)
    max_lat_deg = math.degrees(max_lat)
    min_lon_deg = math.degrees(min_lon)
    max_lon_deg = math.degrees(max_lon)

    return min_lon_deg, min_lat_deg, max_lon_deg, max_lat_deg


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the distance between two points on Earth using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point
        lon1 (float): Longitude of the first point
        lat2 (float): Latitude of the second point
        lon2 (float): Longitude of the second point

    Returns:
        float: Distance in kilometers
    """
    # Earth's radius in kilometers
    earth_radius = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c

    return round(distance, 2)


def get_sights_by_coordinates(
    latitude: float, longitude: float, categories: List[str] = None, limit: int = 20
) -> Dict[str, Any]:
    """
    Get sights near a specific location based on latitude and longitude.

    Args:
        latitude (float): The latitude of the location
        longitude (float): The longitude of the location
        categories (List[str], optional): List of categories to search for. Defaults to tourist attractions.
        limit (int, optional): Maximum number of results to return. Defaults to 20.

    Returns:
        Dict[str, Any]: Sights data including name, address, and coordinates

    Raises:
        ValueError: If API key is not provided
        ConnectionError: If there's an issue connecting to the API
        Exception: For any other errors during the API request
    """
    try:
        # Default categories if none provided
        if categories is None:
            categories = ["tourism.attraction", "tourism.sights"]

        # Calculate bounding box (30km around the coordinates)
        min_lon, min_lat, max_lon, max_lat = calculate_bounding_box(latitude, longitude)

        # Build the categories parameter
        categories_param = ",".join(categories)

        # Build the URL
        url = f"https://api.geoapify.com/v2/places"

        # Set up parameters for the request
        params = {
            "categories": categories_param,
            "filter": f"rect:{min_lon},{min_lat},{max_lon},{max_lat}",
            "limit": limit,
            "apiKey": GEOAPIFY_API_KEY,
        }

        # Set up headers
        headers = {"Accept": "application/json"}

        # Make the request
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the response
        sights_data = response.json()

        # Format the response
        formatted_sights = {
            "sights": [],
            "location": {"latitude": latitude, "longitude": longitude},
            "total_results": len(sights_data.get("features", [])),
            "bounding_box": {
                "min_lon": min_lon,
                "min_lat": min_lat,
                "max_lon": max_lon,
                "max_lat": max_lat,
            },
        }

        # Extract relevant information from each sight
        for feature in sights_data.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})

            sight = {
                "name": properties.get("name", "Unnamed"),
                "categories": properties.get("categories", "").split(","),
                "address": {
                    "formatted": properties.get("formatted", ""),
                    "street": properties.get("street", ""),
                    "city": properties.get("city", ""),
                    "country": properties.get("country", ""),
                },
                "coordinates": {
                    "latitude": geometry.get("coordinates", [0, 0])[1],
                    "longitude": geometry.get("coordinates", [0, 0])[0],
                },
                "distance_km": calculate_distance(
                    latitude,
                    longitude,
                    geometry.get("coordinates", [0, 0])[1],
                    geometry.get("coordinates", [0, 0])[0],
                ),
                "details": {
                    "website": properties.get("website", ""),
                    "phone": properties.get("phone", ""),
                    "opening_hours": properties.get("opening_hours", ""),
                },
            }

            formatted_sights["sights"].append(sight)

        # Sort sights by distance
        formatted_sights["sights"].sort(key=lambda x: x["distance_km"])

        return formatted_sights

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Failed to connect to the Geoapify API. Please check your internet connection."
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401 or e.response.status_code == 403:
            raise ValueError(
                "Invalid API key. Please provide a valid Geoapify API key."
            )
        else:
            raise Exception(f"HTTP error occurred: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while fetching sights data: {e}")


def get_sights_by_city(
    city_name: str, categories: List[str] = None, limit: int = 20
) -> Dict[str, Any]:
    """
    Get sights for a city by name.

    This is a convenience function that combines geocoding and sights retrieval.

    Args:
        city_name (str): The name of the city to get sights for
        categories (List[str], optional): List of categories to search for. Defaults to tourist attractions.
        limit (int, optional): Maximum number of results to return. Defaults to 20.

    Returns:
        Dict[str, Any]: Sights data including name, address, and coordinates

    Raises:
        ValueError: If the city cannot be found
        ConnectionError: If there's an issue connecting to the API
        Exception: For any other errors during the API request
    """
    # First, geocode the city name to get coordinates
    latitude, longitude = geocode_city(city_name)

    # Then, get the sights using those coordinates
    return get_sights_by_coordinates(latitude, longitude, categories, limit)


sights_tool = FunctionTool(
    get_sights_by_city,
    "Tool to get tourist attractions and sights by city name. Returns a list of sights within 30km of the city center.",
)
