import requests
from typing import Dict, Any
from autogen_core.tools import FunctionTool

from react_trip_agent.config import EXCHANGE_RATE_API_KEY


def get_currency_rates(base_currency: str = "RUB") -> Dict[str, Any]:
    """
    Get latest currency exchange rates for a base currency.

    Args:
        base_currency (str): The base currency code (default: "RUB")

    Returns:
        Dict[str, Any]: Exchange rate data including conversion rates for various currencies

    Raises:
        ValueError: If the API key is invalid or the currency code is not supported
        ConnectionError: If there's an issue connecting to the API
        Exception: For any other errors during the API request
    """
    try:
        # API endpoint for getting latest exchange rates
        url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/latest/{base_currency}"

        # Make the request to the API
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the response
        exchange_data = response.json()

        # Check if the request was successful
        if exchange_data.get("result") != "success":
            error_type = exchange_data.get("error-type", "Unknown error")
            raise ValueError(f"API error: {error_type}")

        return exchange_data

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Failed to connect to the exchange rate API. Please check your internet connection."
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401 or e.response.status_code == 403:
            raise ValueError(
                "Invalid API key. Please provide a valid Exchange Rate API key."
            )
        elif e.response.status_code == 404:
            raise ValueError(f"Currency not found or not supported: {base_currency}")
        else:
            raise Exception(f"HTTP error occurred: {e}")
    except Exception as e:
        raise Exception(f"An error occurred while fetching exchange rate data: {e}")


currency_tool = FunctionTool(
    get_currency_rates,
    "Tool to get latest currency exchange rates. Returns conversion rates for various currencies based on a specified base currency (default: RUB).",
)
