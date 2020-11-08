import urllib
import json
from typing import Dict, List, Any
from datascience_starter.io import AsyncFetch

class Geocoder(AsyncFetch):
    """ A geocoder using Google Maps geocoding API.

    Harnesses Google Maps geocoding API to return the 
    Longtitude, Latitude and of the address and the 
    formatted address string.

    Args:
        api_key: A Google Public API key with geocoding enabled.

    """

    def __init__(self, api_key: str):
        super().__init__()
        self.key = api_key  #: The Google Public API key with geocoding enabled.
        self.url_base = "https://maps.googleapis.com/maps/api/geocode/json?"    #: The base url of the API endpoint.
        self.throttle_rate = 2  #: The throttling rate (class per second).

    def encode(self, address: str, write_raw: bool = False):
        """Geoencode an address to lat, long and formatted addresses.

        Args:
            address: The address to geoencode.
            write_raw: A boolean to indicate whether or not to write raw outputs to file.

        Yields:
            The parsed json result object.

        """
        self.encode_all([address], write_raw)
    
    def encode_all(self, addresses: List[str], write_raw: bool = False):
        """Geoencode an address to lat, long and formatted addresses.

        Args:
            addresses: The address to geoencode.
            write_raw: A boolean to indicate whether or not to write raw outputs to file.

        Yields:
            The parsed json result object.

        """
        urls = list(map(self._build_url, addresses))
        results = self.fetch_all(urls, rate=self.throttle_rate)
        if write_raw:
            self._write(results)
        for address, result in zip(addresses, results):
            result = self._parse(result, address)
            yield result

    def _build_url(self, address: str):
        params = {"address": address, "key": self.key}
        return self.url_base + urllib.parse.urlencode(params)

    def _write(self, results: str):
        with open("./geocode_results.json", 'w') as f:
            for result in results:
                f.write(json.dumps(result) + "/n")

    def _parse(self, result: Dict[str, Any], address: str):
        self.log.info(f"Parsing: {address}")
        try:
            result = result["results"][0]
            geometry = result["geometry"]["location"]
            return {"formatted_address": result["formatted_address"], "lng": geometry["lng"], "lat": geometry["lat"], "address": address}
        except Exception as e:
            self.log.error(f"{e} : {result}")
            return { "formatted_address" : "ERROR", "lng": None, "lat": None, "address": address }