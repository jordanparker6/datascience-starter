import urllib
import json
from datascience_starter.io.async_fetch import AsyncFetch

class Geocoder(AsyncFetch):
    def __init__(self, api_key):
        super().__init__()
        self.key = api_key
        self.url_base = "https://maps.googleapis.com/maps/api/geocode/json?"
        self.throttle_rate = 2

    def encode(self, address):
        self.encode_all([address])
    
    def encode_all(self, addresses, write_raw=False):
        urls = list(map(self._build_url, addresses))
        results = self.fetch_all(urls, rate=self.throttle_rate)
        if write_raw:
            self._write(results)
        for address, result in zip(addresses, results):
            result = self._parse(result, address)
            yield result

    def _build_url(self, address):
        params = {"address": address, "key": self.key}
        return self.url_base + urllib.parse.urlencode(params)

    def _write(self, results):
        with open("./geocode_results.json", 'w') as f:
            for result in results:
                f.write(json.dumps(result) + "/n")

    def _parse(self, result, address):
        self.log.info(f"Parsing: {address}")
        try:
            result = result["results"][0]
            geometry = result["geometry"]["location"]
            return {"formatted_address": result["formatted_address"], "lng": geometry["lng"], "lat": geometry["lat"], "address": address}
        except Exception as e:
            self.log.error(f"{e} : {result}")
            return { "formatted_address" : "ERROR", "lng": None, "lat": None, "address": address }