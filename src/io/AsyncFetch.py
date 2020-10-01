import asyncio, aiohttp, json
from asyncio_throttle import Throttler
import tqdm
from base.logging import Logger

class AsyncFetch(Logger):
    """
    AsyncFetch
     Provides methods for asyncronously fetching
     json data from a list of URLs with throttling.
    """
    def __init__(self):
        super().__init__()

    def fetch_all(self, urls, rate=None):
        """ Executes a throtled async fetch of url list
            Args:
                urls: list of url strings
                rate (optional): an int for the number of coroutines
            Return:
                responses: list of json responses
        """
        return asyncio.run(self._async_fetch_all(urls, self._async_throttled_fetch, rate))

    # ------------------------------------------------
    # - Async Handling Functions ---------------------
    # ------------------------------------------------

    async def _async_fetch(self, session, url):
        """Execute an http call async
        Args:
            session: context for making the http call
            url: URL to call
        Return:
            responses: The json response
        """
        async with session.get(url, timeout=60*30) as response:
            resp = await response.read()
            self.log.debug(f'Made request: {url}. Status: {response.status}')
            return json.loads(resp)

    async def _async_throttled_fetch(self, session, url, throttler):
        """Execute an http call async
            Args:
                session: context for making the http call
                url: URL to call
                rate: the number of concurrent tasks
                throttler : asyncio-throttle class
            Return:
                responses: json response
        """
        if throttler:
            async with throttler:
                return await self._async_fetch(session, url)
        else:
            return await self._async_fetch(session, url)


    async def _async_fetch_all(self, urls, fetch, rate=None):
        """ Gather many HTTP call made async
        Args:
            urls: a list of url strings
            fetch: an async function that contains the fetch logic
            rate: the max number of coroutines
        Return:
            responses: list of json responses
        """
        self.log.info('Starting: _async_fetch_all')
        connector = aiohttp.TCPConnector(verify_ssl=False, limit=100)
        throttler = None
        if rate:
            throttler = Throttler(rate_limit=rate, period=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [fetch(session, url, throttler)  for url in urls]
            responses = [await f for f in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]
        return responses
    