import asyncio, aiohttp, json
from asyncio_throttle import Throttler
import tqdm
from datascience_starter.base.logging import Logger

class AsyncFetch(Logger):
    """
    AsyncFetch
     Provides methods for asyncronously fetching
     json data from a list of URLs with throttling.
    """
    def __init__(self):
        super().__init__()

    def fetch(self, url, rate=None):
        return self.fetch_all([url], rate)

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

    async def _fetch(self, session, url, i):
        """Execute an http call async
        Args:
            session: context for making the http call
            url: URL to call
            i: index of fetch
        Return:
            responses: The json response
        """
        async with session.get(url, timeout=60*30) as response:
            resp = await response.read()
            self.log.debug(f'Made request: {url}. Status: {response.status}')
            return json.loads(resp), i

    async def _throttler(self, session, url, throttler, i):
        """A wrapper to throttle async fetches
            Args:
                session: context for making the http call
                url: URL to call
                rate: the number of concurrent tasks
                throttler : asyncio-throttle class
                i: index of fetch
            Return:
                responses: json response
        """
        if throttler:
            async with throttler:
                return await self._fetch(session, url, i)
        else:
            return await self._fetch(session, url, i)


    async def _fetch_all(self, urls, rate=None):
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
            tasks = [self._throttler(session, url, throttler, i)  for i, url in enumerate(urls)]
            responses = [await f for f in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]
        responses = sorted(responses, key=lambda x: x[1])
        responses = list(map(lambda x: x[0], responses))
        return responses
    