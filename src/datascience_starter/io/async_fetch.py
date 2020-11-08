from typing import Dict, List, Any, Union
from aiohttp import ClientSession, TCPConnector
from asyncio_throttle import Throttler
import tqdm, json, asyncio
from datascience_starter.base.logging import Logger

class AsyncFetch(Logger):
    """ A base class for asyncronus HTTP fetching.
    """

    def __init__(self):
        super().__init__()

    def fetch(self, url: str) -> Dict[str, Any]:
        """Executes an async fetch.

        Args:
            url: A url string.
        
        Returns:
            A json response object.

        """
        return self.fetch_all([url], rate=None)

    def fetch_all(self, urls: List[str], rate: Union[int, None] = None) -> List[Dict[str, Any]]:
        """Executes a throtled async fetch for a list of urls.

        Args:
            urls: A list of url strings.
            rate (optional): The rate to throttle (calls per second).

        Returns:
            A list of json responses.

        """
        return asyncio.run(self._async_fetch_all(urls, self._async_throttled_fetch, rate))

    # ------------------------------------------------
    # - Async Handling Functions ---------------------
    # ------------------------------------------------

    async def _fetch(self, session: ClientSession, url: str, i: int) -> Dict[str, Any]:
        """A handler to execulte a async HTTP request.

        Args:
            session: context for making the http call.
            url: URL to call.
            i: index of fetch.

        Returns:
            A json response object.

        """
        async with session.get(url, timeout=60*30) as response:
            resp = await response.read()
            self.log.debug(f'Made request: {url}. Status: {response.status}')
            return json.loads(resp), i

    async def _throttler(self, session: ClientSession, url: str, throttler: Throttler, i: int):
        """A throttling wrapper.

        Args:
            session: context for making the http call.
            url: URL to call.
            rate: the number of concurrent tasks.
            throttler : asyncio-throttle class.
            i: index of fetch.

        Return:
            The json response object.

        """
        if throttler:
            async with throttler:
                return await self._fetch(session, url, i)
        else:
            return await self._fetch(session, url, i)


    async def _fetch_all(self, urls: List[str], rate: Union[int, None] = None) -> List[Dict[str, Any]]:
        """ Gather many HTTP call made async

        Args:
            urls: a list of url strings
            rate: the max number of coroutines

        Returns:
            A list of json response objects.

        """
        self.log.info('Starting: _async_fetch_all')
        connector = TCPConnector(verify_ssl=False, limit=100)
        throttler = None
        if rate:
            throttler = Throttler(rate_limit=rate, period=1)
        async with ClientSession(connector=connector) as session:
            tasks = [self._throttler(session, url, throttler, i)  for i, url in enumerate(urls)]
            responses = [await f for f in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks))]
        responses = sorted(responses, key=lambda x: x[1])
        responses = list(map(lambda x: x[0], responses))
        return responses
    