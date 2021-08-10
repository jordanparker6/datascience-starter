import tqdm
import json
import asyncio
import aiohttp
import logging
from asyncio_throttle import Throttler
from typing import Dict, List, Any, Optional

log = logging.getLogger(__name__)

JsonType = Dict[str, Any]

class AsyncFetch:
    """ A base class for asyncronus HTTP fetching.
    """

    def __init__(self):
        super().__init__()

    def fetch(self, url: str) -> JsonType:
        """Executes an async fetch.

        Args:
            url: A url string.
        
        Returns:
            A json response object.

        """
        return self.fetch_all([url], rate=None)

    def fetch_all(self, urls: List[str], rate: Optional[int] = None) -> List[JsonType]:
        """Executes a throtled async fetch for a list of urls.

        Args:
            urls: A list of url strings.
            rate (optional): The rate to throttle (calls per second).

        Returns:
            A list of json responses.

        """
        async def _fetch_all(self, urls: List[str], rate: Optional[int] = None) -> List[JsonType]:
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
            
        return asyncio.run(_fetch_all(self, urls, rate))

    # ------------------------------------------------
    # - Async Handling Functions ---------------------
    # ------------------------------------------------

    async def _fetch(self, session: aiohttp.ClientSession, url: str, i: int) -> JsonType:
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
            log.debug(f'Made request: {url}. Status: {response.status}', data=resp)
            return json.loads(resp), i

    async def _throttler(self, session: aiohttp.ClientSession, url: str, throttler: Throttler, i: int):
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
    