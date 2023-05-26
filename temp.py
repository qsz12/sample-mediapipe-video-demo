import asyncio

async def main():
    print('Hello ...')
    await asyncio.sleep(1)
    print('... World!')

loop1 = asyncio.get_event_loop()
loop1.run_until_complete(main())