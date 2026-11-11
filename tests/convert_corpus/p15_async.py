import asyncio
async def work(n):
    await asyncio.sleep(0.01 * n)
    return n * 2
async def main():
    r = await asyncio.gather(*(work(i) for i in range(3)))
    print(r)
asyncio.run(main())
