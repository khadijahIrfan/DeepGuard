import time
import requests
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:5000/about"
N = 20  # number of requests at the same time

def hit(_):
    start = time.time()
    try:
        r = requests.get(URL, timeout=60)
        return r.status_code, time.time() - start
    except Exception:
        return "error", time.time() - start

begin = time.time()
with ThreadPoolExecutor(max_workers=N) as pool:
    results = list(pool.map(hit, range(N)))
total = time.time() - begin

ok = sum(1 for status, _ in results if status == 200)
times = [t for _, t in results]
print("Requests:", N)
print("Succeeded (HTTP 200):", ok)
print("Total time: %.0f ms" % (total * 1000))
print("Slowest request: %.0f ms" % (max(times) * 1000))