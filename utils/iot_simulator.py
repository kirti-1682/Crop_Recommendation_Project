
import random, time

def stream_samples(n=10, delay=0.2):
    for _ in range(n):
        yield {
            'temperature': round(random.uniform(15, 40),2),
            'humidity': round(random.uniform(20, 95),2),
            'ph': round(random.uniform(4.0, 9.0),2),
            'rainfall': round(random.uniform(0, 300),2),
            'N': round(random.uniform(0, 140),2),
            'P': round(random.uniform(0, 140),2),
            'K': round(random.uniform(0, 200),2),
        }
        time.sleep(delay)
