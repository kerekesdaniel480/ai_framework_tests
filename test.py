import psutil
import time

def print_usage(tag=""):
    print(f"{tag} CPU: {psutil.cpu_percent(interval=0.5)}%, RAM: {psutil.virtual_memory().percent}%")

print("Kezdés előtt:")
print_usage("Start")

# Egyszerű terhelés: 1-1000000 szám összeadása
print("\nSzámítás közben:")
start = time.time()
total = 0
for i in range(1, 10000000):
    total += i
    if i % 2000000 == 0:
        print_usage(f"i={i}")
end = time.time()

print("\nVége:")
print_usage("End")
print(f"Összeg: {total}")
print(f"Idő: {end - start:.2f} másodperc")
