from run_custom_notifications import main as run_once
import time

if __name__ == '__main__':
    while True:
        run_once()
        time.sleep(900)
