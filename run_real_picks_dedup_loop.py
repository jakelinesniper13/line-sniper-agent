from linesniper_agent import LineSniperAgent
from notify_custom import notify_opps
import time
import os

# Keep track of notified picks to avoid duplicates
sent_keys = set()

def main():
    agent = LineSniperAgent(bankroll=1000.0)
    real_opps = []

    # Load prices from CSV file if available
    prices_file = os.getenv("PRICES_CSV_PATH", "data/prices.csv")
    if os.path.exists(prices_file):
        try:
            real_opps += agent.analyse_prices_file(prices_file)
        except Exception as e:
            print(f"Error analysing prices file: {e}")

    # Scan NFL moneyline opportunities
    try:
        nfl_opps = agent.scan_nfl_ml()
        real_opps += nfl_opps
    except Exception as e:
        print(f"Error scanning NFL lines: {e}")

    # Add additional sports scanning if needed
    # try:
    #     ncaaf_opps = agent.scan_ncaaf_ml()
    #     real_opps += ncaaf_opps
    # except Exception as e:
    #     print(f"Error scanning NCAAF lines: {e}")

    # Filter positive expected value opportunities
    positive_opps = [opp for opp in real_opps if opp.get("expected_value", 0) > 0]

    # Deduplicate notifications by unique key
    new_opps = []
    for opp in positive_opps:
        key = (opp.get("event"), opp.get("bookmaker", ""), opp.get("odds"))
        if key not in sent_keys:
            new_opps.append(opp)
            sent_keys.add(key)

    if new_opps:
        notify_opps(agent, new_opps)
    else:
        print("No new positive-EV picks found this cycle.")

if __name__ == "__main__":
    while True:
        main()
        # Sleep for 15 minutes between scans
        time.sleep(900)
