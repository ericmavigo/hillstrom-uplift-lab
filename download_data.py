from pathlib import Path
from urllib.request import urlretrieve

URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
DEST = Path(__file__).parent / "data" / "raw" / "hillstrom.csv"
DEST.parent.mkdir(parents=True, exist_ok=True)
if not DEST.exists():
    urlretrieve(URL, DEST)
print(f"Dataset ready: {DEST}")

