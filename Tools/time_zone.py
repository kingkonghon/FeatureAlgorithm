from pytz import timezone, utc
from datetime import datetime

tz = timezone('America/New_York')
now = utc.localize(datetime.utcnow())
now = now.astimezone(tz)

print(now)

if now.hour > 17 and now.hour < 18:
    True