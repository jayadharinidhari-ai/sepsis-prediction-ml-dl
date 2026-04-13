# SMS Configuration Guide for Real Alert System

## For Real SMS Alerts to Work:

### Step 1: Get Twilio Account
1. Go to https://www.twilio.com/try-twilio
2. Sign up for a free account
3. Get your:
   - ACCOUNT SID
   - AUTH TOKEN
   - PHONE NUMBER (Twilio number for sending SMS)

### Step 2: Set Environment Variables

**Windows Command Prompt:**
```batch
set TWILIO_ACCOUNT_SID=your_account_sid_here
set TWILIO_AUTH_TOKEN=your_auth_token_here
set TWILIO_PHONE_NUMBER=+1234567890
```

**Or in Python (before running):**
```python
import os
os.environ['TWILIO_ACCOUNT_SID'] = 'your_sid'
os.environ['TWILIO_AUTH_TOKEN'] = 'your_token'
os.environ['TWILIO_PHONE_NUMBER'] = '+1234567890'
```

### Step 3: Update Doctor Phone Numbers

Edit `app/services/store.py` and update doctor phone numbers:

```python
"doctorPhone": "+91-your-actual-phone-number"  # With country code
```

### Step 4: Restart Server
```bash
python run.py
```

### Now:
- High-risk alerts (≥75%) will automatically send SMS every 5 minutes
- Alerts show in the SMS Log with status
- Doctor receives real text messages on their phone

---

## What Happens:

1. **Moderate Patient (Meena):**
   - Starts at 41% risk (Moderate)
   - Gradually increases over ~75 seconds
   - Vitals worsen (HR↑, Temp↑, BP↓)
   - Risk crosses 75% threshold
   - SMS sent to doctor

2. **High Risk Patient (Rajesh):**
   - Stays at 87% high risk (Septic shock)
   - Every 5 minutes: SMS alert sent
   - Shows "SMS sent to Dr. XXX" in log

3. **Stable Patient (Anbu):**
   - Stays low risk (12%)
   - Normal vital variation
   - No alerts

---

## Real-Time Features:

✅ **Live Vital Updates** - Every 5 seconds
✅ **Risk Score Changes** - Updates in real-time
✅ **Audio Alert** - Buzzer when patient hits high risk
✅ **SMS Alerts** - Real text messages to doctors
✅ **Progression Simulation** - Moderate→High demo
✅ **Realistic Data** - Clinically accurate vital changes

---

## Testing Without Real SMS:

If you don't have Twilio yet, alerts are logged in:
- SMS Alert Log (on dashboard)
- Console output
- Can be viewed anytime

To enable SMS later, just set env variables and restart.

---

**Questions?** Check the doctor phone numbers in `app/services/store.py`
