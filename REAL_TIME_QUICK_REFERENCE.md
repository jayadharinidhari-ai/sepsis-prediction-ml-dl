# ⚡ Real-Time Monitoring - Quick Reference Card

## 🚀 Quick Start

### **1. Start System**
```bash
cd C:\Users\aruna\OneDrive\Desktop\ML - Sepsis
python run.py
```
Expected output:
```
✓ Patient simulator started
✓ Alert sent to Dr. Priya Nair
* Running on http://127.0.0.1:5000
```

### **2. Open Dashboard**
```
http://localhost:5000/dashboard
```

### **3. Watch Meena Deteriorate**
1. Click "Meena Subramaniam" in sidebar
2. Watch vitals update every 5 seconds
3. Watch risk increase from 41% → 88%
4. After ~75 seconds: Audio alert + SMS log

---

## 📊 Patient Overview

| Patient | Status | Risk | Key Vitals | Trend |
|---------|--------|------|-----------|-------|
| **Rajesh** | Septic Shock | 🔴 87% | HR 120, MAP 55, Temp 39 | Stable HIGH |
| **Meena** | Deteriorating | 🟡→🔴 41%→88% | HR ↑, Temp ↑, MAP ↓ | PROGRESSION |
| **Anbu** | Stable | 🟢 8% | HR 78, MAP 90, Temp 37 | Normal |

---

## ⚡ Real-Time Data Flow

```
Simulator        Dashboard        User
(5 sec updates)  (3 sec polling)  (visual)
     ↓                ↓             ↓
Vitals Change → API Request → Charts Refresh
     ↓                ↓             ↓
Risk Update → Data Fetched → Risk Badge Changes
     ↓                ↓             ↓
Alert Check → Alert Log → Audio Buzzer
```

---

## 🎯 Meena's Progression Timeline

```
Time        Risk    HR      MAP     Alert
----        ----    ----    ----    -----
0:00        41%     96      76      🟢
0:15        50%     102     74      🟡
0:30        60%     108     72      🟡
0:45        70%     114     68      🟡
1:00        75%     118     65      🔴 ← ALERT!
1:15        82%     124     63      🔴
1:30        88%     128     62      🔴 stable
```

---

## 🔊 Alert System

### **Audio Alert**
- **Trigger**: Risk ≥ 75%
- **Sound**: Double-beep (800Hz)
- **Pattern**: beep-pause-beep
- **Duration**: 0.5s per beep

### **SMS Alert**
- **Trigger**: Risk ≥ 75%
- **Frequency**: Every 5 minutes
- **Format**: `🚨 SEPSIS ALERT: Name - Risk X% | Ward: ICU-X`
- **Status**: Logging works, SMS requires Twilio

---

## 📈 What to Watch For

### **Vital Signs Indicating Sepsis**
- ⬆️ Heart Rate (tachycardia)
- ⬆️ Temperature (fever)
- ⬇️ MAP (hypotension) ← **CRITICAL**
- ⬇️ O₂ Saturation
- ⬆️ Respiratory Rate

### **Lab Values Indicating Sepsis**
- ⬆️ Lactate (>2 mmol/L = concern)
- ⬆️ WBC (infection response)
- ⬆️ Creatinine (kidney injury)
- ⬇️ Platelets (coagulopathy)
- ⬇️ pH (acidosis)

---

## 🎨 Risk Color Coding

| Color | Risk Level | Action | Example |
|-------|-----------|--------|---------|
| 🟢 Green | <40% | Routine monitoring | Anbu (8%) |
| 🟡 Yellow | 40-75% | Close monitoring | Meena phase 1 (41%) |
| 🔴 Red | ≥75% | Immediate intervention | Rajesh (87%), Meena phase 2 (88%) |

---

## 📱 API Endpoints (Real-Time)

### **Get Live Patient**
```bash
GET http://localhost:5000/api/patient/2
```
Returns: Current vitals, labs, risk score

### **Get All Patients**
```bash
GET http://localhost:5000/api/patients
```
Returns: All admitted patients with live data

### **Health Check**
```bash
GET http://localhost:5000/api/health
```
Returns: `{"status": "ok"}`

---

## 🔧 Configuration Quick Reference

### **Update Interval** (how often vitals change)
- File: `app/services/simulator.py` Line 34
- Current: `time.sleep(5)` = 5 seconds
- Change: Any number of seconds

### **Poll Interval** (how often dashboard updates)
- File: `app/templates/dashboard.html` Line 194
- Current: `}, 3000);` = 3 seconds
- Change: Any number in milliseconds

### **Alert Interval** (how often SMS sent)
- File: `app/services/simulator.py` Line 199
- Current: `>= 300` = 5 minutes
- Change: Any number in seconds

### **Risk Thresholds**
- File: `app/services/simulator.py` Lines 160-165
- High: `>= 0.75` (75%)
- Moderate: `0.40-0.75` (40-75%)
- Low: `< 0.40` (<40%)

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Vitals not changing | Refresh page (F5), check Flask logs |
| Audio not playing | Click dashboard first, check volume |
| Charts not updating | Click another tab, hard refresh (Ctrl+Shift+R) |
| Meena not deteriorating | Wait 75+ seconds, hard refresh |
| API errors | Verify `http://localhost:5000/api/patient/2` works |
| Server not running | `python run.py` in project folder |

---

## 📋 Demo Checklist

- [ ] Server running on http://localhost:5000
- [ ] Dashboard accessible
- [ ] 3 patients loaded (Rajesh, Meena, Anbu)
- [ ] Click Meena and watch vitals change
- [ ] Watch risk increase over time
- [ ] Hear audio alert at 75% threshold
- [ ] See alert logged in Active Alerts
- [ ] Try "Send Alert" button manually
- [ ] Check trend charts showing history
- [ ] Review contributing features for high-risk

---

## 📚 Learn More

- **Full Guide**: `REAL_TIME_GUIDE.md`
- **Setup**: `QUICK_START.md`
- **SMS Config**: `SMS_SETUP.md`
- **Deployment**: `DEPLOYMENT_SUMMARY.md`
- **Full Docs**: `README.md`

---

## 🎯 Next Steps

1. ✅ Understand real-time system (you are here)
2. ⏭️ Watch Meena's progression demo
3. ⏭️ Configure SMS alerts (SMS_SETUP.md)
4. ⏭️ Deploy to production (DEPLOYMENT_SUMMARY.md)
5. ⏭️ Integrate with hospital systems

---

**System**: ICU Sepsis Early Warning System | **Status**: OPERATIONAL ✓ | **Updated**: 2025
