# 🧠 AI-Based Personal Finance Manager

## 📌 Abstract
Managing personal finances effectively is a challenge in today’s fast-paced and financially complex world.  
The **AI-Based Personal Finance Manager** leverages Artificial Intelligence (AI) and Machine Learning (ML) to automate budgeting, track spending, analyze financial habits, and provide tailored financial advice.  

By integrating data from multiple sources (bank accounts, credit cards, and investment portfolios), the system offers:
- Real-time insights  
- Predictive analytics  
- Personalized recommendations  

This empowers users to make smarter financial decisions and plan for the long term.

---

## ⚡ Existing System
Current finance management tools rely heavily on:
- Manual inputs (spreadsheets, basic apps)  
- Limited automation (simple categorization of transactions)  
- No meaningful guidance or predictive insights  

These systems lack adaptability, personalization, and proactive financial planning.

---

## 🚀 Proposed System
The proposed system integrates **AI & ML models** to:
- Analyze past transactions and spending patterns  
- Predict future financial behaviors  
- Provide **personalized financial advice**  
- Send **real-time spending alerts**  
- Offer **smart budgeting & investment suggestions**  

Additional features:
- Automatic syncing with bank accounts, credit cards, and investments  
- Natural Language Processing (NLP) for chatbot-based interaction  
- Predictive analytics to warn about cash flow issues, upcoming expenses, or opportunities  


## 🖥️ System Requirements

### Hardware
- Processor: Pentium IV 2.4 GHz or higher  
- Hard Disk: 40 GB  
- RAM: 512 MB (minimum)  

### Software
- Operating System: Windows  
- Language: Python  
- Framework: Django  
- Dependencies: Listed in `requirements.txt`  

---

## 📂 Project Structure
---
AI-Based-Personal-Finance-Manager/
│
├── Finance/ # Main Django project (settings, urls, wsgi)
├── FinanceApp/ # App (models, views, static, templates)
├── DatasetLink # Dataset link file
├── manage.py
├── requirements.txt
├── runServer.bat # Optional script to run server
├── SCREENS.docx # Screenshots / documentation
└── README.md

## ▶️ How to Run the Project
1. Clone the repository:
```bash
   git clone https://github.com/your-username/AI-Based-Personal-Finance-Manager.git
   cd AI-Based-Personal-Finance-Manager
```
2.Create and activate a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate  # On Linux/Mac
```

3.Install dependencies:
```bash
pip install -r requirements.txt
```

4.Run migrations:
```bash
python manage.py migrate
```
5.Start the server:
```bash
python manage.py runserver
```

6.Open your browser at:
```bash
http://127.0.0.1:8000/
```

###✨ Features
---
✅ AI-powered budgeting & recommendations
✅ Real-time spending alerts
✅ Predictive analytics for financial planning
✅ Chat-based interaction with NLP
✅ Consolidated financial dashboard

###📌 Future Enhancements
---
->Support for multi-currency accounts
->Mobile application version
->Integration with investment APIs
->Advanced visualization dashboards

👩‍💻 Authors
---
Mareddy Bhuvana Kruthi

---
