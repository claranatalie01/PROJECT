import requests
import json

task = {
    "id": "test-001",
    "type": "RAG_RETRIEVE",
    "role": "retriever",
    "input": {
        "query": {
                
            "month": 1,
            "income": 0.5,
            "source": "INTERNET",
            "device_os": "windows",
            "velocity_4w": 5313.605189761942,
            "velocity_6h": 3513.5461220118077,
            "customer_age": 50,
            "payment_type": "AC",
            "velocity_24h": 4656.066201712021,
            "zip_count_4w": 1209,
            "email_is_free": 1,
            "housing_status": "BA",
            "foreign_request": 0,
            "has_other_cards": 0,
            "phone_home_valid": 0,
            "bank_months_count": -1,
            "credit_risk_score": 257,
            "employment_status": "CC",
            "days_since_request": 1.3104234868826556,
            "device_fraud_count": 0,
            "keep_alive_session": 0,
            "phone_mobile_valid": 1,
            "bank_branch_count_8w": 1,
            "name_email_similarity": 0.1023998325466471,
            "proposed_credit_limit": 1500.0,
            "intended_balcon_amount": -0.7177080059889509,
            "device_distinct_emails_8w": 1,
            "prev_address_months_count": -1,
            "session_length_in_minutes": 34.86797228018343,
            "current_address_months_count": 235,
            "date_of_birth_distinct_emails_4w": 1
            } # your application dict
    }   
}


resp = requests.post("http://localhost:5001/agent/retriever", json=task)
print(json.dumps(resp.json(), indent=2))