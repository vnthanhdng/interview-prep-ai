import requests
import json

def test_interview_response():
    # API endpoint
    url = "http://localhost:8000/analyze"
    
    # Test cases
    test_responses = [
        {
            "name": "Complete STAR Response",
            "text": """When our team faced a tight deadline for a critical project, 
                      I was responsible for coordinating the final deliverables. 
                      I created a detailed timeline and held daily check-ins with team members. 
                      As a result, we delivered the project two days ahead of schedule."""
        },
        {
            "name": "Missing Result",
            "text": """In my previous role, we had a major system outage. 
                      I was tasked with restoring service as quickly as possible. 
                      I implemented a backup recovery protocol and worked with the infrastructure team."""
        },
        {
            "name": "Technical Response",
            "text": """While working on our e-commerce platform, I noticed significant performance issues. 
                      I needed to optimize the database queries to improve load times. 
                      I implemented database indexing and query caching, and also refactored the ORM queries. 
                      This resulted in a 60% reduction in page load time."""
        }
    ]

    # Test each response
    for test_case in test_responses:
        print(f"\n=== Testing: {test_case['name']} ===")
        try:
            response = requests.post(url, json={"text": test_case['text']})
            
            # Check if request was successful
            if response.status_code == 200:
                # Pretty print the results
                print("\nAnalysis Results:")
                results = response.json()
                print(json.dumps(results, indent=2))
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")

if __name__ == "__main__":
    test_interview_response()