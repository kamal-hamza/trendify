#!/usr/bin/env python
"""
Test script for Trendify API endpoints
Tests all major endpoints to ensure RSS feed API is working correctly
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000/api"
COLORS = {
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'BLUE': '\033[94m',
    'YELLOW': '\033[93m',
    'END': '\033[0m'
}


def print_header(text):
    """Print formatted header"""
    print(f"\n{COLORS['BLUE']}{'='*60}{COLORS['END']}")
    print(f"{COLORS['BLUE']}{text}{COLORS['END']}")
    print(f"{COLORS['BLUE']}{'='*60}{COLORS['END']}")


def print_success(text):
    """Print success message"""
    print(f"{COLORS['GREEN']}✓ {text}{COLORS['END']}")


def print_error(text):
    """Print error message"""
    print(f"{COLORS['RED']}✗ {text}{COLORS['END']}")


def print_info(text):
    """Print info message"""
    print(f"{COLORS['YELLOW']}→ {text}{COLORS['END']}")


def test_endpoint(name, url, method='GET', data=None, params=None):
    """Test a single endpoint"""
    print(f"\nTesting: {name}")
    print(f"URL: {url}")
    
    try:
        if method == 'GET':
            response = requests.get(url, params=params, timeout=10)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=10)
        else:
            print_error(f"Unknown method: {method}")
            return False
        
        print(f"Status: {response.status_code}")
        
        if response.status_code in [200, 201]:
            print_success(f"{name} - PASSED")
            
            # Pretty print first few items if it's a list
            try:
                json_data = response.json()
                if isinstance(json_data, dict):
                    if 'results' in json_data:
                        print(f"  → Results count: {len(json_data['results'])}")
                        if json_data['results']:
                            print(f"  → First result keys: {list(json_data['results'][0].keys())[:5]}")
                    elif 'count' in json_data:
                        print(f"  → Total count: {json_data['count']}")
                    else:
                        print(f"  → Response keys: {list(json_data.keys())[:10]}")
                elif isinstance(json_data, list):
                    print(f"  → Items count: {len(json_data)}")
                    if json_data:
                        print(f"  → First item keys: {list(json_data[0].keys())[:5]}")
            except:
                pass
            
            return True
        else:
            print_error(f"{name} - FAILED")
            print(f"  → Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error(f"{name} - Connection Error (Is the server running?)")
        return False
    except requests.exceptions.Timeout:
        print_error(f"{name} - Timeout")
        return False
    except Exception as e:
        print_error(f"{name} - Exception: {str(e)}")
        return False


def main():
    """Run all API tests"""
    print_header("Trendify API Test Suite")
    print(f"Testing API at: {BASE_URL}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: API Root (if it exists)
    print_header("1. Basic Connectivity")
    results.append(test_endpoint(
        "API Root",
        BASE_URL + "/"
    ))
    
    # Test 2: Topics Endpoints
    print_header("2. Topics Endpoints")
    
    results.append(test_endpoint(
        "List Topics",
        BASE_URL + "/topics/"
    ))
    
    results.append(test_endpoint(
        "Topics - Filter by Category",
        BASE_URL + "/topics/",
        params={"category": "LLM"}
    ))
    
    results.append(test_endpoint(
        "Topics - Search",
        BASE_URL + "/topics/",
        params={"search": "AI"}
    ))
    
    results.append(test_endpoint(
        "Trending Topics",
        BASE_URL + "/topics/trending/"
    ))
    
    results.append(test_endpoint(
        "Trending Topics (7 days, min momentum)",
        BASE_URL + "/topics/trending/",
        params={"days": 7, "min_momentum": 0, "limit": 10}
    ))
    
    results.append(test_endpoint(
        "Topic Categories",
        BASE_URL + "/topics/categories/"
    ))
    
    # Test 3: Posts Endpoints
    print_header("3. Posts Endpoints")
    
    results.append(test_endpoint(
        "List Posts",
        BASE_URL + "/posts/"
    ))
    
    results.append(test_endpoint(
        "Posts - Filter by Source",
        BASE_URL + "/posts/",
        params={"source": "HN"}
    ))
    
    results.append(test_endpoint(
        "Posts - Last 7 Days",
        BASE_URL + "/posts/",
        params={"days": 7}
    ))
    
    results.append(test_endpoint(
        "Posts - Min Engagement",
        BASE_URL + "/posts/",
        params={"min_engagement": 100}
    ))
    
    results.append(test_endpoint(
        "Posts - Positive Sentiment",
        BASE_URL + "/posts/",
        params={"sentiment": "positive"}
    ))
    
    results.append(test_endpoint(
        "Top Posts",
        BASE_URL + "/posts/top/",
        params={"limit": 10, "days": 7}
    ))
    
    results.append(test_endpoint(
        "Posts Feed",
        BASE_URL + "/posts/feed/",
        params={"limit": 20}
    ))
    
    # Test 4: Entities Endpoints
    print_header("4. Entities Endpoints")
    
    results.append(test_endpoint(
        "List Entities",
        BASE_URL + "/entities/"
    ))
    
    results.append(test_endpoint(
        "Entities - Filter by Type",
        BASE_URL + "/entities/",
        params={"entity_type": "LLM_FAMILY"}
    ))
    
    results.append(test_endpoint(
        "Entity Nodes",
        BASE_URL + "/entity-nodes/"
    ))
    
    # Test 5: Metrics Endpoints
    print_header("5. Metrics Endpoints")
    
    results.append(test_endpoint(
        "Daily Metrics",
        BASE_URL + "/metrics/"
    ))
    
    results.append(test_endpoint(
        "Metrics Heatmap",
        BASE_URL + "/metrics/heatmap/",
        params={"days": 30, "limit": 10}
    ))
    
    # Test 6: Statistics Endpoints
    print_header("6. Statistics Endpoints")
    
    results.append(test_endpoint(
        "Feed Statistics Overview",
        BASE_URL + "/stats/overview/",
        params={"days": 7}
    ))
    
    results.append(test_endpoint(
        "Sentiment Analysis",
        BASE_URL + "/stats/sentiment_analysis/",
        params={"days": 7}
    ))
    
    # Test 7: API Documentation
    print_header("7. API Documentation")
    
    results.append(test_endpoint(
        "OpenAPI Schema",
        "http://localhost:8000/api/schema/"
    ))
    
    # Test 8: Data Refresh (POST endpoint - only test if safe)
    print_header("8. Data Management Endpoints")
    print_info("Skipping POST /stats/refresh_data/ to avoid triggering actual data fetch")
    print_info("Skipping POST /stats/recalculate_metrics/ to avoid triggering actual calculation")
    print_info("These endpoints exist but require manual testing")
    
    # Summary
    print_header("Test Summary")
    passed = sum(results)
    total = len(results)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {COLORS['GREEN']}{passed}{COLORS['END']}")
    print(f"Failed: {COLORS['RED']}{total - passed}{COLORS['END']}")
    print(f"Success Rate: {percentage:.1f}%")
    
    if passed == total:
        print_success("\nAll tests passed! ✨")
    elif passed > 0:
        print(f"\n{COLORS['YELLOW']}Some tests failed. Check the output above.{COLORS['END']}")
    else:
        print_error("\nAll tests failed. Is the server running?")
    
    print_header("Additional Information")
    print("\nAvailable Endpoints:")
    print("  • GET  /api/topics/                    - List all topics")
    print("  • GET  /api/topics/{id}/               - Get topic details")
    print("  • GET  /api/topics/trending/           - Get trending topics")
    print("  • GET  /api/topics/{id}/timeline/      - Get topic timeline")
    print("  • GET  /api/topics/categories/         - Get topic categories")
    print("  • GET  /api/posts/                     - List all posts")
    print("  • GET  /api/posts/{id}/                - Get post details")
    print("  • GET  /api/posts/top/                 - Get top posts")
    print("  • GET  /api/posts/feed/                - Get personalized feed")
    print("  • GET  /api/entities/                  - List entities")
    print("  • GET  /api/entity-nodes/              - List entity nodes")
    print("  • GET  /api/metrics/                   - List daily metrics")
    print("  • GET  /api/metrics/heatmap/           - Get metrics heatmap")
    print("  • GET  /api/stats/overview/            - Get overall statistics")
    print("  • GET  /api/stats/sentiment_analysis/  - Get sentiment breakdown")
    print("  • POST /api/stats/refresh_data/        - Trigger data refresh")
    print("  • POST /api/stats/recalculate_metrics/ - Recalculate metrics")
    print("\nAPI Documentation:")
    print("  • Swagger UI: http://localhost:8000/api/docs/")
    print("  • ReDoc:      http://localhost:8000/api/redoc/")
    print("  • Schema:     http://localhost:8000/api/schema/")
    
    print(f"\n{COLORS['BLUE']}{'='*60}{COLORS['END']}")
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()