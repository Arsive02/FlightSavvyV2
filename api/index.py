from http.server import BaseHTTPRequestHandler
from api.prediction import predict_best_time_to_buy_ticket
import json

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        # Get parameters
        origin = data.get('origin')
        destination = data.get('destination')
        granularity = data.get('granularity', 'month')
        future_year = data.get('futureYear')
        weeks_ahead = data.get('weeksAhead')
        start_month = data.get('start_month')
        end_month = data.get('end_month')
        carrier = data.get('carrier')
        
        # Convert month names to numbers if needed
        if isinstance(start_month, str) and not start_month.isdigit():
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            try:
                start_month = months.index(start_month) + 1
            except ValueError:
                pass
            
        if isinstance(end_month, str) and not end_month.isdigit():
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            try:
                end_month = months.index(end_month) + 1
            except ValueError:
                pass
                
        # Call prediction function
        try:
            result = predict_best_time_to_buy_ticket(
                origin=origin,
                destination=destination,
                granularity=granularity,
                future_year=future_year,
                weeks_ahead=weeks_ahead,
                start_month=start_month,
                end_month=end_month,
                carrier=carrier
            )
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            self.wfile.write(json.dumps(result).encode())
            return
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            self.wfile.write(json.dumps({
                'error': str(e),
                'success': False
            }).encode())
            return
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()