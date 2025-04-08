# FlightSavvy - AI Flight Price Prediction

![FlightSavvy Logo](https://i.ibb.co/nqyz2sb1/budget-small.png)

FlightSavvy is an advanced flight price prediction application that leverages machine learning algorithms to help travelers find the optimal time to book flights, saving money and reducing travel planning stress.

## 🚀 Features

- **Price Navigator**: Advanced AI analysis to find the optimal booking time for any route
- **Interactive Flight Assistant**: Conversational AI to answer flight-related questions and provide personalized recommendations
- **Comprehensive Analysis**: View price trends across different time periods (daily, weekly, monthly, quarterly)
- **Airline Comparison**: Compare pricing patterns across different carriers
- **Visual Analytics**: Interactive charts and visualizations of flight price trends
- **Student Discounts**: Special pricing information for students and other passenger types

## 💻 Technology Stack

### Frontend
- **React** with TypeScript for a robust UI experience
- **Recharts** for interactive data visualization
- **CSS** with custom "Cockpit Theme" for an immersive aviation-inspired interface

### Backend
- **Python** with Flask for the API server
- **Machine Learning** models for price prediction:
  - Random Forest for pattern recognition
  - Time Series analysis for seasonal trends

### AI Integration
- Conversational AI assistant for natural language flight queries
- Watson Assistant integration for advanced chatbot capabilities

## 🛠️ Installation & Setup

### Prerequisites
- Node.js (v14+)
- Python (v3.8+)
- pip (for Python packages)

### Frontend Setup
```bash
# Clone the repository
git clone https://github.com/Arsive02/FlightSavvyV2.git
cd flight_price_predictor

# Install dependencies
npm install

# Start the development server
npm run dev
```

### Backend Setup
Download files from https://huggingface.co/spaces/Arsive/flight-fare-hf/tree/main

Run the Docker container:
```bash
# Build and run the Docker container
docker build -t flightsavvy-backend .
docker run -p 7860:7860 flightsavvy-backend
```

## 📝 API Documentation

### Flight Price Prediction Endpoint

```
POST /api/predict
```

#### Request Body
```json
{
  "origin": "String",
  "destination": "String",
  "granularity": "String (date|week|month|quarter)",
  "futureYear": "Number",
  "weeksAhead": "Number (for date granularity)",
  "start_month": "Number (1-12)",
  "end_month": "Number (1-12)",
  "carrier": "String (optional)"
}
```

#### Response
```json
{
  "route": "String",
  "best_time": {
    "predicted_fare": "Number",
    "date|month|quarter": "Value"
  },
  "formatted_best_time": "String",
  "chart_data": { /* Data for visualization */ },
  "success": "Boolean"
}
```

## 🧠 How It Works

FlightSavvy uses a combination of historical flight data analysis and machine learning models to predict future flight prices. The system considers:

1. **Route Factors**: Distance, competition, popularity
2. **Temporal Patterns**: Seasonal trends, day-of-week effects, holiday periods
3. **Carrier Variables**: Different pricing strategies by airline
4. **Market Dynamics**: Supply and demand fluctuations

The machine learning pipeline includes:
- Feature engineering to extract relevant patterns
- Random Forest model for complex pattern recognition
- Time series analysis for trend identification
- Ensemble approach for improved prediction accuracy

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Flight data sources and partners
- Open-source libraries and frameworks
- AI and ML research communities

---

**FlightSavvy** - Powered by AI ✈️ © 2025