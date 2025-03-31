import React, { useState } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
  Cell
} from 'recharts';

// City and airport data from the training dataset
const CITIES = [
  "Allentown/Bethlehem/Easton, PA", "Albuquerque, NM", "Nantucket, MA",
  "Colorado Springs, CO", "Dallas/Fort Worth, TX", "Pittsburgh, PA",
  "Huntsville, AL", "Albany, NY", "Amarillo, TX", "Denver, CO",
  "Atlanta, GA (Metropolitan Area)", "Austin, TX", "Asheville, NC",
  "Tucson, AZ", "Phoenix, AZ", "Hartford, CT", "Seattle, WA", "Birmingham, AL",
  "El Paso, TX", "Cleveland, OH (Metropolitan Area)", "Nashville, TN",
  "Boise, ID", "Boston, MA (Metropolitan Area)", "Burlington, VT",
  "Buffalo, NY", "Bozeman, MT", "Washington, DC (Metropolitan Area)",
  "Chicago, IL", "Charleston, SC", "Cedar Rapids/Iowa City, IA",
  "Charlotte, NC", "Columbus, OH", "St. Louis, MO", "Myrtle Beach, SC",
  "Jacksonville, FL", "Detroit, MI", "Des Moines, IA", "Houston, TX",
  "Orlando, FL", "Panama City, FL", "Valparaiso, FL", "Eugene, OR",
  "Key West, FL", "Fargo, ND", "Kalispell, MT", "Minneapolis/St. Paul, MN",
  "New York City, NY (Metropolitan Area)", "Fort Myers, FL",
  "Greenville/Spartanburg, SC", "Grand Rapids, MI",
  "Greensboro/High Point, NC", "Las Vegas, NV", "Indianapolis, IN",
  "Jackson, WY", "Jackson/Vicksburg, MS",
  "San Francisco, CA (Metropolitan Area)", "Miami, FL (Metropolitan Area)",
  "Los Angeles, CA (Metropolitan Area)", "Little Rock, AR", "Louisville, KY",
  "Cincinnati, OH", "Sacramento, CA", "Tampa, FL (Metropolitan Area)",
  "Kansas City, MO", "San Antonio, TX", "Memphis, TN", "Omaha, NE",
  "Milwaukee, WI", "Madison, WI", "New Orleans, LA", "Martha's Vineyard, MA",
  "San Diego, CA", "Norfolk, VA (Metropolitan Area)", "Pensacola, FL",
  "Oklahoma City, OK", "Portland, OR", "Philadelphia, PA", "Palm Springs, CA",
  "Portland, ME", "Bend/Redmond, OR", "Raleigh/Durham, NC", "Reno, NV",
  "Rochester, NY", "Salt Lake City, UT", "Savannah, GA", "Springfield, MO",
  "Sarasota/Bradenton, FL", "Syracuse, NY", "Knoxville, TN",
  "Bismarck/Mandan, ND", "Columbia, SC", "Fresno, CA", "Fayetteville, AR",
  "Richmond, VA", "Atlantic City, NJ", "Aspen, CO", "Appleton, WI",
  "Bangor, ME"
].sort();

// Major airports data
const AIRPORTS = [
  { code: "ABE", city: "Allentown/Bethlehem/Easton, PA" },
  { code: "ABQ", city: "Albuquerque, NM" },
  { code: "ACK", city: "Nantucket, MA" },
  { code: "ATL", city: "Atlanta, GA (Metropolitan Area)" },
  { code: "AUS", city: "Austin, TX" },
  { code: "BDL", city: "Hartford, CT" },
  { code: "BNA", city: "Nashville, TN" },
  { code: "BOS", city: "Boston, MA (Metropolitan Area)" },
  { code: "CLT", city: "Charlotte, NC" },
  { code: "DCA", city: "Washington, DC (Metropolitan Area)" },
  { code: "DEN", city: "Denver, CO" },
  { code: "DFW", city: "Dallas/Fort Worth, TX" },
  { code: "DTW", city: "Detroit, MI" },
  { code: "EWR", city: "New York City, NY (Metropolitan Area)" },
  { code: "IAD", city: "Washington, DC (Metropolitan Area)" },
  { code: "IAH", city: "Houston, TX" },
  { code: "JFK", city: "New York City, NY (Metropolitan Area)" },
  { code: "LAS", city: "Las Vegas, NV" },
  { code: "LAX", city: "Los Angeles, CA (Metropolitan Area)" },
  { code: "LGA", city: "New York City, NY (Metropolitan Area)" },
  { code: "MCO", city: "Orlando, FL" },
  { code: "MIA", city: "Miami, FL (Metropolitan Area)" },
  { code: "MSP", city: "Minneapolis/St. Paul, MN" },
  { code: "ORD", city: "Chicago, IL" },
  { code: "PHL", city: "Philadelphia, PA" },
  { code: "PHX", city: "Phoenix, AZ" },
  { code: "PIT", city: "Pittsburgh, PA" },
  { code: "SAN", city: "San Diego, CA" },
  { code: "SEA", city: "Seattle, WA" },
  { code: "SFO", city: "San Francisco, CA (Metropolitan Area)" },
  { code: "SLC", city: "Salt Lake City, UT" }
].sort((a, b) => a.code.localeCompare(b.code));

// Major airlines
const AIRLINES = [
  { code: "G4", name: "Allegiant Air" },
  { code: "DL", name: "Delta Air Lines" },
  { code: "WN", name: "Southwest Airlines" },
  { code: "AA", name: "American Airlines" },
  { code: "UA", name: "United Airlines" },
  { code: "B6", name: "JetBlue Airways" },
  { code: "AS", name: "Alaska Airlines" },
  { code: "F9", name: "Frontier Airlines" },
  { code: "NK", name: "Spirit Airlines" },
  { code: "SY", name: "Sun Country Airlines" }
].sort((a, b) => a.name.localeCompare(b.name));

// Passenger types
const PASSENGER_TYPES = [
  { value: "adult", label: "Adult" },
  { value: "child", label: "Child (2-11)" },
  { value: "infant", label: "Infant (under 2)" },
  { value: "student", label: "Student" }
];

interface SearchFormData {
  origin: string;
  originAirport: string;
  destination: string;
  destinationAirport: string;
  departureStartMonth: string;
  departureEndMonth: string;
  departureYear: number;
  analysisType: 'quarter' | 'month' | 'week' | 'date';
  weeksAhead: number;
  passengerType: string;
  carrier: string;
}

interface ChartData {
  type: string;
  data: any[];
  xAxisKey: string;
  yAxisKey: string;
  xAxisLabel: string;
  yAxisLabel: string;
  title: string;
}

interface PredictionResult {
  id: number;
  route: string;
  origin: string;
  destination: string;
  price: number;
  bestTime: string;
  predictedSavings: string;
  priceHistory: number[];
  chart_data: ChartData;
  full_analysis_chart_data: ChartData;
  analysisType: 'quarter' | 'month' | 'week' | 'date';
  travelPeriod: string;
  departureYear: number;
  carrier?: string;
  carrier_name?: string;
  rawData?: any;  // Original API response data
}

// Custom tooltip component for charts
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    const dataPoint = payload[0].payload;
    return (
      <div className="custom-tooltip">
        <p className="label">{label}</p>
        <p className="info">Fare: ${dataPoint.fare}</p>
        {dataPoint.isHoliday && <p className="holiday">Holiday Period</p>}
        {dataPoint.isBest && <p className="best">Best Price!</p>}
        {dataPoint.inTravelPeriod === false && <p className="outside">Outside Travel Period</p>}
      </div>
    );
  }
  return null;
};

// Flight price chart component
const FlightPriceChart: React.FC<{ chartData: ChartData }> = ({ chartData }) => {
  if (!chartData || !chartData.data || chartData.data.length === 0) {
    return <div>No chart data available</div>;
  }

  // Render line chart for date granularity
  if (chartData.type === 'line') {
    // Format dates for the x-axis
    const formatXAxis = (dateStr: string) => {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    };

    return (
      <div className="chart-container">
        <h3>{chartData.title}</h3>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart
            data={chartData.data}
            margin={{ top: 20, right: 30, left: 20, bottom: 50 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey={chartData.xAxisKey} 
              label={{ value: chartData.xAxisLabel, position: 'insideBottom', offset: -15 }}
              tickFormatter={formatXAxis}
              interval="preserveStartEnd"
            />
            <YAxis 
              label={{ value: chartData.yAxisLabel, angle: -90, position: 'insideLeft' }}
              domain={['auto', 'auto']}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey={chartData.yAxisKey} 
              name="Fare" 
              stroke="#5090d3" 
              dot={(props: any) => {
                const { cx, cy, payload } = props;
                
                // Holiday period dots in red
                if (payload.isHoliday) {
                  return <circle cx={cx} cy={cy} r={4} fill="#ff5252" />;
                }
                
                // Best price dot in green and larger
                if (payload.isBest) {
                  return <circle cx={cx} cy={cy} r={6} fill="#4caf50" stroke="#fff" strokeWidth={2} />;
                }
                
                // Regular dots
                return <circle cx={cx} cy={cy} r={3} fill="#5090d3" />;
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }
  
  // Render bar chart for week/month/quarter granularity
  return (
    <div className="chart-container">
      <h3>{chartData.title}</h3>
      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={chartData.data}
          margin={{ top: 20, right: 30, left: 20, bottom: 50 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey={chartData.xAxisKey}
            label={{ value: chartData.xAxisLabel, position: 'insideBottom', offset: -15 }}
          />
          <YAxis 
            label={{ value: chartData.yAxisLabel, angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Bar 
            dataKey={chartData.yAxisKey} 
            name="Fare"
            fill="#5090d3"
            fillOpacity={0.8}
            stroke="#000"
            strokeWidth={1}
            isAnimationActive={true}
            animationDuration={300}
          >
            {chartData.data.map((entry: any, index: number) => {
              // In full analysis chart, color by travel period and best
              let color = '#5090d3'; // default color
              if ('inTravelPeriod' in entry) {
                if (entry.isBest) color = '#4caf50'; // Best time (green)
                else color = entry.inTravelPeriod ? '#5090d3' : '#e0e0e0'; // In travel period (blue) or not (gray)
              } else {
                // Regular chart - highlight best time
                color = entry.isBest ? '#4caf50' : '#5090d3';
              }
              
              return <Cell key={`cell-${index}`} fill={color} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

// API service function
const predictBestTime = async (params: {
  origin: string;
  destination: string;
  granularity: 'quarter' | 'month' | 'week' | 'date';
  futureYear?: number;
  weeksAhead?: number;
  start_month?: number;
  end_month?: number;
  carrier?: string;
}) => {
  try {
    const response = await fetch('http://localhost:5000/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API call failed:', error);
    throw error;
  }
};

// Modal component for full analysis
const FullAnalysisModal = ({ isOpen, onClose, result }: { 
  isOpen: boolean; 
  onClose: () => void; 
  result: PredictionResult | null;
}) => {
  if (!isOpen || !result) return null;

  // Calculate lowest and average prices
  const calculateAverageSavings = () => {
    if (!result.rawData || !result.rawData.filtered_predictions) return { avgPrice: 0, savingAmount: 0 };
    
    const prices = result.rawData.filtered_predictions.map((p: any) => p.predicted_fare);
    const avgPrice = prices.reduce((a: number, b: number) => a + b, 0) / prices.length;
    const savingAmount = avgPrice - result.price;
    
    return { avgPrice: avgPrice.toFixed(2), savingAmount: savingAmount.toFixed(2) };
  };
  
  const { avgPrice, savingAmount } = calculateAverageSavings();

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Flight Deck Analysis: {result.origin} to {result.destination} ({result.departureYear})</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>
        
        <div className="modal-body">
          <div className="cockpit-display">
            <div className="main-display">
              <div className="panel-title">Full Route Analysis</div>
              <div className="visualization-container">
                {result.full_analysis_chart_data ? (
                  <FlightPriceChart chartData={result.full_analysis_chart_data} />
                ) : (
                  <div className="no-data-message">
                    Full analysis visualization not available
                  </div>
                )}
              </div>
            </div>
            
            <div className="secondary-displays">
              <div className="instrument-panel">
                <div className="panel-title">Optimal Booking Window</div>
                <div className="instrument-content">
                  <div className="price-gauge">
                    <div className="gauge-label">BEST TIME TO BOOK</div>
                    <div className="gauge-value">{result.bestTime}</div>
                  </div>
                </div>
              </div>
              
              <div className="instrument-panel">
                <div className="panel-title">Price Analysis</div>
                <div className="instrument-content">
                  <div className="price-gauge">
                    <div className="gauge-label">OPTIMAL FARE</div>
                    <div className="gauge-value">${result.price.toFixed(2)}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div className="status-indicators">
            <div className="status-item">
              <span className="status-label">Travel Period</span>
              <span className="status-value">{result.departureYear} - {result.travelPeriod}</span>
            </div>
            
            <div className="status-item">
              <span className="status-label">Predicted Savings</span>
              <span className="status-value good">{result.predictedSavings}</span>
            </div>
            
            <div className="status-item">
              <span className="status-label">Average Price</span>
              <span className="status-value">${avgPrice}</span>
            </div>
            
            <div className="status-item">
              <span className="status-label">Potential Savings</span>
              <span className="status-value good">${savingAmount}</span>
            </div>

            {result.carrier && (
              <div className="status-item">
                <span className="status-label">Selected Airline</span>
                <span className="status-value highlight">{result.carrier_name || result.carrier}</span>
              </div>
            )}
          </div>
          
          <div className="recommendations">
            <h3>Flight Navigator Recommendations</h3>
            <p>
              <strong>✈️ Optimal Booking Strategy:</strong> Based on our analysis of historical data and price trends, we recommend booking your {result.origin} to {result.destination} flight during <strong>{result.bestTime}</strong> to secure the best fare of <strong>${result.price.toFixed(2)}</strong>.
            </p>
            <p>
              <strong>⏰ Timing:</strong> Start monitoring prices approximately 
              {result.analysisType === 'month' ? ` 2-3 months` : result.analysisType === 'quarter' ? ` 3-4 months` : ` 4-6 weeks`} before 
              your planned travel dates. Set a price alert to receive notifications when fares drop below average.
            </p>
            <p>
              <strong>🗓️ Day Selection:</strong> When possible, choose mid-week departures (Tuesday or Wednesday) as these typically offer 10-20% lower fares than weekend flights. If your travel dates are flexible, consider adjusting by a day or two for significant savings.
            </p>
            {result.carrier && (
              <p>
                <strong>🏢 Carrier Notes:</strong> {result.carrier_name} often {
                  result.carrier === 'WN' ? 'offers more generous baggage policies but may have limited availability.' :
                  result.carrier === 'AA' || result.carrier === 'DL' || result.carrier === 'UA' ? 'tends to be priced at a premium but may offer more route options and better service.' :
                  result.carrier === 'NK' || result.carrier === 'F9' || result.carrier === 'G4' ? 'offers the lowest base fares but charges extra for most add-ons like baggage and seat selection.' :
                  'has unique pricing patterns that our system has analyzed for this route.'
                }
              </p>
            )}
          </div>
        </div>
        
        <div className="modal-footer">
          <button className="action-button secondary" onClick={onClose}>Close Flight Deck</button>
          <button className="action-button primary">Set Price Alert</button>
        </div>
      </div>
    </div>
  );
};

const FlightSearch: React.FC = () => {
  const currentYear = new Date().getFullYear();
  const years = [currentYear, currentYear + 1, currentYear + 2, currentYear + 3, currentYear + 4];

  const [searchData, setSearchData] = useState<SearchFormData>({
    origin: '',
    originAirport: '',
    destination: '',
    destinationAirport: '',
    departureStartMonth: '',
    departureEndMonth: '',
    departureYear: currentYear,
    analysisType: 'month', // Changed default to monthly since it's most useful
    weeksAhead: 8,
    passengerType: 'adult',
    carrier: ''
  });

  const [searchResults, setSearchResults] = useState<PredictionResult[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedResult, setSelectedResult] = useState<PredictionResult | null>(null);

  const months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  // Helper function to get month index (0-11)
  const getMonthIndex = (monthName: string): number => {
    return months.findIndex(month => month === monthName);
  };

  // Handle origin city selection
  const handleOriginCityChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const city = e.target.value;
    setSearchData(prev => ({
      ...prev,
      origin: city,
      // Clear airport if city changes
      originAirport: ''
    }));
  };

  // Handle origin airport selection
  const handleOriginAirportChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSearchData(prev => ({
      ...prev,
      originAirport: e.target.value
    }));
  };

  // Handle destination city selection
  const handleDestinationCityChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const city = e.target.value;
    setSearchData(prev => ({
      ...prev,
      destination: city,
      // Clear airport if city changes
      destinationAirport: ''
    }));
  };

  // Handle destination airport selection
  const handleDestinationAirportChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSearchData(prev => ({
      ...prev,
      destinationAirport: e.target.value
    }));
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    
    try {
      // Get month numbers (1-12)
      const start_month = getMonthIndex(searchData.departureStartMonth) + 1;
      const end_month = getMonthIndex(searchData.departureEndMonth) + 1;
      
      // Use airport code if selected, otherwise use city
      const origin = searchData.originAirport || searchData.origin;
      const destination = searchData.destinationAirport || searchData.destination;

      // Call the API
      const response = await predictBestTime({
        origin: origin,
        destination: destination,
        granularity: searchData.analysisType,
        futureYear: searchData.departureYear,
        weeksAhead: searchData.analysisType === 'date' ? searchData.weeksAhead : undefined,
        start_month: start_month,
        end_month: end_month,
        carrier: searchData.carrier || undefined
      });
      
      if (response.success) {
        // Create result object
        const result: PredictionResult = {
          id: 1,
          route: response.route,
          origin: origin,
          destination: destination,
          price: response.best_time.predicted_fare,
          bestTime: response.formatted_best_time,
          predictedSavings: calculateSavings(response),
          priceHistory: extractPriceHistory(response),
          chart_data: response.chart_data,
          full_analysis_chart_data: response.full_analysis_chart_data,
          analysisType: searchData.analysisType,
          travelPeriod: `${searchData.departureStartMonth} to ${searchData.departureEndMonth}`,
          departureYear: searchData.departureYear,
          carrier: response.carrier,
          carrier_name: response.carrier_name,
          rawData: response  // Store full response for debugging
        };
        
        setSearchResults([result]);
        setSelectedResult(result);
      } else {
        setError(response.error || 'An unknown error occurred');
      }
    } catch (error) {
      console.error('Error during API call:', error);
      setError('Failed to get prediction. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Helper function to calculate savings
  const calculateSavings = (response: any) => {
    if (!response.filtered_predictions || response.filtered_predictions.length === 0) {
      return "Calculating...";
    }
    
    // Calculate average price across all predictions
    const prices = response.filtered_predictions.map((p: any) => p.predicted_fare);
    const avgPrice = prices.reduce((a: number, b: number) => a + b, 0) / prices.length;
    
    // Calculate savings percentage
    const bestPrice = response.best_time.predicted_fare;
    const savingsPercent = ((avgPrice - bestPrice) / avgPrice * 100).toFixed(0);
    
    return `${savingsPercent}% below average`;
  };
  
  // Helper function to extract price history from response
  const extractPriceHistory = (response: any) => {
    if (!response.filtered_predictions || response.filtered_predictions.length === 0) {
      return [350, 340, 330, 320, 310, 320, 330];
    }
    
    // Return the predicted fares from filtered predictions
    return response.filtered_predictions.map((p: any) => p.predicted_fare);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    
    if (name === 'weeksAhead') {
      setSearchData(prev => ({
        ...prev,
        [name]: parseInt(value)
      }));
    } else {
      setSearchData(prev => ({
        ...prev,
        [name]: value
      }));
    }
  };

  // Get filtered airports for the selected city
  const getFilteredAirports = (cityName: string) => {
    return AIRPORTS.filter(airport => airport.city === cityName || !cityName);
  };

  // Generate label for analysis type
  const getAnalysisTypeLabel = (type: string) => {
    switch(type) {
      case 'quarter': return 'Quarterly Flight Analysis';
      case 'month': return 'Monthly Flight Analysis';
      case 'week': return 'Weekly Flight Analysis';
      case 'date': return 'Daily Flight Analysis';
      default: return 'Flight Analysis';
    }
  };
  
  // Open modal with selected result
  const openFullAnalysis = (result: PredictionResult) => {
    setSelectedResult(result);
    setIsModalOpen(true);
  };

  // Calculate approximate student discount
  const getStudentDiscount = () => {
    if (searchData.passengerType === 'student') {
      return 'Students may receive 10-15% discount depending on airline and destination';
    }
    return null;
  };

  const studentDiscountNote = getStudentDiscount();

  return (
    <div className="flight-search">
      <div className="search-header">
        <h2>Flight Price Navigator</h2>
        <p>Advanced AI flight analysis to pinpoint the optimal time to book your journey</p>
      </div>

      <div className="search-container w-full box-border" style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}>
        <form onSubmit={handleSearch} className="search-form">
          <div className="form-grid" style={{ width: '100%', maxWidth: '100%' }}>
            {/* Origin City Selection */}
            <div className="form-group" style={{ width: '100%' }}>
              <label htmlFor="origin">Departure City</label>
              <select
                id="origin"
                name="origin"
                value={searchData.origin}
                onChange={handleOriginCityChange}
                required
                className="w-full max-w-full box-border"
                style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
              >
                <option value="">Select City</option>
                {CITIES.map(city => (
                  <option key={`origin-${city}`} value={city}>{city}</option>
                ))}
              </select>
            </div>

            {/* Origin Airport Selection */}
            <div className="form-group" style={{ width: '100%' }}>
              <label htmlFor="originAirport">Departure Airport</label>
              <select
                id="originAirport"
                name="originAirport"
                value={searchData.originAirport}
                onChange={handleOriginAirportChange}
                disabled={!searchData.origin}
                className="w-full max-w-full box-border"
                style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
              >
                <option value="">All Airports</option>
                {getFilteredAirports(searchData.origin).map(airport => (
                  <option key={`origin-airport-${airport.code}`} value={airport.code}>
                    {airport.code} - {airport.city}
                  </option>
                ))}
              </select>
            </div>

            {/* Destination City Selection */}
            <div className="form-group" style={{ width: '100%' }}>
              <label htmlFor="destination">Arrival City</label>
              <select
                id="destination"
                name="destination"
                value={searchData.destination}
                onChange={handleDestinationCityChange}
                required
                className="w-full max-w-full box-border"
                style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
              >
                <option value="">Select City</option>
                {CITIES.map(city => (
                  <option key={`dest-${city}`} value={city}>{city}</option>
                ))}
              </select>
            </div>

            {/* Destination Airport Selection */}
            <div className="form-group" style={{ width: '100%' }}>
              <label htmlFor="destinationAirport">Arrival Airport</label>
              <select
                id="destinationAirport"
                name="destinationAirport"
                value={searchData.destinationAirport}
                onChange={handleDestinationAirportChange}
                disabled={!searchData.destination}
                className="w-full max-w-full box-border"
                style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
              >
                <option value="">All Airports</option>
                {getFilteredAirports(searchData.destination).map(airport => (
                  <option key={`dest-airport-${airport.code}`} value={airport.code}>
                    {airport.code} - {airport.city}
                  </option>
                ))}
              </select>
            </div>

            {/* Year Selection */}
            <div className="form-group" style={{ width: '100%' }}>
              <label htmlFor="departureYear">Travel Year</label>
              <select
                id="departureYear"
                name="departureYear"
                value={searchData.departureYear}
                onChange={handleInputChange}
                required
                className="w-full max-w-full box-border"
                style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
              >
                {years.map(year => (
                  <option key={year} value={year}>{year}</option>
                ))}
              </select>
            </div>

            {/* Start Month Selection */}
            <div className="form-group" style={{ width: '100%' }}>
              <label htmlFor="departureStartMonth">Travel Window Start</label>
              <select
                id="departureStartMonth"
                name="departureStartMonth"
                value={searchData.departureStartMonth}
                onChange={handleInputChange}
                required
                className="w-full max-w-full box-border"
                style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
              >
                <option value="">Select Month</option>
                {months.map(month => (
                  <option key={`start-${month}`} value={month}>{month}</option>
                ))}
              </select>
            </div>

            {/* End Month Selection */}
            <div className="form-group" style={{ width: '100%' }}>
              <label htmlFor="departureEndMonth">Travel Window End</label>
              <select
                id="departureEndMonth"
                name="departureEndMonth"
                value={searchData.departureEndMonth}
                onChange={handleInputChange}
                required
                className="w-full max-w-full box-border"
                style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
              >
                <option value="">Select Month</option>
                {months.map(month => (
                  <option key={`end-${month}`} value={month}>{month}</option>
                ))}
              </select>
            </div>

            {/* Airline/Carrier Selection */}
            <div className="form-group" style={{ width: '100%' }}>
              <label htmlFor="carrier">Preferred Airline</label>
              <select
                id="carrier"
                name="carrier"
                value={searchData.carrier}
                onChange={handleInputChange}
                className="w-full max-w-full box-border"
                style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
              >
                <option value="">All Airlines</option>
                {AIRLINES.map(airline => (
                  <option key={airline.code} value={airline.code}>
                    {airline.name} ({airline.code})
                  </option>
                ))}
              </select>
            </div>

            {/* Passenger Type Selection */}
            <div className="form-group" style={{ width: '100%' }}>
              <label htmlFor="passengerType">Passenger Type</label>
              <select
                id="passengerType"
                name="passengerType"
                value={searchData.passengerType}
                onChange={handleInputChange}
                className="w-full max-w-full box-border"
                style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
              >
                {PASSENGER_TYPES.map(type => (
                  <option key={type.value} value={type.value}>{type.label}</option>
                ))}
              </select>
              {studentDiscountNote && (
                <div className="select-help">{studentDiscountNote}</div>
              )}
            </div>

            {/* Analysis Type Selection */}
            <div className="form-group" style={{ width: '100%' }}>
              <label htmlFor="analysisType">Analysis Mode</label>
              <select
                id="analysisType"
                name="analysisType"
                value={searchData.analysisType}
                onChange={handleInputChange}
                className="w-full max-w-full box-border"
                style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
              >
                <option value="quarter">Quarterly</option>
                <option value="month">Monthly</option>
                <option value="week">Weekly</option>
                <option value="date">Daily</option>
              </select>
            </div>

            {/* Weeks Ahead Selection (only for Daily analysis) */}
            {searchData.analysisType === 'date' && (
              <div className="form-group" style={{ width: '100%' }}>
                <label htmlFor="weeksAhead">Analysis Timeframe</label>
                <input
                  type="number"
                  id="weeksAhead"
                  name="weeksAhead"
                  min="1"
                  max="52"
                  value={searchData.weeksAhead}
                  onChange={handleInputChange}
                  className="w-full max-w-full box-border"
                  style={{ width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}
                />
                <div className="select-help">Number of weeks to analyze</div>
              </div>
            )}
          </div>

          <button 
            type="submit" 
            className="search-button" 
            disabled={isLoading}
          >
            {isLoading ? 'Analyzing Flight Data...' : 'Run Price Analysis'}
          </button>
        </form>
      </div>

      {/* Loading indicator */}
      {isLoading && (
        <div className="loading-indicator">
          <div className="spinner"></div>
          <p>Processing flight data and generating predictions...</p>
        </div>
      )}
      
      {/* Error message display */}
      {error && (
        <div className="error-message">
          <p>Flight Analysis Error: {error}</p>
        </div>
      )}

      {/* Results section */}
      {searchResults && (
        <div className="search-results">
          <h3>{getAnalysisTypeLabel(searchResults[0].analysisType)}</h3>
          
          {searchResults.map(result => (
            <div key={result.id} className="result-card">
              <div className="result-header">
                <div className="route-info">
                  <h4>{result.origin} to {result.destination} ({result.departureYear})</h4>
                  <div className="travel-period">Travel window: {result.travelPeriod}</div>
                  <div className="current-price">
                    ${result.price.toFixed(2)}
                    {searchData.passengerType === 'student' && (
                      <span className="discount-note"> (Student discount may apply)</span>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="prediction-details">
                <div className="prediction-item">
                  <span className="prediction-label">Optimal Booking Time</span>
                  <strong className="prediction-value">{result.bestTime}</strong>
                </div>
                
                <div className="prediction-item">
                  <span className="prediction-label">Price Advantage</span>
                  <strong className="prediction-value good">{result.predictedSavings}</strong>
                </div>

                {result.carrier && (
                  <div className="prediction-item">
                    <span className="prediction-label">Selected Airline</span>
                    <strong className="prediction-value">
                      {result.carrier_name || result.carrier}
                    </strong>
                  </div>
                )}
                
                <div className="prediction-item">
                  <span className="prediction-label">Passenger Type</span>
                  <strong className="prediction-value">
                    {PASSENGER_TYPES.find(t => t.value === searchData.passengerType)?.label || 'Adult'}
                  </strong>
                </div>
              </div>
              
              <div className="price-history">
                <h5>Price Trend Analysis</h5>
                <div className="chart-container">
                  {result.chart_data ? (
                    <FlightPriceChart chartData={result.chart_data} />
                  ) : (
                    <div className="chart-placeholder">
                      <div className="bar-chart">
                        {result.priceHistory.map((price: number, index: number) => {
                          const maxPrice: number = Math.max(...result.priceHistory);
                          const minPrice: number = Math.min(...result.priceHistory);
                          
                          return (
                            <div 
                              key={index} 
                              className="bar" 
                              style={{ 
                                height: `${(price / maxPrice) * 100}%`,
                                backgroundColor: price === minPrice ? 'var(--accent-teal)' : 'var(--accent-blue)'
                              }}
                              title={`$${price}`}
                            ></div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div className="action-buttons">
                <button className="set-alert">Set Price Alert</button>
                <button 
                  className="view-details" 
                  onClick={() => openFullAnalysis(result)}
                >
                  Open Flight Deck
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
      
      {/* Full Analysis Modal */}
      <FullAnalysisModal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)} 
        result={selectedResult}
      />
    </div>
  );
};

export default FlightSearch;