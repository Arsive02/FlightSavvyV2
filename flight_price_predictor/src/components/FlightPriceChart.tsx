import React from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, Cell
} from 'recharts';

interface ChartProps {
  chartData: any;
}

const FlightPriceChart: React.FC<ChartProps> = ({ chartData }) => {
  if (!chartData || !chartData.data || chartData.data.length === 0) {
    return <div>No chart data available</div>;
  }

  // Custom tooltip to display formatted values
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload;
      return (
        <div className="custom-tooltip">
          <p className="label">{`${label}`}</p>
          <p className="info">{`Fare: $${dataPoint.fare}`}</p>
          {dataPoint.isHoliday && <p className="holiday">Holiday Period</p>}
          {dataPoint.isBest && <p className="best">Best Price!</p>}
        </div>
      );
    }
    return null;
  };

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
            fillOpacity={1}
            stroke="#5090d3"
            strokeWidth={1}
            isAnimationActive={true}
          >
            {chartData.data.map((entry: any, index: number) => {
              // Determine color based on the entry
              let color = '#5090d3'; // Default blue
              if ('inTravelPeriod' in entry) {
                if (entry.isBest) color = '#4caf50'; // Best time (green)
                else color = entry.inTravelPeriod ? '#5090d3' : '#e0e0e0'; // In travel period (blue) or not (gray)
              } else if (entry.isBest) {
                color = '#4caf50'; // Best time (green)
              }
              return <Cell key={`cell-${index}`} fill={color} stroke={color} />;
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default FlightPriceChart;