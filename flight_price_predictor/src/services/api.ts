interface PredictionParams {
    origin: string;
    destination: string;
    granularity?: 'date' | 'week' | 'month' | 'quarter';
    futureYear?: number;
    weeksAhead?: number;
  }
  
export const API_URL = '/api';
  
  export const flightApi = {
    predictBestTime: async (params: PredictionParams) => {
      try {
        const response = await fetch(`${API_URL}`, {
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
    }
  };