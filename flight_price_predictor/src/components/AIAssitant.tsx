import React, { useState, useRef, useEffect } from 'react';
import { flightApi } from '../services/api';
import OpenAI from 'openai';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';  // GitHub Flavored Markdown
import rehypeHighlight from 'rehype-highlight';


interface Message {
  id: string;
  type: 'user' | 'bot';
  text: string;
  timestamp: Date;
}

// Month names array
const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
];

// Interface for flight prediction parameters
interface FlightPredictionParams {
  origin: string;
  destination: string;
  granularity?: 'date' | 'week' | 'month' | 'quarter';
  futureYear?: number;
  weeksAhead?: number;
  start_month?: number;
  end_month?: number;
  carrier?: string;
  passengerType?: 'adult' | 'child' | 'student';
}

// Flight intent interface
interface FlightIntent {
  origin?: string;
  destination?: string;
  startMonth?: string;
  endMonth?: string;
  carrier?: string;
  year?: number;
  granularity?: 'date' | 'week' | 'month' | 'quarter';
  weeksAhead?: number;
  passengerType?: 'adult' | 'child' | 'student';
  isComplete: boolean;
  missingFields: string[];
}

const AIAssistant: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'bot',
      text: "Hello! I'm your FlightSavvy AI assistant. I can help you find the best time to book your flights for the lowest fares. Simply tell me your travel plans, including where you're flying from and to, and when you're planning to travel.",
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Sample suggestions for the user
  const suggestions = [
    "When should I book a flight from New York to Los Angeles in summer?",
    "What's the best time to fly from Chicago to Miami?",
    "When is the cheapest time to fly to Europe?",
    "How far in advance should I book domestic flights?",
    "Find me the best time to book a flight from Denver to Seattle in December"
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value);
  };

  // Initialize OpenAI client
  const createOpenAIClient = () => {
    return new OpenAI({
      apiKey: import.meta.env.VITE_OPENAI_API_KEY,
      dangerouslyAllowBrowser: true
    });
  };

  // Extract flight information from user input using OpenAI
  const extractFlightInformation = async (userInput: string): Promise<FlightIntent> => {
    try {
      const openai = createOpenAIClient();
      
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `You are a flight information extraction assistant. Extract flight information from the user's request. Ask them for necessary details if any information is missing.
            Return a JSON object with the following keys: 
            - origin (city name or airport code)
            - destination (city name or airport code)
            - startMonth (if provided, full month name like "January")
            - endMonth (if provided, full month name like "January")
            - carrier (airline code if provided, e.g. "AA" for American Airlines)
            - year (if provided, full 4-digit year)
            - granularity (one of: "date", "week", "month", "quarter", depending on how specific the user wants)
            - passengerType (one of: "adult", "child", "student", if mentioned)
            
            For missing information, ask the user again for those particular missing informations. Do not include explanations, only output valid JSON.
            
            Example output:
            {"origin":"New York","destination":"Los Angeles","startMonth":"June","endMonth":"August","carrier":null,"year":2025,"granularity":"month","passengerType":null}`
          },
          { role: "user", content: userInput }
        ],
        temperature: 0.0,
        response_format: { type: "json_object" }
      });
      
      // Parse the response
      const content = response.choices[0].message.content;
      if (!content) {
        throw new Error("No content in response");
      }
      
      const flightInfo = JSON.parse(content);
      
      // Check which fields are missing
      const missingFields = [];
      if (!flightInfo.origin) missingFields.push('origin');
      if (!flightInfo.destination) missingFields.push('destination');
      if (!flightInfo.startMonth && !flightInfo.endMonth) missingFields.push('travel period');
      if (!flightInfo.passengerType) missingFields.push('passenger type');
      if (!flightInfo.carrier) missingFields.push('preferred airline');
      
      // Set isComplete flag - only require essential fields for completion
      const isComplete = !missingFields.includes('origin') && 
                         !missingFields.includes('destination') && 
                         !missingFields.includes('travel period');
      
      return {
        ...flightInfo,
        isComplete,
        missingFields
      };
    } catch (error) {
      console.error('Error extracting flight information:', error);
      return { 
        isComplete: false,
        missingFields: ['all information']
      };
    }
  };

  // Convert month name to number (1-12)
  const getMonthNumber = (monthName: string | undefined | null): number | undefined => {
    if (!monthName) return undefined;
    
    const index = MONTHS.findIndex(month => 
      month.toLowerCase() === monthName.toLowerCase()
    );
    return index !== -1 ? index + 1 : undefined;
  };

  // Format flight parameters for backend API
  const formatFlightParams = (intent: FlightIntent): FlightPredictionParams => {
    const params: FlightPredictionParams = {
      origin: intent.origin || '',
      destination: intent.destination || '',
      granularity: intent.granularity || 'month',
    };
    
    // Add year if provided
    if (intent.year) {
      params.futureYear = intent.year;
    } else {
      // Default to current year
      params.futureYear = new Date().getFullYear();
    }
    
    // Add start and end months if provided
    const startMonthNum = getMonthNumber(intent.startMonth);
    if (startMonthNum) {
      params.start_month = startMonthNum;
    }
    
    const endMonthNum = getMonthNumber(intent.endMonth);
    if (endMonthNum) {
      params.end_month = endMonthNum;
    }
    
    // If only start month provided, assume same month for end
    if (params.start_month && !params.end_month) {
      params.end_month = params.start_month;
    }
    
    // If only end month provided, assume same month for start
    if (!params.start_month && params.end_month) {
      params.start_month = params.end_month;
    }
    
    // Add carrier if specified
    if (intent.carrier) {
      params.carrier = intent.carrier;
    }
    
    // Add passenger type if specified
    if (intent.passengerType) {
      params.passengerType = intent.passengerType;
    }
    
    // Add weeks ahead if using date granularity
    if (intent.granularity === 'date') {
      params.weeksAhead = intent.weeksAhead || 12; // Default to 12 weeks
    }
    
    return params;
  };

  // Generate response based on user input
  const generateFlightResponse = async (userMessage: string, providedIntent?: FlightIntent): Promise<string> => {
    try {
      // Use provided intent or extract flight information
      const flightIntent = providedIntent || await extractFlightInformation(userMessage);
      
      // If we don't have complete information, ask for the missing parts
      if (!flightIntent.isComplete) {
        return generateMissingInfoPrompt(flightIntent);
      }
      
      // Format parameters for API call
      const params = formatFlightParams(flightIntent);
      
      // Call the flight prediction API
      try {
        const results = await flightApi.predictBestTime(params);
        return formatPredictionResults(results, flightIntent);
      } catch (error) {
        console.error('Error calling flight prediction API:', error);
        return `I encountered an error while analyzing flight prices. This could be due to limited data for this specific route. Please try another route or time period.`;
      }
    } catch (error) {
      console.error('Error generating flight response:', error);
      return `I'm sorry, but I couldn't process your request. Please try again with more specific flight information like origin, destination, and travel dates.`;
    }
  };

  // Generate a prompt for missing information
  const generateMissingInfoPrompt = (intent: FlightIntent): string => {
    const { missingFields } = intent;
    
    if (missingFields.includes('origin') && missingFields.includes('destination')) {
      return "I'd be happy to help you find the best time to book your flight. To get started, I need to know both your departure and destination cities or airports. Where are you flying from and to?";
    }
    
    if (missingFields.includes('origin')) {
      return `To find the best flight prices, I need to know where you'll be departing from. Could you please tell me your departure city or airport?`;
    }
    
    if (missingFields.includes('destination')) {
      const origin = intent.origin ? `from ${intent.origin}` : '';
      return `Where are you planning to fly to ${origin}? Please specify your destination city or airport.`;
    }
    
    if (missingFields.includes('travel period')) {
      const route = intent.origin && intent.destination 
        ? `from ${intent.origin} to ${intent.destination}` 
        : 'for your trip';
      return `What time of year are you planning to travel ${route}? Please specify the month(s) you're considering (e.g., "June to August" or "December").`;
    }
    
    // Default message if something else is missing
    return "I need a bit more information to find the best flight prices for you. Please provide details about your origin, destination, and travel dates.";
  };

  // Format the prediction results into a user-friendly response
  const formatPredictionResults = (results: any, intent: FlightIntent): string => {
    if (!results.success) {
      return `I couldn't find prediction results for that route. ${results.error || 'Please try with different parameters.'}`;
    }
    
    const bestTime = results.best_time;
    const formattedBestTime = results.formatted_best_time;
    const predictedFare = bestTime.predicted_fare.toFixed(2);
    
    // Format the message with markdown for better presentation
    let message = `## Flight Price Analysis: ${intent.origin} to ${intent.destination}

### Optimal Booking Time
The best time to book your flight is: **${formattedBestTime}**

### Price Forecast
Expected fare: **$${predictedFare}**`;
    
    if (results.travel_period && results.travel_period.start_month_name && results.travel_period.end_month_name) {
      message += `\n\n### Travel Window
This analysis is for travel during **${results.travel_period.start_month_name}** to **${results.travel_period.end_month_name}**.`;
    }
    
    // Add carrier info if available
    if (results.carrier_name) {
      message += `\n\n### Airline
This analysis is specific to **${results.carrier_name}** (${results.carrier}).`;
    }
    
    // Add booking strategy table
    message += "\n\n### Booking Strategy\n";
    message += "| Strategy | Details |\n";
    message += "|----------|--------|\n";
    
    if (intent.granularity === 'date') {
      message += "| **Optimal days** | Book on Tuesday or Wednesday for best fares |\n";
      message += "| **Price tracking** | Set alerts for this specific date |\n";
    } else if (intent.granularity === 'week') {
      message += "| **Within week** | Aim for Tuesday-Thursday departures |\n";
      message += "| **Time of day** | Early morning flights often have better rates |\n";
    } else {
      message += "| **Booking window** | 6-8 weeks before domestic, 3-4 months before international |\n";
      message += "| **Flexibility** | Mid-week departures save 10-15% on average |\n";
    }
    
    message += "| **Price alerts** | Consider setting up fare tracking notifications |\n";
    message += "| **Flexibility bonus** | Flexible dates can save additional 10-15% |\n";
    
    // Add fare comparison if we have the data
    if (results.all_predictions && results.all_predictions.length > 0) {
      // Calculate average fare
      const avgFare = results.all_predictions.reduce((sum: number, pred: any) => sum + pred.predicted_fare, 0) / results.all_predictions.length;
      const savingsPercent = ((avgFare - bestTime.predicted_fare) / avgFare * 100).toFixed(0);
      
      message += `\n### Savings Opportunity
Booking at the optimal time saves approximately **${savingsPercent}%** compared to average fares for this route.`;
    }
    
    message += "\n\nIs there anything else you'd like to know about this route or another trip?";
    
    return message;
  };

  // Generate helpful information for general flight questions with conversation context
  const generateGeneralFlightInfo = async (userMessage: string, messageHistory: Message[]): Promise<string> => {
    try {
      const openai = createOpenAIClient();
      
      // Format the conversation history
      const conversationHistory = formatConversationHistory(messageHistory);
      
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `You are a flight planning assistant that specializes in helping users find the best time to book flights for the lowest fares.
            
            When answering questions about flight bookings and travel planning, use structured markdown formatting with headers, tables, and bullet points to make your responses easy to read.
            
            Focus on providing actionable advice about:
            - Best times to book flights
            - Seasonal pricing patterns
            - Day-of-week effects on prices
            - Carrier comparisons
            - Booking window recommendations
            - Discount strategies for different passenger types
            
            Always format numerical data in tables when possible. Use markdown formatting to organize information clearly.
            
            Consider the full conversation context when answering questions. If the user has mentioned specific routes, travel dates, or preferences in previous messages, incorporate that information in your response when relevant.`
          },
          ...conversationHistory.slice(1), // Skip the system message
          {
            role: "user", 
            content: userMessage
          }
        ],
        temperature: 0.5,
        max_tokens: 800
      });
      
      return response.choices[0].message.content || "I'm not sure about that. Could you provide more details about your travel plans?";
    } catch (error) {
      console.error('Error generating general flight info:', error);
      return "I'm having trouble processing your request. Could you try asking again about specific flight information or booking strategy?";
    }
  };

  // Format the conversation history for API calls
  const formatConversationHistory = (messages: Message[]): Array<{ role: 'user' | 'assistant' | 'system', content: string }> => {
    // Create the system message
    const formattedHistory: Array<{ role: 'user' | 'assistant' | 'system', content: string }> = [
      {
        role: 'system',
        content: `You are a flight planning assistant that specializes in helping users find the best time to book flights for the lowest fares.
        
        You help users plan their trips by extracting flight information and providing pricing advice. Consider the full conversation context when responding.
        
        When users mention places, dates, or airlines, remember this information for future responses. If information is missing, ask for it specifically.
        
        Always format responses using markdown with proper structure, tables, and emphasis where appropriate.`
      }
    ];
    
    // Add the conversation history (skip the first message which is just the intro)
    for (let i = 1; i < messages.length; i++) {
      const message = messages[i];
      formattedHistory.push({
        role: message.type === 'user' ? 'user' : 'assistant',
        content: message.text
      });
    }
    
    return formattedHistory;
  };

  // Track extracted flight information across conversations
  const [flightContext, setFlightContext] = useState<FlightIntent | null>(null);

  // Process the message and generate appropriate response
  const processMessage = async (userMessage: string): Promise<string> => {
    const openai = createOpenAIClient();
    
    try {
      // Get conversation history
      const conversationHistory = formatConversationHistory(messages);
      
      // Add the current message
      const fullConversation = [
        ...conversationHistory,
        { role: 'user' as const, content: userMessage }
      ];
      
      // First determine if this is a flight booking request or a general question
      const classificationResponse = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `Classify the user's query in the context of this conversation as either a "flight_booking_request" or a "general_flight_question".

A "flight_booking_request" is when the user is asking about a specific route or trip with intent to find booking information.
A "general_flight_question" is when the user is asking about general flight knowledge, pricing trends, or best practices.

Respond with just one word: either "flight_booking_request" or "general_flight_question".`
          },
          ...fullConversation.slice(1) // Skip the system message
        ],
        temperature: 0.0,
        max_tokens: 25
      });
      
      const classification = classificationResponse.choices[0].message.content?.trim();
      
      if (classification === "flight_booking_request") {
        // Extract flight information with conversation context
        const flightIntent = await extractFlightInformationWithContext(userMessage, messages);
        
        // Update the flight context
        setFlightContext(flightIntent);
        
        if (flightIntent.isComplete) {
          // We have all the necessary information, generate flight response
          return await generateFlightResponse(userMessage, flightIntent);
        } else {
          // We need more information
          return generateMissingInfoPrompt(flightIntent);
        }
      } else {
        // For general flight questions, include conversation context
        return await generateGeneralFlightInfo(userMessage, messages);
      }
    } catch (error) {
      console.error('Error processing message:', error);
      return "I'm having trouble understanding your request. Could you please rephrase it or provide more specific details about your travel plans?";
    }
  };
  
  // Extract flight information with conversation context
  const extractFlightInformationWithContext = async (userMessage: string, messageHistory: Message[]): Promise<FlightIntent> => {
    try {
      const openai = createOpenAIClient();
      
      // Format the conversation history
      const conversationHistory = formatConversationHistory(messageHistory);
      
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "system",
            content: `You are a flight information extraction assistant. Extract flight information from the entire conversation context.
            
            Consider all information shared in previous messages. If information is provided in the latest message, it should override older information.
            
            Return a JSON object with the following keys: 
            - origin (city name or airport code)
            - destination (city name or airport code)
            - startMonth (if provided, full month name like "January")
            - endMonth (if provided, full month name like "January")
            - carrier (airline code if provided, e.g. "AA" for American Airlines)
            - year (if provided, full 4-digit year)
            - granularity (one of: "date", "week", "month", "quarter", depending on how specific the user wants)
            - passengerType (one of: "adult", "child", "student", if mentioned)
            
            For missing information, leave the field null. The response must be valid JSON only, with no explanations.`
          },
          ...conversationHistory.slice(1), // Skip the system message
          { role: "user", content: userMessage }
        ],
        temperature: 0.0,
        response_format: { type: "json_object" }
      });
      
      // Parse the response
      const content = response.choices[0].message.content;
      if (!content) {
        throw new Error("No content in response");
      }
      
      const flightInfo = JSON.parse(content);
      
      // Check which fields are missing
      const missingFields = [];
      if (!flightInfo.origin) missingFields.push('origin');
      if (!flightInfo.destination) missingFields.push('destination');
      if (!flightInfo.startMonth && !flightInfo.endMonth) missingFields.push('travel period');
      if (!flightInfo.passengerType) missingFields.push('passenger type');
      if (!flightInfo.carrier) missingFields.push('preferred airline');
      
      // Set isComplete flag - only require essential fields for completion
      const isComplete = !missingFields.includes('origin') && 
                         !missingFields.includes('destination') && 
                         !missingFields.includes('travel period');
      
      return {
        ...flightInfo,
        isComplete,
        missingFields
      };
    } catch (error) {
      console.error('Error extracting flight information with context:', error);
      
      // If we have existing context, use it
      if (flightContext) {
        return flightContext;
      }
      
      return { 
        isComplete: false,
        missingFields: ['all information']
      };
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      text: input,
      timestamp: new Date()
    };

    // Update messages state first so the user sees their message immediately
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    // Create a local copy of messages that includes the new user message

    try {
      // Process the message with updated conversation history
      const responseText = await processMessage(input);
      
      // Add bot response
      const botMessage: Message = {
        id: (Date.now() + 100).toString(),
        type: 'bot',
        text: responseText,
        timestamp: new Date()
      };

      setMessages(updatedMessages => [...updatedMessages, botMessage]);
    } catch (error) {
      console.error('Error in handleSubmit:', error);
      
      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 100).toString(),
        type: 'bot',
        text: "I'm sorry, I encountered an error. Please try again with more specific details about your flight plans.",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }, 0);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion);
  };

  return (
    <div className="ai-assistant">
      <div className="ai-header">
        <h2>AI Travel Assistant</h2>
        <p>Ask me anything about flight prices, booking strategies, or travel trends</p>
      </div>
      
      <div className="chat-container">
        <div className="messages-container">
          {messages.map(message => (
            <div key={message.id} className={`message ${message.type}`}>
              <div className="message-content">
                <div className="message-text">
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                  components={{
                    div: ({node, ...props}) => <div className="markdown-content" {...props} />
                  }}
                >
                  {message.text}
                </ReactMarkdown>
                </div>
                <div className="message-time">
                  {message.timestamp.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                </div>
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="message bot typing">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="suggestions-container">
          <h4>Suggested Questions</h4>
          <div className="suggestions">
            {suggestions.map((suggestion, index) => (
              <button 
                key={index} 
                className="suggestion" 
                onClick={() => handleSuggestionClick(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
        
        <form className="input-container" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          type="text"
          placeholder="Enter your flight details or ask a question..."
          value={input}
          onChange={handleInputChange}
          disabled={isTyping}
        />
          <button type="submit" disabled={isTyping || !input.trim()}>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="send-icon">
              <path d="M22 2L11 13"></path>
              <path d="M22 2L15 22L11 13L2 9L22 2Z"></path>
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
};

export default AIAssistant;