import React from 'react';

const About: React.FC = () => {
    return (
        <div className="about-container">
            <h1>About FlightSavvy</h1>
            
            <div className="about-section">
                <h2>What We Do</h2>
                <p>
                    FlightSavvy is an innovative application that uses advanced machine learning algorithms 
                    to predict flight prices based on various factors such as departure date, destination, 
                    airline, and more. Our goal is to help travelers find the best time to book their flights 
                    and save money.
                </p>
                <p>
                    By analyzing historical pricing data and patterns, our predictor provides reliable estimates 
                    for future flight costs, helping you make informed decisions about your travel plans.
                </p>
            </div>
            
            <div className="about-section">
                <h2>How It Works</h2>
                <ul className="feature-list">
                    <li>Enter your flight details including origin, destination, and travel dates</li>
                    <li>Our machine learning model processes your request</li>
                    <li>Receive accurate price predictions with confidence intervals</li>
                    <li>Use our price trend analysis to determine the best time to book</li>
                </ul>
            </div>
            
            <div className="about-section">
                <h2>Our Technology</h2>
                <p>
                    This application is built using a combination of modern technologies:
                </p>
                <div className="tech-grid">
                    <div className="tech-item">
                        <div className="tech-icon">⚛️</div>
                        <div className="tech-details">
                            <div className="tech-name">Frontend</div>
                            <div className="tech-description">React with TypeScript and CSS</div>
                        </div>
                    </div>
                    <div className="tech-item">
                        <div className="tech-icon">🐍</div>
                        <div className="tech-details">
                            <div className="tech-name">Backend</div>
                            <div className="tech-description">Python with Flask</div>
                        </div>
                    </div>
                    <div className="tech-item">
                        <div className="tech-icon">🧠</div>
                        <div className="tech-details">
                            <div className="tech-name">Machine Learning</div>
                            <div className="tech-description">Advanced regression models trained on extensive flight data</div>
                        </div>
                    </div>
                    <div className="tech-item">
                        <div className="tech-icon">📊</div>
                        <div className="tech-details">
                            <div className="tech-name">Data Processing</div>
                            <div className="tech-description">Real-time data processing and analysis pipelines</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default About;