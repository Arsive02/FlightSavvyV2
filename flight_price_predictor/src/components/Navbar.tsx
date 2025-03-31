import React from 'react';

interface NavbarProps {
  activeTab: 'search' | 'ai' | 'about';
  setActiveTab: (tab: 'search' | 'ai' | 'about') => void;
}

const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab }) => {
  return (
    <nav className="navbar">
      <div className="navbar-logo">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="plane-icon">
          <path d="M21 16l-2 2-6.5-2.5L9 18l-3-1.5L6 15 3 13.5 8 8l2.5 2.5L17 7l2 2-3 4.5 5 2.5z"/>
          <path d="M16 16l-2.5-7"/>
        </svg>
        <h1>FlightSavvy</h1>
      </div>
      <div className="navbar-tabs">
        <button
          className={`tab ${activeTab === 'about' ? 'active' : ''}`}
          onClick={() => setActiveTab('about')}
        >
          Flight Info
        </button>
        <button 
          className={`tab ${activeTab === 'search' ? 'active' : ''}`}
          onClick={() => setActiveTab('search')}
        >
          Price Navigator
        </button>
        <button 
          className={`tab ${activeTab === 'ai' ? 'active' : ''}`}
          onClick={() => setActiveTab('ai')}
        >
          Flight Assistant
        </button>
      </div>
    </nav>
  );
};

export default Navbar;