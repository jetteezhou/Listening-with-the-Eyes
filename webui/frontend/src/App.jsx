import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Benchmark from './pages/Benchmark';
import Leaderboard from './pages/Leaderboard';
import Dataset from './pages/Dataset';

function App() {
  return (
    <Router basename="/embodied_benchmark">
      <div className="min-h-screen bg-black">
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/benchmark" element={<Benchmark />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
          <Route path="/dataset" element={<Dataset />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

