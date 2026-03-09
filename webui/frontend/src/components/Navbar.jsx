import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Terminal, Activity, Database, Github, Trophy } from 'lucide-react';

export default function Navbar() {
  const location = useLocation();

  const NavItem = ({ to, icon: Icon, label }) => {
    const isActive = location.pathname === to;
    return (
      <Link
        to={to}
        className={`flex items-center gap-2 px-4 py-2 text-sm transition-colors border-b-2 ${
          isActive 
            ? 'text-white border-white bg-zinc-900' 
            : 'text-zinc-500 border-transparent hover:text-zinc-300 hover:bg-zinc-900/50'
        }`}
      >
        <Icon size={16} />
        <span>{label}</span>
      </Link>
    );
  };

  return (
    <nav className="sticky top-0 z-50 w-full border-b border-zinc-800 bg-black/80 backdrop-blur">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Terminal className="text-white" />
          <span className="font-bold text-lg tracking-wider">EcoG<span className="text-zinc-500">BENCH</span></span>
        </div>
        
        <div className="flex items-center gap-1">
          <NavItem to="/" icon={Activity} label="Task Intro" />
          <NavItem to="/benchmark" icon={Terminal} label="Benchmark" />
          <NavItem to="/leaderboard" icon={Trophy} label="Leaderboard" />
          <NavItem to="/dataset" icon={Database} label="Dataset" />
        </div>

        <div className="flex items-center gap-4">
           <a href="#" className="text-zinc-500 hover:text-white transition-colors">
             <Github size={20} />
           </a>
        </div>
      </div>
    </nav>
  );
}

