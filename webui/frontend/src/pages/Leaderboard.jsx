import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Trophy, Medal, Star, ArrowUpRight, Loader2 } from 'lucide-react';

const API_BASE = '/embodied_benchmark/api';

export default function Leaderboard() {
  const [leaderboardData, setLeaderboardData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchLeaderboard = async () => {
      try {
        const res = await axios.get(`${API_BASE}/leaderboard`);
        setLeaderboardData(res.data);
      } catch (err) {
        console.error("Failed to fetch leaderboard:", err);
        setError("Failed to load leaderboard data. Please ensure the backend is running.");
      } finally {
        setLoading(false);
      }
    };
    fetchLeaderboard();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
      <div className="flex flex-col items-center mb-12">
        <div className="flex items-center gap-3 mb-4">
          <Trophy className="w-10 h-10 text-yellow-500" />
          <h1 className="text-4xl font-bold text-white tracking-tight">Leaderboard</h1>
        </div>
        <p className="text-zinc-500 text-center max-w-2xl">
          Comparison of state-of-the-art vision-language models on the EcoG benchmark.
          Scores are dynamically loaded from full evaluation results in the repository.
        </p>
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center py-20 gap-4">
          <Loader2 className="w-8 h-8 text-white animate-spin" />
          <p className="text-zinc-500">Loading ranking data...</p>
        </div>
      ) : error ? (
        <div className="glass-panel p-8 rounded-xl border-red-900/30 text-center">
          <p className="text-red-400">{error}</p>
        </div>
      ) : leaderboardData.length === 0 ? (
        <div className="glass-panel p-8 rounded-xl text-center">
          <p className="text-zinc-500">No evaluation results found in the results directory.</p>
        </div>
      ) : (
        <div className="glass-panel overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/20 backdrop-blur-sm">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-zinc-800 bg-zinc-900/50">
                  <th className="px-6 py-4 text-xs font-bold text-zinc-500 uppercase tracking-wider">Rank</th>
                  <th className="px-6 py-4 text-xs font-bold text-zinc-500 uppercase tracking-wider">Model</th>
                  <th className="px-6 py-4 text-xs font-bold text-zinc-500 uppercase tracking-wider text-center">Overall</th>
                  <th className="px-6 py-4 text-xs font-bold text-zinc-500 uppercase tracking-wider text-center">Intent</th>
                  <th className="px-6 py-4 text-xs font-bold text-zinc-500 uppercase tracking-wider text-center">Spatial</th>
                  <th className="px-6 py-4 text-xs font-bold text-zinc-500 uppercase tracking-wider text-center">Speech Temporal</th>
                  <th className="px-6 py-4 text-xs font-bold text-zinc-500 uppercase tracking-wider text-right">Date</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/50">
                {leaderboardData.map((item) => (
                  <tr key={item.rank} className="group hover:bg-white/5 transition-colors">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        {item.rank === 1 && <Medal className="w-4 h-4 text-yellow-500" />}
                        {item.rank === 2 && <Medal className="w-4 h-4 text-zinc-400" />}
                        {item.rank === 3 && <Medal className="w-4 h-4 text-amber-600" />}
                        <span className={`font-mono ${item.rank <= 3 ? 'text-white font-bold' : 'text-zinc-500'}`}>
                          #{item.rank}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex flex-col">
                        <span className="text-white font-semibold group-hover:text-blue-400 transition-colors cursor-pointer flex items-center gap-1">
                          {item.model}
                          <ArrowUpRight className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                        </span>
                        <span className="text-xs text-zinc-500">{item.provider}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-sm font-bold bg-white/10 text-white border border-white/20">
                        {item.acc_eco.toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-zinc-300 font-mono">{item.acc_cls.toFixed(1)}%</span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-zinc-300 font-mono">{item.acc_s.toFixed(1)}%</span>
                    </td>
                    <td className="px-6 py-4 text-center">
                      <span className="text-zinc-300 font-mono">{item.acc_t.toFixed(1)}%</span>
                    </td>
                    <td className="px-6 py-4 text-right text-zinc-500 text-xs font-mono">
                      {item.date}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="mt-12 grid md:grid-cols-3 gap-6">
        <div className="p-6 rounded-xl border border-zinc-800 bg-zinc-900/20">
          <div className="flex items-center gap-2 mb-4 text-white">
            <Star className="w-5 h-5 text-yellow-500" />
            <h3 className="font-semibold">Automatic Updates</h3>
          </div>
          <p className="text-sm text-zinc-500 leading-relaxed">
            The leaderboard is automatically updated whenever new full evaluation results (metrics_summary.json) are added to the results directory.
          </p>
        </div>
      </div>
    </div>
  );
}
