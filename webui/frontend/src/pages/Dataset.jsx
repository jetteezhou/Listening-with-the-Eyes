import React from 'react';

export default function Dataset() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="flex justify-between items-center mb-8">
            <h1 className="text-3xl font-bold text-white">Dataset Explorer</h1>
            <span className="px-3 py-1 text-xs font-mono bg-zinc-800 text-zinc-400 rounded">v1.0.0</span>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
            {/* Sidebar / Tree View Placeholder */}
            <div className="glass-panel p-4 rounded-lg h-[600px] overflow-y-auto">
                <h3 className="text-zinc-500 text-sm uppercase tracking-wider mb-4 font-bold">Structure</h3>
                <div className="space-y-4 font-mono text-sm">
                    <div>
                        <div className="text-white mb-2">data/</div>
                        <div className="pl-4 border-l border-zinc-800 space-y-2 text-zinc-500">
                            <div>四楼实训区_场景1.../</div>
                            <div className="pl-4 border-l border-zinc-800 space-y-1">
                                <div className="text-zinc-400">指令1/ (Pointing)</div>
                                <div className="text-zinc-400">指令2/ (Pick up)</div>
                                <div className="text-zinc-400">指令3/ (Pick & Place)</div>
                                <div className="text-zinc-400">指令4/ (Spatial Rel)</div>
                                <div className="text-zinc-400">指令5/ (Sequential)</div>
                                <div className="text-zinc-400">指令6/ (Complex)</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Main Content Placeholder */}
            <div className="md:col-span-2 glass-panel p-8 rounded-lg flex flex-col items-center justify-center text-center space-y-6">
                <div className="w-16 h-16 bg-zinc-800 rounded-full flex items-center justify-center">
                    <span className="text-2xl">📦</span>
                </div>
                <div>
                    <h3 className="text-xl font-semibold text-white">Dataset Download</h3>
                    <p className="text-zinc-400 mt-2 max-w-md">
                        The full dataset contains 100+ hours of ego-centric manipulation videos with dense spatial-temporal annotations.
                    </p>
                </div>
                <button className="px-6 py-2 bg-white text-black font-semibold rounded hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed" disabled>
                    Download (Coming Soon)
                </button>
            </div>
        </div>
    </div>
  );
}

