import React from 'react';

export default function Home() {
  return (
    <div className="max-w-4xl mx-auto px-6 py-12 space-y-12">
      <header className="space-y-4">
        <h1 className="text-4xl font-bold tracking-tight text-white">
          Ego-centric Co-Speech Intent Grounding
        </h1>
        <p className="text-xl text-zinc-400">
          A benchmark for evaluating Embodied AI agents on their ability to understand 
          multimodal instructions in ego-centric environments.
        </p>
      </header>

      <section className="glass-panel p-8 rounded-lg space-y-6">
        <h2 className="text-2xl font-semibold text-white">Task Definition</h2>
        <div className="space-y-4 text-zinc-300 leading-relaxed">
          <p>
            The EcoG (Ego-centric Co-Speech Intent Grounding) task requires an agent to perceive an environment 
            through an ego-centric camera and interpret natural language instructions from a user.
          </p>
          <p>
            Unlike traditional grounding tasks, EcoG involves:
          </p>
          <ul className="list-disc pl-6 space-y-2 text-zinc-400">
            <li><strong className="text-white">Multimodal Input:</strong> Video frames + Audio/Transcript.</li>
            <li><strong className="text-white">Spatial & Temporal Grounding:</strong> Identifying objects (what), locations (where), and timing (when).</li>
            <li><strong className="text-white">Intent Understanding:</strong> Translating ambiguous commands ("put this there") into explicit actionable instructions.</li>
          </ul>
        </div>
      </section>

      <section className="grid md:grid-cols-2 gap-6">
        <div className="glass-panel p-6 rounded-lg">
            <h3 className="text-xl font-medium text-white mb-4">Input</h3>
            <pre className="bg-black p-4 rounded text-sm text-green-400 overflow-x-auto">
{`{
  "video": "frames or video_file",
  "audio": "transcription",
  "user_prompt": "put this ... there"
}`}
            </pre>
        </div>
        <div className="glass-panel p-6 rounded-lg">
            <h3 className="text-xl font-medium text-white mb-4">Output</h3>
            <pre className="bg-black p-4 rounded text-sm text-blue-400 overflow-x-auto">
{`{
  "explicit_command": "...",
  "point_list": [
    { "type": "target", "point": [x, y] },
    { "type": "spatial", "point": [x, y] }
  ]
}`}
            </pre>
        </div>
      </section>

      <footer className="pt-8 border-t border-zinc-800">
        <p className="text-xs text-zinc-500 text-center">
          Note: Normalized coordinates <code className="text-zinc-400">[x, y]</code> in range <code className="text-zinc-400">0-1000</code>. Origin <code className="text-zinc-400">(0,0)</code> at top-left.
        </p>
      </footer>
    </div>
  );
}

