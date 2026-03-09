/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'tech-black': '#0a0a0a',
        'tech-gray': '#1a1a1a',
        'tech-border': '#333333',
        'tech-accent': '#ffffff',
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'monospace'],
      }
    },
  },
  plugins: [],
}

