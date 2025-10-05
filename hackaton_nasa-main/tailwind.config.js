/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'space-background': '#0C1023',
        'ocean-blue': '#1A3351',
        'ocean-light': '#698BB3',
        'purple-labubu': '#8698FF',
        'gray-bar': '#A0A9BB',
        'our-purple': '#7083FF',
        'our-orange': '#EB7018',
        'our-roxo': '#402EE6',
        'our-green': '#0CB01F',
        'our-white': '#C0C6C1',
      }
    },
  },
  plugins: [],
}

