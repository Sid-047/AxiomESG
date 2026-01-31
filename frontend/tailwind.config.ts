import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        heading: ["\"Bodoni 72 Smallcaps\"", "\"Bodoni MT\"", "\"Didot\"", "serif"],
        crest: ["\"Baskerville Old Face\"", "\"Baskerville\"", "serif"],
        hero: ["\"Anton\"", "\"Bebas Neue\"", "\"Impact\"", "sans-serif"],
        body: ["\"Inter\"", "\"Montserrat\"", "system-ui", "sans-serif"],
      },
      colors: {
        bg: "var(--bg)",
        fg: "var(--fg)",
        muted: "var(--muted)",
        hairline: "var(--hairline)",
      },
    },
  },
  plugins: [],
};

export default config;
