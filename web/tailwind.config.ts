import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        cream: "var(--color-cream)",
        forest: "var(--color-forest)",
        sage: "var(--color-sage)",
        disclaimer: "var(--color-disclaimer-bg)",
        "disclaimer-text": "var(--color-disclaimer-text)",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        serif: ["var(--font-fraunces)", "Georgia", "serif"],
      },
      boxShadow: {
        card: "0 1px 3px rgba(27, 48, 34, 0.06), 0 8px 24px rgba(27, 48, 34, 0.04)",
      },
      borderRadius: {
        card: "14px",
      },
    },
  },
  plugins: [],
};

export default config;
