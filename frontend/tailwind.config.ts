import type { Config } from 'tailwindcss'

/**
 * Tailwind CSS v4 configuration.
 *
 * This project is **CSS-first**: the design tokens (colours, fonts, radii,
 * shadows, the `up`/`down` trading palette and the `pnl-positive`/`pnl-negative`
 * utilities) are declared with `@theme` / `@utility` in `src/index.css`, and the
 * `@tailwindcss/vite` plugin in `vite.config.ts` drives the build. No
 * `postcss.config.js` is required (v4 replaces the PostCSS pipeline).
 *
 * This file is loaded via `@config "../tailwind.config.ts"` in `src/index.css`.
 * It supplies the content globs and a couple of additive extensions (the
 * `terminal` surface palette for dense data views and explicit font stacks)
 * that complement — rather than duplicate — the CSS tokens.
 */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: ['class'],
  theme: {
    extend: {
      colors: {
        // Warm graphite "terminal" surfaces for dense market-data panels,
        // tuned to match the amber/graphite token system in index.css.
        terminal: {
          bg: '#15130e',
          surface: '#1c1914',
          card: '#221e18',
          border: '#322c23',
          hover: '#2b261e',
          text: '#ece6da',
          muted: '#8a8275',
          accent: '#e8ae49',
          'accent-light': '#f3c873',
        },
      },
      fontFamily: {
        display: ['Bricolage Grotesque', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        sans: ['Hanken Grotesk', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['IBM Plex Mono', 'ui-monospace', 'SF Mono', 'monospace'],
      },
    },
  },
  plugins: [],
} satisfies Config
