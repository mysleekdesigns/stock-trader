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
        // Dark "terminal" surfaces for dense market-data panels.
        terminal: {
          bg: '#0a0e17',
          surface: '#111827',
          card: '#1a2332',
          border: '#1f2937',
          hover: '#253347',
          text: '#e5e7eb',
          muted: '#6b7280',
          accent: '#3b82f6',
          'accent-light': '#60a5fa',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
} satisfies Config
