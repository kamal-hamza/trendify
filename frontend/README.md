# Trendify Frontend

The frontend for Trendify is a modern React application built with TypeScript, Vite, and Material UI. It provides an interactive dashboard for visualizing tech trends aggregated from various sources.

## Setup Instructions

### Prerequisites
- [Node.js](https://nodejs.org/) (v18.0.0 or higher)
- [npm](https://www.npmjs.com/) (usually comes with Node.js)

### Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd trendify/frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Environment Variables:**
   Create a `.env` file in the `trendify/frontend/` directory if you need to point to a specific API backend:
   ```env
   VITE_API_URL=http://localhost:8000/api
   ```
   *Note: If not set, it defaults to `/api` and relies on the Vite proxy configuration in `vite.config.ts`.*

## Available Scripts

### Development
Starts the development server with hot-module replacement:
```bash
npm run dev
```

### Build
Builds the application for production to the `dist` folder:
```bash
npm run build
```

### Lint
Runs ESLint to check for code quality issues:
```bash
npm run lint
```

### Preview
Locally preview the production build:
```bash
npm run preview
```

## Tech Stack
- **Framework:** React 19
- **Build Tool:** Vite
- **Language:** TypeScript
- **Styling/UI:** Material UI (MUI), Emotion, Lucide Icons
- **State Management:** Zustand
- **Data Fetching:** TanStack Query (React Query) & Axios
- **Charts:** Recharts