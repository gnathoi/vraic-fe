import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
// CSS order: PatternFly base, then (via App) PatternFly component styles, then ChatBot, then app overrides.
import '@patternfly/react-core/dist/styles/base.css';
import App from './App';
import '@patternfly/chatbot/dist/css/main.css';
import './app.css';
import { initAppearance } from './appearance';

initAppearance(); // theme classes on <html> before the first paint (no inline script allowed by the CSP)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
