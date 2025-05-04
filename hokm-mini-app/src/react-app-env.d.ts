/// <reference types="react-scripts" />
interface TelegramWebApp {
    initDataUnsafe: {
      user?: {
        id: number;
        first_name?: string;
        last_name?: string;
        username?: string;
        language_code?: string;
      };
    };
    expand: () => void;
    ready: () => void;
    requestFullscreen?: () => void;
    // Add other WebApp methods/properties as needed
  }
  
  interface Window {
    Telegram: {
      WebApp: TelegramWebApp;
    };
  }
