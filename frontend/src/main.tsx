import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { installCardLight } from "./lib/motion";
import "./styles/tokens.css";
import "./styles/app.css";

installCardLight();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
