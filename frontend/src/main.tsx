import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { IconContext } from "@phosphor-icons/react";

import { App } from "./App";
import { installCardLight } from "./lib/motion";
import "./styles/tokens.css";
import "./styles/app.css";

installCardLight();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <IconContext.Provider value={{ size: 18, weight: "regular" }}>
      <App />
    </IconContext.Provider>
  </StrictMode>,
);
