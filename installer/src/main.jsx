import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";
import { mockSetup } from "./mock.js";

// Opening dist/index.html in a normal browser (no Electron) shows a working preview.
if (!window.setup) window.setup = mockSetup();

createRoot(document.getElementById("root")).render(<App />);
