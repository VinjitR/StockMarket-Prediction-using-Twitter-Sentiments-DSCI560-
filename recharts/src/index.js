import { StrictMode } from "react";
import ReactDOM from "react-dom";
import * as V from "victory";

import App from "./App";

const rootElement = document.getElementById("root");
ReactDOM.render(
  <StrictMode>
    <App />
  </StrictMode>,
  rootElement
);
