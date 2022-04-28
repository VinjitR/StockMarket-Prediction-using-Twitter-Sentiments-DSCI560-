import "./styles.css";
import RenderLineChart from "./Components/Linechart";
import AAPL from "./Components/AAPL";
import AMZN from "./Components/AMZN";
import FB from "./Components/FB";
import Sidebar from "./Components/Sidebar";
import{BrowserRouter as Router, Routes, Route} from 'react-router-dom';
import Home from "./Components/Home";
import MSFT from "./Components/MSFT";
import TSLA from "./Components/TSLA";
import GOOGL from "./Components/GOOGL";


export default function App() {
  return (
    <div className="App">
      <Router>
      <div className="a-left"><Sidebar/></div>
      
      <div className="a-right"><Routes>
        <Route path="/" element={<Home />}/>
        <Route path="/AAPL" element={<AAPL />}/>
        <Route path="/AMZN" element={<AMZN />}/>
        <Route path="/FB" element={<FB />}/>
        <Route path="/MSFT" element={<MSFT />}/>
        <Route path="/TSLA" element={<TSLA/>}/>
        <Route path="/GOOGL" element={<GOOGL />}/>

        </Routes></div>
      </Router>
    </div>
  );
}
