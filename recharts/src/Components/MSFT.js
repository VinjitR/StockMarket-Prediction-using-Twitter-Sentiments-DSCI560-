import React, { useState, useEffect } from 'react'
import CompanyInfo from './CompanyInfo';
import RenderLineChart from './Linechart'
import RenderPiechart from './Piechart';

export default function MSFT() {
    let [stockdata, setStockData] = useState([]);
    let [loading,setLoading]=useState(true);
    let [arimadata,setArimadata]=useState([]);
    let [lstmdata,setLstmdata]=useState([]);
    let [comdata,setComData]=useState([]);
    let [piedata,setPiedata]=useState([]);


    let [loada,setLoadA]=useState(true);
    let [loadl,setLoadL]=useState(true);
    let [comloading,setComloading]=useState(true);
    let [pieloading,setpieloading]=useState(true);
    const get_date=(data)=>{
        const newData = data.map((d) => ({
            ...d,
            "Date": new Intl.DateTimeFormat('en-US', { year: 'numeric', month: '2-digit', day: '2-digit' }).format(d["Date"])  // just for example
          }));
          return newData
    }

    // 3. Create out useEffect function
  useEffect(() => {
    setLoadA(true);
    setLoading(true);



    fetch('https://www.alphavantage.co/query?function=OVERVIEW&symbol=MSFT&apikey=7NT5QY9WX6BZKLQ6')
    .then(response => response.json())
    .then(data => setComData(data))
    .then(setComloading(false));



    fetch("http://localhost:5000/stockdata/MSFT")
    .then(response => response.json())
    .then(data=>get_date(data))
    .then(data => setStockData(data))
    .then(setLoading(false));



  },[])

  const fetchARIMA = async(targetsize) => {
    setLoadA(true);
    alert("Switching to ARIMA MSFT "+targetsize.toString()+"day predictions")
    fetch("http://localhost:5000/ARIMA/"+targetsize.toString()+"/MSFT")
    .then(response => response.json())
    .then(data=>get_date(data))  
    .then(data => setArimadata(data))
    .then(setLoadA(false));
  }
  const fetchLSTM = async(targetsize) => {
    alert("Switching to LSTM MSFT "+targetsize.toString()+"day predictions")
    fetch("http://localhost:5000/LSTM/"+targetsize.toString()+"/MSFT")
    .then(response => response.json())
    .then(data=>get_date(data))
    .then(data => setLstmdata(data))
    .then(setLoadL(false));
}

const fetchTwitterPie= async()=>{
  fetch('http://localhost:5000/tweetper/MSFT')
  .then(response => response.json())
  .then(data => setPiedata(data))
  .then(setpieloading(false));
}
  return (
    <>
      <h1>MSFT</h1>
      <div className="s-center">
      <CompanyInfo data={comdata}/>
      </div>
      {
            loading?<h3>Loading</h3>:
    <div className="s-center">
        

            <div className='scroll-container'>
            <h2 className='con-heading'>Stockdata</h2>
        
        
              <RenderLineChart data={stockdata} val="Close"></RenderLineChart>
        </div>
        </div>}
        <div className="s-center">
      <button className='c-button' onClick={()=>fetchTwitterPie()}>Twitter_Info</button></div>
        {pieloading&&piedata.length==0?<h2>Click to load Twitter Info</h2>:
      <div className='s-center'>
        <RenderPiechart data={piedata}></RenderPiechart>
      </div>}
        
        
        <div className="s-center">
      <button className='c-button' onClick={()=>fetchARIMA(1)}>ARIMA-1</button>
      <button className='c-button' onClick={()=>fetchARIMA(2)}>ARIMA-2</button>
      <button className='c-button' onClick={()=>fetchARIMA(3)}>ARIMA-3</button></div>
      {
            loada?<h3>Click one option from above</h3>:
            <div className="s-center">
        <div className='scroll-container'>
            <h2 className='con-heading'>ARIMA</h2>
        
        
        <RenderLineChart data={arimadata} val="Predict"></RenderLineChart>
        </div>
        </div>
        
      }
      <div className="s-center">
      <button className='c-button' onClick={()=>fetchLSTM(1)}>LSTM-1</button>
      <button className='c-button' onClick={()=>fetchLSTM(2)}>LSTM-2</button>
      <button className='c-button' onClick={()=>fetchLSTM(3)}>LSTM-3</button></div>
      {loadl? <h3>Click one option from above</h3>:
      <div className="s-center">
      <div className='scroll-container'>
      <h2 className='con-heading'>LSTM</h2>
        <RenderLineChart data={lstmdata} val="Predict"></RenderLineChart></div>
        </div>}

      </>
  )
}
