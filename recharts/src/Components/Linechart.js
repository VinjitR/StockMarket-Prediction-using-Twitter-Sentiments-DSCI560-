import React, { useEffect, useState } from "react";

import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer,Brush ,Tooltip,Legend} from "recharts";

const RenderLineChart = ({data,val}) => {
// console.log(data)

  return (
    
    <div className="linechart">
      
      <LineChart width={1000} height={400} data={data}>
        <Line type="line" dataKey={val} dot={false} stroke="blue" />
        {
          val=="Predict"&&<Line type="line" dataKey="Close" dot={false} stroke="orange" />}{ 
          val=="Predict"&& <Line type="line" dataKey="Buy"  dot={{ stroke: 'green', strokeWidth: 4 }} stroke="green"/>
        }
        { 
          val=="Predict"&& <Line type="line" dataKey="Sell"  dot={{ stroke: 'red', strokeWidth: 4 }}stroke="red"/>
        }
        <Legend />
        <XAxis dataKey="Date" angle={20}/>
        <YAxis />
        <Tooltip />
 
        <CartesianGrid stroke="#ccc" strokeDasharray="1 1" />
        <Brush dataKey="Date" />
      </LineChart>

    </div>
  );
};

export default RenderLineChart;
