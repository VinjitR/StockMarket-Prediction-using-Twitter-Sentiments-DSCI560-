import React from 'react';
import { PieChart, Pie, Sector, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';


const COLORS = ['green','red','blue'];
export default function RenderPiechart({data}) {
    console.log(data);
  return (
    <div className='piechart'>
    <PieChart width={800} height={400}>    
        <Pie
    data={data}
    cx={600}
    cy={300}
    startAngle={180}
    endAngle={0}
    innerRadius={150}
    outerRadius={200}
    fill="#8884d8"
    paddingAngle={5}
    dataKey="value"
  >
    {data.map((entry, index) => (
      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
    ))}
    
  </Pie>
  <Tooltip/>
  <Legend/>
</PieChart>
</div>
  )
}
