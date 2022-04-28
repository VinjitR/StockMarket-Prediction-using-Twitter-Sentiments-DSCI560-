import React from 'react'

export default function CompanyInfo({data}) {
  return (
    <div><h2>{data["Name"]}</h2>
      <h4>Description: </h4><p>{data["Description"]}</p>
    <h4>Sector: {data["Sector"]}</h4>
    <h5>Best High: {data["52WeekHigh"]}</h5>
    <h5>Worst Low: {data["52WeekLow"]}</h5>
    <h3></h3></div>
  )
}
