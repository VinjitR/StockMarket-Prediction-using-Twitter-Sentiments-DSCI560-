import React from 'react';
import { sidebardata } from './Sidebardata';

export default function Sidebar() {
  return (
    <div className='m-sidebar'>
        <ul className='sidebar-list'>{sidebardata.map((val,key)=>{
        return<li key={key} 
        className="sidebar-row" 
        id={window.location.pathname==val.link?"active":""}
        onClick={()=>{window.location.pathname=val.link;}}>
            <div className='title'>
                {val.title}
                </div>
            </li>
    })}</ul>
    </div>
  )
}
