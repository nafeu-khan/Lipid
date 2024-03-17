import React, { useEffect } from "react";
import { Route, Routes } from "react-router-dom";
import AboutUs from "./Page/AboutUs";
import Lipid from "./Page/Lipid";
import Upload from "./Page/Upload";
import { useSelector } from "react-redux";

function App() {
  const data = useSelector(state => state.evaluation.data)

  useEffect(() => {
    if(data) localStorage.setItem('data', JSON.stringify(data));
  }, [data])

  return (
    <Routes>
      <Route path="/" element={<Lipid />} />
      <Route path="/upload" element={<Upload />} />
      <Route path="/about-us" element={<AboutUs />} />
    </Routes>
  );
}

export default App;
