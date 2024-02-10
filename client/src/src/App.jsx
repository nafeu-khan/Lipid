import React from "react";
import { Route, Routes } from "react-router-dom";
import AboutUs from "./Page/AboutUs";
import Lipid from "./Page/Lipid";
import Upload from "./Page/Upload";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Lipid />} />
      <Route path="/upload" element={<Upload />} />
      <Route path="/about-us" element={<AboutUs />} />
    </Routes>
  );
}

export default App;
