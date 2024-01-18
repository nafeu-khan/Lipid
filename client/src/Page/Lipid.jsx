import { ArrowRight } from "@mui/icons-material";
import React, { useState } from "react";
import { Link } from "react-router-dom";
import Header from "../Components/Header";
import LipidInputForm from "../Components/LipidInputForm";
import OperationsPanel from "../Components/OperationsPanel";
import SelectComponentType from "../Components/SelectComponentType";
import UploadFile from "../Components/UploadFile";

// const data = JSON.parse(graph_data);
const data = [];

function Lipid() {
  const [collapse, setCollapse] = useState(false);
  const [type, setType] = useState("single");
  const [lipidInput, setLipidInput] = useState([{ name: "", percentage: 100 }]);

  const handleInputChange = (index, field, value) => {
    const updatedInputs = [...lipidInput];
    updatedInputs[index][field] = value;
    setLipidInput(updatedInputs);
  };

  return (
    <div className="h-screen  relative overflow-hidden">
      <Header />
      <div
        className={`relative w-[320px] border-r-2 shadow-xl p-4 bg-[whitesmoke] ${
          collapse && "max-w-[0] !p-0"
        }`}
        style={{ height: "calc(100vh - 68px)" }}
      >
        <span
          className={`absolute right-0 top-1/2 translate-x-1/2 bg-white shadow-lg border-2 cursor-pointer ${
            collapse && "!translate-x-full z-10"
          }`}
          onClick={() => setCollapse(!collapse)}
        >
          <ArrowRight />
        </span>
        {!collapse && (
          <>
            <div>
              <SelectComponentType
                setLipidInput={setLipidInput}
                type={type}
                setType={setType}
              />
              <LipidInputForm
                handleInputChange={handleInputChange}
                lipidInput={lipidInput}
                type={type}
              />
            </div>
            {lipidInput.length > 0 && <OperationsPanel />}
            <UploadFile />
            <div className="absolute z-50 left-0 text-center bottom-0 border-t-2 w-full py-2">
              <Link
                to={"/about-us"}
                className="underline text-gray-700/90 text-sm font-medium hover:text-gray-900"
              >
                About Us
              </Link>
            </div>
          </>
        )}
      </div>
      <div className="w-full h-full relative"></div>
    </div>
  );
}

export default Lipid;
