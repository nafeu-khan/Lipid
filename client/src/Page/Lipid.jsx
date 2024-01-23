import { ArrowRight } from "@mui/icons-material";
import { CircularProgress } from "@mui/material";
import React, { useEffect, useState } from "react";
import { Toaster } from "react-hot-toast";
import { useDispatch, useSelector } from "react-redux";
import { Link } from "react-router-dom";
import ActualPredicted from "../Components/ActualPredicted";
import ChartComponent from "../Components/ChartComponent";
import Header from "../Components/Header";
import LipidInputForm from "../Components/LipidInputForm";
import OperationsPanel from "../Components/OperationsPanel";
import SelectComponentType from "../Components/SelectComponentType";
import UploadFile from "../Components/UploadFile";
import { changeActiveLipid, changeOperationID } from "../Slices/LipidSlice";

// const data = JSON.parse(graph_data);
// console.log(JSON.stringify(data["APC"]))

function Lipid() {
  const [collapse, setCollapse] = useState(false);
  const [type, setType] = useState("single");
  const [lipidInput, setLipidInput] = useState([{ name: "", percentage: 100 }]);
  const dispatch = useDispatch();
  const operationID = useSelector((state) => state.lipid.operationID);
  const isLoading = useSelector((state) => state.lipid.loading);
  const data = useSelector((state) => state.lipid.data);
  const [graph_data, setGraphData] = useState([]);

  useEffect(() => {
    if (data && data.predicted) {
      let temp = [];
      for (const val of lipidInput) {
        if (data.predicted[val.name]) temp.push(data.predicted[val.name]);
      }
      setGraphData(temp);
    } else setGraphData(undefined);
  }, [data, lipidInput]);

  useEffect(() => {
    dispatch(changeActiveLipid(lipidInput));
    dispatch(changeOperationID("0"));
  }, [lipidInput]);

  const handleInputChange = (index, field, value) => {
    // Deep copy the object to ensure we're not modifying a read-only reference
    const updatedInputs = lipidInput.map((input, idx) =>
      idx === index ? { ...input, [field]: value } : { ...input }
    );

    // Update the state with the new array
    setLipidInput(updatedInputs);
  };

  return (
    <div className="h-screen  relative overflow-hidden">
      <Header />
      <div className="flex h-full w-full">
        <div
          className={`relative border-r-2 shadow-xl p-4 bg-[whitesmoke] flex flex-col ${
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
        <div
          className="w-full flex-grow relative"
          style={{ height: "calc(100vh - 68px)" }}
        >
          <Toaster />
          {isLoading ? (
            <div className="w-full h-full flex flex-col items-center justify-center relative -top-12">
              <h3 className="font-medium mb-2">Fetching Data...</h3>
              <CircularProgress />
            </div>
          ) : (
            <div className="w-full h-full grid place-items-center">
              {operationID === "0" && (
                <h1 className="font-medium text-3xl mt-10">
                  Select an operation
                </h1>
              )}
              {operationID === "1" && (
                <>
                  <ActualPredicted />
                  <ChartComponent graph_data={graph_data} />
                </>
              )}
              {operationID === "2" && (
                <div className="w-full h-full flex flex-col items-center p-2">
                  <h1 className="bg-violet-500 text-gray-100 mt-2 p-2 px-4 text-lg rounded shadow font-mono">
                    Prediction Value:{" "}
                    <span className="text-white font-semibold">20.34</span>
                  </h1>
                  {data.graph ? (
                    <img
                      className="mt-8"
                      src={`data:image/png;base64,${data.graph}`}
                      alt=""
                    />
                  ) : (
                    <h1 className="mt-4">No predicted data found.</h1>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Lipid;
