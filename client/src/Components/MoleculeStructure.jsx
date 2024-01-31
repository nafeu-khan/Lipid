import { CircularProgress } from "@mui/material";
import React, { useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import {
  changeActiveLipid,
  changeShowTable,
  getMoleculeStructure,
} from "../Slices/StructureAnalysisSlice";
import ActualPredicted from "./ActualPredicted";
import ChartComponent from "./ChartComponent";
import GraphTable from "./GraphTable";
import LipidInputForm from "./LipidInputForm";
import SelectCompositionType from "./SelectCompositionType";

function MoleculeStructure() {
  const [type, setType] = useState("single");
  const [lipidInput, setLipidInput] = useState([{ name: "", percentage: 100 }]);
  const dispatch = useDispatch();
  const { loading, data, showTable, lipid } = useSelector((state) => state.structure);
  const [open, setOpen] = useState(false);

  const handleInputChange = (index, field, value) => {
    // Deep copy the object to ensure we're not modifying a read-only reference
    const updatedInputs = lipidInput.map((input, idx) =>
      idx === index ? { ...input, [field]: value } : { ...input }
    );

    // Update the state with the new array
    setLipidInput(updatedInputs);
    dispatch(changeActiveLipid(updatedInputs));
  };

  return (
    <div className="w-full h-full">
      <div
        className={`mt-3 relative max-w-[350px] mx-auto border p-4 shadow rounded `}
      >
        <SelectCompositionType
          setLipidInput={setLipidInput}
          setType={setType}
          type={type}
        />
        <LipidInputForm
          handleInputChange={handleInputChange}
          lipidInput={lipidInput}
          type={type}
        />

        <button
          className="mt-2 ml-auto  bg-blue-500 w-full py-2 text-white rounded"
          onClick={() => {
            if (type === "single") {
              dispatch(getMoleculeStructure(lipidInput[0].name));
            } else {
              // TODO: Handle por multiple composition
            }
          }}
        >
          Analyze
        </button>
        {/* <span
          className="border cursor-pointer p-1 px-1.5 absolute bottom-0 left-1/2 translate-y-2/3 -translate-x-1/2 bg-slate-100 shadow text-sm"
          onClick={() => setCollapse(true)}
        >
          <KeyboardDoubleArrowDownIcon className="!w-5 !h-5 rotate-180" />
        </span> */}
      </div>

      {loading ? (
        <div className="flex flex-col items-center justify-center mt-10">
          <h3 className="font-medium mb-2">Fetching Data...</h3>
          <CircularProgress />
        </div>
      ) : (
        <div className={`${data && "w-full h-full"}`}>
          {data && (
            <>
              <h1 className="text-2xl font-bold font-mono text-center mt-8">
                Structure Analysis for {lipid[0].name}
              </h1>
              <div className="w-full h-full border p-2 mt-6 shadow rounded">
                <div className="text-center space-x-4 mt-2">
                  <button
                    className="p-2 bg-violet-500 shadow px-3 rounded text-sm text-white"
                    onClick={() => dispatch(changeShowTable())}
                  >
                    {showTable ? "Hide" : "Show"} Table
                  </button>
                  <button
                    className="p-2 bg-violet-500 shadow px-3 rounded text-sm text-white"
                    onClick={() => setOpen(true)}
                  >
                    Actual vs Predicted
                  </button>
                </div>
                <ActualPredicted open={open} setIsOpen={setOpen} />
                <div className="w-full h-full flex items-center">
                  <ChartComponent
                    id={Date.now()}
                    graph_data={
                      data &&
                      data.predicted && [data.predicted[lipid[0].name]]
                    }
                  />
                  <div className={`${showTable ? "w-[600px] px-2" : "w-0"}`}>
                    <GraphTable />
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default MoleculeStructure;
